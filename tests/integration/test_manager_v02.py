from dataclasses import replace

import pytest
from pydantic import BaseModel

from reactive_context import CompilerPolicy, Context, ContextIRSegment
from reactive_context.core.part import source_part
from reactive_context.errors import AccessRevoked, AuthorityViolation, ContextBudgetExceeded
from reactive_context.manager import ContextManager
from reactive_context.sources.config import ConfigSource
from reactive_context.store import ContextStore, Principal, VersionConflict

HOST = Principal("host", admin=True)
USER = Principal("agent")


class Note(BaseModel):
    text: str


@pytest.fixture
def setup(tmp_path):
    with ContextStore(tmp_path) as store:
        store.register_schema("note", Note)
        for scope in ("a", "b"):
            store.create_scope(HOST, scope)
            store.grant_scope(HOST, scope, USER.id)
        yield store


async def test_query_load_activate_are_separate_and_scopes_do_not_share_state(setup):
    skill = ConfigSource("skill", "release-one")
    async with ContextManager([skill.part], store=setup, principal=USER) as manager:
        manager.publish_catalog(HOST, "a")
        manager.publish_catalog(HOST, "b")
        row = (await manager.query("a", "skill"))[0]
        loaded = await manager.load("a", row["id"])
        assert loaded.snapshot.value == "release-one"
        assert not (await manager.next_context("a")).parts
        await manager.activate("a", row["id"])
        assert (await manager.next_context("a")).part("skill").value == "release-one"
        assert not (await manager.next_context("b")).parts
        skill.set("release-two")
        assert (await manager.next_context("a")).part("skill").value == "release-two"
        await manager.deactivate("a", row["id"])
        assert not (await manager.next_context("a")).parts


async def test_active_refs_restore_after_context_restart_and_resource_rollback(setup):
    source = ConfigSource("schema", "old")
    async with ContextManager([source.part], store=setup, principal=USER) as manager:
        manager.publish_catalog(HOST, "a")
        row = (await manager.query("a"))[0]
        await manager.activate("a", row["id"])
    source.set("new")
    async with ContextManager([source.part], store=setup, principal=USER) as manager:
        current = await manager.next_context("a")
        assert current.part("schema").value == "new"
        source.set("old")
        rollback = await manager.next_context("a")
        assert rollback.part("schema").value == "old"
        assert rollback.part("schema").revision != current.part("schema").revision


async def test_pinned_resource_never_silently_switches_on_restart(setup):
    source = ConfigSource("rules", "one")
    async with ContextManager([source.part], store=setup, principal=USER) as manager:
        manager.publish_catalog(HOST, "a")
        row = (await manager.query("a"))[0]
        await manager.activate("a", row["id"], policy="pinned")
        source.set("two")
        assert (await manager.next_context("a")).part("rules").value == "one"
    async with ContextManager([source.part], store=setup, principal=USER) as manager:
        with pytest.raises(VersionConflict):
            await manager.next_context("a")


async def test_fact_payload_not_retained_by_active_set_and_compile_rechecks_acl(setup):
    evidence = setup.observe_evidence(USER, "a", "source", "1")
    fact = setup.commit(
        USER, setup.propose_fact(USER, "a", "note", "fact", {"text": "data"}, [evidence])
    )
    async with ContextManager(store=setup, principal=USER) as manager:
        await manager.activate("a", fact.id)
        snapshot = await manager.next_context("a")
        compiled = manager.compile(snapshot)
        assert compiled.compile_id != manager.compile(snapshot).compile_id
        assert dict(compiled.revision_vector) == {"fact." + fact.id: "1"}
        assert all(p.role == "user" for p in compiled.provider_messages)
        assert manager.registry.instance("a", "fact." + fact.id).snapshot is None
        setup.revoke_part_access(HOST, fact.id, USER.id)
        with pytest.raises(AccessRevoked):
            manager.compile(snapshot)


async def test_oversized_resource_returns_structured_budget_and_rolls_back_activation(setup):
    source = ConfigSource("large", "x" * 3000)
    async with ContextManager(
        [source.part],
        store=setup,
        principal=USER,
        compiler_policy=CompilerPolicy(max_tokens=1500, output_reserve=100),
    ) as manager:
        manager.publish_catalog(HOST, "a")
        row = (await manager.query("a"))[0]
        with pytest.raises(ContextBudgetExceeded) as error:
            await manager.activate("a", row["id"])
        assert error.value.parts == ("large",)
        assert error.value.suggested_scopes
        assert setup.active_refs(USER, "a") == []


async def test_fact_cannot_be_promoted_even_if_host_misconfigures_role_grant():
    async def resolve(ctx):
        return "agent-created instruction"

    def project(value, ctx):
        return [
            ContextIRSegment(
                "fact", "fact", ctx.revision, value, requested_role="system", kind="instruction"
            )
        ]

    part = source_part(key="fact", schema=str, resolve=resolve, project=project, kind="fact")
    async with Context(
        [part], compiler_policy=CompilerPolicy(grants={"fact": "system"})
    ) as context:
        with pytest.raises(AuthorityViolation):
            await context.next_context()


async def test_unauthorized_dependency_is_checked_before_resolve(setup):
    calls = []
    source = ConfigSource("private", "secret")
    from reactive_context import computed_part

    async def compute(ctx):
        calls.append("computed")
        return "derived"

    part = computed_part(key="plan", schema=str, dependencies=["private"], compute=compute)
    async with ContextManager([source.part, part], store=setup, principal=USER) as manager:
        manager.publish_catalog(HOST, "a")
        rows = {r["key"]: r for r in await manager.query("a")}
        setup.revoke_part_access(HOST, rows["private"]["id"], USER.id)
        with pytest.raises(AccessRevoked):
            await manager.load("a", rows["plan"]["id"])
        assert calls == []


async def test_bounded_staleness_policy_and_dirty_notification(setup):
    source = ConfigSource("data", "one")
    async with ContextManager(
        [replace(source.part, subscribe=None)], store=setup, principal=USER
    ) as manager:
        manager.publish_catalog(HOST, "a")
        row = (await manager.query("a"))[0]
        await manager.activate("a", row["id"], policy="bounded_staleness", max_age=60)
        source.set("two")
        assert (await manager.next_context("a")).part("data").value == "one"
        manager.notify("data", scope="a")
        assert (await manager.next_context("a")).part("data").value == "two"


async def test_live_snapshot_leases_prevent_revision_gc_until_snapshot_release(setup):
    import gc

    evidence = setup.observe_evidence(USER, "a", "source", "1")
    first = setup.commit(
        USER, setup.propose_fact(USER, "a", "note", "key", {"text": "old"}, [evidence])
    )
    async with ContextManager(store=setup, principal=USER) as manager:
        await manager.activate("a", first.id)
        snapshot = await manager.next_context("a")
        setup.commit(
            USER,
            setup.propose_fact(
                USER, "a", "note", "key", {"text": "new"}, [evidence], expected_revision=1
            ),
        )
        setup.gc(HOST, prune_revisions=True)
        assert setup.load(USER, first.id, revision=1).revision == 1
        del snapshot
        gc.collect()
        setup.gc(HOST, prune_revisions=True)
        with pytest.raises(VersionConflict):
            setup.load(USER, first.id, revision=1)


async def test_release_revision_changes_when_old_content_is_republished(setup):
    source = ConfigSource("rules", "one", revision_mode="release")
    async with Context([source.part]) as context:
        first = await context.next_context()
        source.set("two")
        await context.next_context()
        source.set("one")
        rollback = await context.next_context()
        assert rollback.part("rules").value == first.part("rules").value
        assert rollback.part("rules").revision != first.part("rules").revision


async def test_partial_snapshot_lease_failure_releases_acquired_leases(setup, monkeypatch):
    evidence = setup.observe_evidence(USER, "a", "source", "1")
    facts = [
        setup.commit(
            USER, setup.propose_fact(USER, "a", "note", str(i), {"text": str(i)}, [evidence])
        )
        for i in range(2)
    ]
    async with ContextManager(store=setup, principal=USER) as manager:
        for fact in facts:
            setup.activate(USER, "a", fact.id)
        original = setup.lease
        calls = 0

        def lease(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise AccessRevoked(detail="concurrent revocation")
            return original(*args, **kwargs)

        monkeypatch.setattr(setup, "lease", lease)
        with pytest.raises(AccessRevoked):
            await manager.next_context("a")
        assert setup.db.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0
        assert all(
            manager.registry.instance("a", "fact." + fact.id).snapshot is None for fact in facts
        )


async def test_reading_inactive_facts_does_not_accumulate_registry_entries(setup):
    evidence = setup.observe_evidence(USER, "a", "source", "1")
    facts = [
        setup.commit(
            USER, setup.propose_fact(USER, "a", "note", str(i), {"text": str(i)}, [evidence])
        )
        for i in range(100)
    ]
    async with ContextManager(store=setup, principal=USER) as manager:
        for fact in facts:
            loaded = await manager.load("a", fact.id)
            assert loaded.snapshot.value["value"]["text"] == fact.value["text"]
            assert loaded.snapshot.provenance[1] == ("kind", "fact")
        assert not manager.registry.definitions
        assert not manager.registry.instances
        assert not setup.active_refs(USER, "a")
        await manager.activate("a", facts[-1].id)
        assert (await manager.next_context("a")).part("fact." + facts[-1].id).value[
            "verification"
        ] == "unverified"
