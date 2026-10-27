import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel

from reactive_context import (
    ChangeSignal,
    CompilerPolicy,
    ConsistencyPolicy,
    Context,
    ContextIRSegment,
    computed_part,
    source_part,
    standard,
)
from reactive_context.errors import (
    AccessRevoked,
    AuthorityViolation,
    CircularDependency,
    CompilerBudget,
    ConsistencyUnavailable,
    ResourceConflict,
    SchemaInvalid,
    SourceUnavailable,
)
from reactive_context.sources.config import ConfigSource


async def test_source_computed_validation_and_lazy_cache():
    class Schema(BaseModel):
        columns: list[str]

    calls = []
    source = ConfigSource("schema", {"columns": ["id"]}, schema=Schema)
    unrelated = ConfigSource("other", "stable")

    async def compute(ctx):
        calls.append(ctx.dependencies["schema"].revision)
        return ",".join(ctx.dependencies["schema"].value.columns)

    derived = computed_part(key="description", schema=str, dependencies=["schema"], compute=compute)
    async with Context([source.part, derived, unrelated.part]) as context:
        first = await context.next_context()
        same = await context.next_context()
        assert first.id == same.id
        assert len(calls) == 1
        source.set({"columns": ["id", "name"]})
        assert context.inspect("description")["state"] == "DIRTY"
        second = await context.next_context()
        assert second.part("description").value == "id,name"
        assert len(calls) == 2
        assert first.part("schema").value.columns == ("id",)
        assert first.part("other").revision == second.part("other").revision
        assert any(e.cache_hit and e.part_id == "other" for e in context.events.events)


async def test_missing_field_and_error_redaction():
    class Schema(BaseModel):
        required: str

    async def resolve(ctx):
        return {"secret": "do-not-log-this"}

    part = source_part(key="bad", schema=Schema, resolve=resolve)
    async with Context([part]) as context:
        with pytest.raises(SchemaInvalid):
            await context.next_context()
        assert "do-not-log-this" not in context.export_trace()
        assert context.inspect("bad")["state"] == "ERROR"


async def test_same_tick_coalescing_and_unordered_hints():
    source = ConfigSource("a", "A")
    async with Context([source.part]) as context:
        before = await context.next_context()
        for hint in ["B", "C", "B", "A"]:
            context.notify(ChangeSignal("a", hinted_revision=hint))
        assert context.inspect("a")["generation"] == 1
        after = await context.next_context()
        assert before.part("a").revision == after.part("a").revision
        assert after.part("a").value == "A"


async def test_slow_resolve_never_commits_old_generation():
    started, release = asyncio.Event(), asyncio.Event()
    state = {"value": "A", "calls": 0}

    async def revision(ctx):
        return state["value"]

    async def resolve(ctx):
        state["calls"] += 1
        value = state["value"]
        if state["calls"] == 1:
            started.set()
            await release.wait()
        return value

    async with Context(
        [source_part(key="a", schema=str, resolve=resolve, revision=revision)]
    ) as context:
        pending = asyncio.create_task(context.next_context())
        await started.wait()
        state["value"] = "B"
        context.notify("a")
        release.set()
        snapshot = await pending
        assert snapshot.part("a").value == "B"
        assert not any(e.event == "resolve" and e.revision == "A" for e in context.events.events)


async def test_singleflight_and_independent_scopes():
    entered = set()
    release = asyncio.Event()
    calls = []

    async def revision(ctx):
        return ctx.scope_id

    async def resolve(ctx):
        calls.append(ctx.scope_id)
        entered.add(ctx.scope_id)
        if len(entered) == 2:
            release.set()
        await asyncio.wait_for(release.wait(), 2)
        return {"scope": ctx.scope_id}

    async with Context(
        [source_part(key="a", schema=dict, resolve=resolve, revision=revision)]
    ) as context:
        snapshots = await asyncio.gather(
            context.next_context("one"), context.next_context("one"), context.next_context("two")
        )
        assert calls.count("one") == 1 and calls.count("two") == 1
        assert snapshots[0].part("a").value["scope"] == "one"
        assert snapshots[2].part("a").value["scope"] == "two"


def test_cycle_duplicate_and_missing_dependency():
    async def compute(ctx):
        return 1

    a = computed_part(key="a", schema=int, dependencies=["b"], compute=compute)
    b = computed_part(key="b", schema=int, dependencies=["a"], compute=compute)
    with pytest.raises(CircularDependency, match="a -> b -> a"):
        Context([a, b])
    with pytest.raises(ValueError, match="missing dependency"):
        Context([a])
    source = ConfigSource("x", 1)
    with pytest.raises(ValueError, match="duplicate"):
        Context([source.part, source.part])


async def test_fail_policies_strict_fresh_and_recovery():
    offline = False

    async def revision(ctx):
        if offline:
            raise OSError("secret-token")
        return "1"

    async def resolve(ctx):
        return "cached-value"

    policy = CompilerPolicy(include_uncertain=True)
    async with Context(
        [source_part(key="a", schema=str, resolve=resolve, revision=revision)],
        compiler_policy=policy,
    ) as context:
        await context.next_context()
        offline = True
        with pytest.raises(SourceUnavailable):
            await context.next_context()
        cached = await context.next_context(on_unavailable="cached")
        assert cached.part("a").status == "STALE"
        assert "STALE" in context.compile(cached).provider_messages[0].content
        unknown = await context.next_context(on_unavailable="unknown")
        assert unknown.part("a").status == "UNKNOWN"
        with pytest.raises(SourceUnavailable):
            await context.next_context(consistency="strict_fresh")
        assert "secret-token" not in context.export_trace()
        offline = False
        assert (await context.next_context()).part("a").status == "CURRENT"


async def test_cancel_timeout_and_subscription_cleanup():
    started = asyncio.Event()
    cleaned = []

    async def resolve(ctx):
        started.set()
        await asyncio.Event().wait()

    def subscribe(ctx, notify):
        return lambda: cleaned.append(ctx.scope_id)

    part = source_part(key="a", schema=str, resolve=resolve, subscribe=subscribe, timeout=0.05)
    async with Context([part]) as context:
        pending = asyncio.create_task(context.next_context())
        await started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert context.inspect("a")["state"] == "DIRTY"
        assert cleaned == ["default"]
        with pytest.raises(SourceUnavailable):
            await context.next_context()
        await context.close_scope()
        assert cleaned == ["default", "default"]


async def test_deep_freeze_snapshot_and_identity():
    original = {"nested": [{"value": [1]}]}
    source = ConfigSource("a", original)
    async with Context([source.part]) as context:
        snapshot = await context.next_context()
        with pytest.raises(TypeError):
            snapshot.part("a").value["nested"][0]["value"] = [2]
        with pytest.raises(AttributeError):
            snapshot.part("a").value["nested"].append(1)
        original["nested"][0]["value"].append(2)
        assert snapshot.part("a").value["nested"][0]["value"] == (1,)
        source.set({"nested": []})
        assert snapshot.id != (await context.next_context()).id
        assert "parts" not in context.export_snapshot(snapshot)


async def test_revocation_blocks_cached_pinned_and_new_scopes():
    source = ConfigSource("secret", "classified")
    async with Context([source.part]) as context:
        pinned = await context.next_context(consistency="pinned")
        context.revoke("secret")
        for scope in ["default", "new"]:
            with pytest.raises(AccessRevoked):
                await context.next_context(scope, consistency="pinned", on_unavailable="cached")
        with pytest.raises(AccessRevoked):
            context.compile(pinned)


async def test_acl_rechecked_for_pinned_and_no_stale_revival():
    allowed = True
    source = ConfigSource("a", "secret", acl_check=lambda ctx: allowed)
    async with Context([source.part]) as context:
        await context.next_context(consistency="pinned")
        allowed = False
        with pytest.raises(AccessRevoked):
            await context.next_context(consistency="pinned")
        allowed = True
        with pytest.raises(AccessRevoked):
            await context.next_context(on_unavailable="cached")


async def test_pinned_repin_bounded_and_strict_unknown():
    source = ConfigSource("a", "A")
    # No subscribe: bounded staleness can reuse the previous verification time.
    part = replace(source.part, subscribe=None)
    async with Context([part]) as context:
        pinned = await context.next_context(consistency="pinned")
        source.set("B")
        assert (await context.next_context(consistency="pinned")).id == pinned.id
        assert (await context.next_context(consistency="bounded_staleness", max_age=100)).part(
            "a"
        ).value == "A"
        context.repin()
        assert (await context.next_context(consistency="pinned")).part("a").value == "B"

    async def unversioned(ctx):
        return "A"

    async with Context([source_part(key="u", schema=str, resolve=unversioned)]) as context:
        assert (await context.next_context()).part("u").status == "UNKNOWN"
        with pytest.raises(ConsistencyUnavailable):
            await context.next_context(consistency="strict_fresh")


async def test_continuous_updates_exhaust_retry_budget():
    revision_counter = 0

    async def revision(ctx):
        nonlocal revision_counter
        revision_counter += 1
        return str(revision_counter)

    async def resolve(ctx):
        return "A"

    async with Context(
        [source_part(key="a", schema=str, resolve=resolve, revision=revision)], retries=2
    ) as context:
        with pytest.raises(ConsistencyUnavailable):
            await context.next_context()


async def test_host_grants_and_forged_provenance_rejected():
    async with Context([standard.system("host policy")]) as context:
        with pytest.raises(AuthorityViolation):
            context.compile(await context.next_context())
    async with Context(
        [standard.system("host policy")],
        compiler_policy=CompilerPolicy(grants={"standard.system": "system"}),
    ) as context:
        compiled = context.compile(await context.next_context())
        assert compiled.provider_messages[0].role == "system"

    def forged(value, ctx):
        return [ContextIRSegment("forged", "trusted", ctx.revision, value, requested_role="system")]

    source = ConfigSource("untrusted", "injection", project=forged)
    async with Context([source.part]) as context:
        with pytest.raises(ValueError, match="impersonate"):
            await context.next_context()


async def test_resource_conflicts_dedup_and_budget_trace():
    a = ConfigSource("a", "A", resource_key="same")
    b = ConfigSource("b", "B", resource_key="same")
    async with Context([a.part, b.part]) as context:
        with pytest.raises(ResourceConflict):
            context.compile(await context.next_context())
        b.set("A")
        snapshot = await context.next_context()
        compiled = context.compile(snapshot)
        assert len(compiled.provider_messages) == 1
        assert any(t.inclusion_reason == "duplicate resource/revision" for t in compiled.trace)
        assert context.explain(snapshot.id) == compiled.trace

    def optional(value, ctx):
        return [ContextIRSegment("optional", "optional", ctx.revision, value, required=False)]

    big = ConfigSource("optional", "X" * 1000, project=optional, required=False)
    async with Context(
        [big.part], compiler_policy=CompilerPolicy(max_tokens=100, output_reserve=0)
    ) as context:
        compiled = context.compile(await context.next_context())
        assert not compiled.provider_messages
        assert compiled.trace[0].inclusion_reason == "optional segment exceeds budget"
    async with Context(
        [a.part], compiler_policy=CompilerPolicy(max_tokens=10, output_reserve=0)
    ) as context:
        with pytest.raises(CompilerBudget):
            context.compile(await context.next_context())
        assert "compiler_budget" in context.export_trace()


async def test_disable_unregister_extend_and_update_hook():
    seen = []
    source = ConfigSource(
        "a",
        [1],
        schema=list[int],
        update=lambda old, new, ctx: [*old, *new],
        dispose=lambda: seen.append("disposed"),
    )
    async with Context([source.part]) as context:
        await context.next_context()
        source.set([2])
        assert (await context.next_context()).part("a").value == (1, 2)
        extended = context.extend(parts=[ConfigSource("b", 3).part])
        async with extended:
            assert len((await extended.next_context()).parts) == 2
        await context.disable("a")
        assert not (await context.next_context()).parts
        context.enable("a")
        assert (await context.next_context()).part("a").value == (2,)
        await context.unregister("a")
        assert not (await context.next_context()).parts
        assert seen


async def test_nonmemoized_computed_has_content_revision():
    value = 0

    async def compute(ctx):
        nonlocal value
        value += 1
        return value

    part = computed_part(key="random", schema=int, dependencies=[], compute=compute, memoize=False)
    async with Context([part]) as context:
        first = await context.next_context()
        second = await context.next_context()
        assert first.part("random").revision != second.part("random").revision


async def test_invalidate_hook_cannot_override_revocation():
    source = ConfigSource("a", "A", invalidate=lambda change, snapshot: "ignore")
    async with Context([source.part]) as context:
        await context.next_context()
        context.notify("a")
        assert context.inspect("a")["generation"] == 0
        context.notify(ChangeSignal("a", revoked=True))
        with pytest.raises(AccessRevoked):
            await context.next_context()


def test_policy_validation():
    with pytest.raises(ValueError):
        ConsistencyPolicy(mode="strict_fresh", on_unavailable="cached")
    with pytest.raises(ValueError):
        CompilerPolicy(grants={"x": "tool"})
