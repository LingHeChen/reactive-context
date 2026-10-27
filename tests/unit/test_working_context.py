import asyncio
from dataclasses import replace

import pytest

from reactive_context import (
    ContextManager,
    ContextStore,
    HistoryGroup,
    HistoryManager,
    HistoryPolicy,
    Objective,
    ObjectivePolicy,
    Principal,
    SummaryDraft,
    WorkObjective,
)
from reactive_context.errors import AccessRevoked
from reactive_context.history import HistoryBudgetExceeded
from reactive_context.objective import ObjectiveUpdateRejected
from reactive_context.sources.config import ConfigSource
from reactive_context.store import VersionConflict

HOST, USER = Principal("host", True), Principal("agent")


@pytest.fixture
def store(tmp_path):
    with ContextStore(tmp_path) as value:
        for scope in ("chat", "other"):
            value.create_scope(HOST, scope)
            value.grant_scope(HOST, scope, USER.id)
        yield value


def groups(count=12):
    return tuple(
        HistoryGroup(({"id": i, "content": "past work " + "x" * 1000},)) for i in range(count)
    )


def compact(events):
    return SummaryDraft(
        text="ARCHIVED_ONLY completed database migration 工作记录",
        evidence_ids=[HistoryManager.event_id(events[0])],
    )


async def test_default_short_history_never_implicitly_calls_summary_or_validator(store):
    calls = []
    history = HistoryManager(
        store,
        USER,
        compactor=lambda events: calls.append(1),
        validator=lambda *_: calls.append(2),
        policy=HistoryPolicy(recent_groups=2),
    )
    raw = groups()
    view = await history.prepare("chat", raw, total_tokens=5000)
    assert view.raw_groups == raw[-2:]
    assert not view.summaries and not calls
    assert len(history.retrieve("chat", [history.event_id(e) for g in raw for e in g.events])) == 12


async def test_archived_summaries_are_searchable_but_not_projected(store):
    history = HistoryManager(
        store,
        USER,
        compactor=compact,
        validator=lambda *_: True,
        policy=HistoryPolicy(recent_groups=2),
    )
    raw = groups()
    identity = await history.archive("chat", raw)
    hits = history.search("chat", "migration")
    assert hits[0]["id"] == identity and hits[0]["authority"] == "historical_evidence"
    assert history.search("chat", "工作")
    assert not history.search("other", "migration")
    view = await history.prepare("chat", raw, total_tokens=5000)
    assert not view.summaries and view.raw_groups == raw[-2:]
    history.request_compaction("chat")
    await history.prepare("chat", raw, total_tokens=5000)
    assert store.db.execute("SELECT COUNT(*) FROM compaction_records").fetchone()[0] == 2


async def test_history_search_and_compaction_respect_source_revocation(store):
    store.register_schema("note", dict)
    evidence = store.observe_evidence(USER, "chat", "source", "1")
    fact = store.commit(
        USER, store.propose_fact(USER, "chat", "note", "one", {"text": "data"}, [evidence])
    )
    history = HistoryManager(
        store,
        USER,
        compactor=compact,
        validator=lambda *_: True,
        policy=HistoryPolicy(recent_groups=1),
    )
    raw = tuple(HistoryGroup(g.events, revisions={fact.id: "1"}) for g in groups(4))
    await history.archive("chat", raw)
    assert history.search("chat", "migration")
    store.revoke_part_access(HOST, fact.id, USER.id)
    assert not history.search("chat", "migration")
    with pytest.raises(AccessRevoked):
        await history.archive("chat", raw + groups(2))


async def test_history_reindex_migrates_existing_archives_in_bounded_pages(store):
    history = HistoryManager(store, USER)
    history.observe("chat", groups(5))
    store.db.execute("DELETE FROM history_search")
    store.db.execute("DELETE FROM history_search_entries")
    cursor = history.reindex("chat", limit=2)
    assert cursor == 2
    cursor = history.reindex("chat", after_seq=cursor, limit=2)
    assert cursor == 4
    assert history.reindex("chat", after_seq=cursor, limit=2) == 5
    assert len(history.search("chat", "past", kind="event")) == 5


async def test_manager_checks_external_acl_before_search_or_archive_model_calls(store):
    allowed = True
    source = ConfigSource("schema", "private", acl_check=lambda ctx: allowed)
    history = HistoryManager(
        store,
        USER,
        compactor=compact,
        validator=lambda *_: True,
        policy=HistoryPolicy(recent_groups=1),
    )
    async with ContextManager(
        [source.part], store=store, principal=USER, history=history
    ) as manager:
        manager.publish_catalog(HOST, "chat")
        raw = tuple(HistoryGroup(g.events, revisions={"schema": "1"}) for g in groups(5))
        await manager.compact_history("chat", raw)
        assert await manager.search_history("chat", "migration")
        allowed = False
        assert not await manager.search_history("chat", "migration")
        with pytest.raises(AccessRevoked):
            await manager.load_history("chat", [history.event_id(raw[0].events[0])])


async def test_objective_versions_restart_scope_permissions_and_user_role(store):
    objective = Objective(store, USER)
    assert objective.read("chat") is None
    first = objective.update("chat", {"goal": "Ship report"}, expected_revision=0)
    assert first["revision"] == 1
    with pytest.raises(VersionConflict):
        objective.update_from_agent("chat", {"goal": "stale write"}, expected_revision=0)
    async with ContextManager(store=store, principal=USER, objective=objective) as manager:
        snapshot = await manager.next_context("chat")
        assert "Ship report" in str(manager.compile(snapshot).provider_messages)
        assert all(m.role == "user" for m in manager.compile(snapshot).provider_messages)
        objective.update_from_agent("chat", {"goal": "Ship revised report"}, expected_revision=1)
        assert snapshot.part(objective.key).value["objective"]["goal"] == "Ship report"
        assert (await manager.next_context("chat")).part(objective.key).value["revision"] == 2
    restored = Objective(store, USER)
    assert restored.read("chat")["revision"] == 2
    assert restored.read("chat", revision=1)["objective"]["goal"] == "Ship report"
    assert restored.read("other") is None
    store.revoke_scope(HOST, "chat", USER.id)
    with pytest.raises(AccessRevoked):
        restored.read("chat")


async def test_threshold_updates_are_opt_in_bounded_deduplicated_and_can_use_decider(store):
    calls = []

    def updater(request):
        calls.append(request)
        return WorkObjective(goal="Updated next step")

    objective = Objective(
        store,
        USER,
        updater=updater,
        policy=ObjectivePolicy(update_mode="threshold", min_update_interval=0),
    )
    raw = groups()
    assert (
        await objective.refresh_at_threshold("chat", raw, used_tokens=8000, total_tokens=10000)
        is None
    )
    await objective.refresh_at_threshold("chat", raw, used_tokens=8001, total_tokens=10000)
    assert len(calls) == 1 and len(calls[0].events) <= 8
    await objective.refresh_at_threshold("chat", raw, used_tokens=9000, total_tokens=10000)
    assert len(calls) == 1
    rejected = Objective(
        store,
        USER,
        key="other.objective",
        updater=updater,
        decider=lambda request: False,
        policy=replace(objective.policy),
    )
    assert (
        await rejected.refresh_at_threshold("chat", raw, used_tokens=9000, total_tokens=10000)
        is None
    )
    assert len(calls) == 1
    manual = Objective(store, USER, key="manual", updater=updater)
    assert (
        await manual.refresh_at_threshold("chat", raw, used_tokens=9999, total_tokens=10000) is None
    )
    assert len(calls) == 1


async def test_async_updater_cannot_overwrite_explicit_concurrent_change(store):
    started, finish = asyncio.Event(), asyncio.Event()

    async def updater(request):
        started.set()
        await finish.wait()
        return {"goal": "Outdated model update"}

    objective = Objective(
        store, USER, updater=updater, policy=ObjectivePolicy(min_update_interval=0)
    )
    objective.update("chat", {"goal": "Original"})
    pending = asyncio.create_task(objective.refresh("chat", groups(2)))
    await started.wait()
    objective.update("chat", {"goal": "User changed goal"}, expected_revision=1)
    finish.set()
    with pytest.raises(VersionConflict):
        await pending
    assert objective.read("chat")["objective"]["goal"] == "User changed goal"


async def test_agent_mode_is_exclusive_and_host_can_disable_agent_writes(store):
    with pytest.raises(ValueError):
        ObjectivePolicy(update_mode="agent", allow_agent_updates=False)
    objective = Objective(store, USER, policy=ObjectivePolicy(allow_agent_updates=False))
    with pytest.raises(ObjectiveUpdateRejected):
        objective.update_from_agent("chat", {"goal": "Unauthorized"}, expected_revision=0)
    agent = Objective(store, USER, policy=ObjectivePolicy(update_mode="agent"))
    assert agent.read("chat")["maintenance"]["expected_revision"] == 0
    assert not agent.policy.auto_update


async def test_recent_protected_history_is_never_silently_dropped(store):
    history = HistoryManager(store, USER, policy=HistoryPolicy(recent_groups=1))
    raw = (HistoryGroup(({"content": "x" * 6000},), protected=True), *groups(1))
    with pytest.raises(HistoryBudgetExceeded):
        await history.prepare("chat", raw, total_tokens=5000)


async def test_agent_noop_keeps_goal_revision_and_policy_changes_refresh_hints(store):
    objective = Objective(store, USER, policy=ObjectivePolicy(update_mode="agent"))
    objective.update("chat", {"goal": "Keep this goal"})
    assert (
        objective.update_from_agent("chat", {"goal": "Keep this goal"}, expected_revision=1)[
            "revision"
        ]
        == 1
    )
    async with ContextManager(store=store, principal=USER, objective=objective) as manager:
        first = await manager.next_context("chat")
        objective.policy = replace(objective.policy, update_conditions=("A NEW HOST CONDITION",))
        second = await manager.next_context("chat")
        assert first.part(objective.key).revision != second.part(objective.key).revision
        assert "A NEW HOST CONDITION" in str(manager.compile(second).provider_messages)


async def test_forgetting_fact_removes_dependent_summary_search_payload(store):
    store.register_schema("note", dict)
    evidence = store.observe_evidence(USER, "chat", "source", "1")
    fact = store.commit(
        USER, store.propose_fact(USER, "chat", "note", "one", {"text": "data"}, [evidence])
    )
    history = HistoryManager(
        store,
        USER,
        compactor=compact,
        validator=lambda *_: True,
        policy=HistoryPolicy(recent_groups=1),
    )
    raw = tuple(HistoryGroup(g.events, revisions={fact.id: "1"}) for g in groups(4))
    identity = await history.archive("chat", raw)
    assert history.search("chat", "migration")
    store.retire(USER, fact.id, status="forgotten")
    assert not history.search("chat", "migration")
    assert not store.db.execute(
        "SELECT 1 FROM history_search_entries WHERE record_id=?", (identity,)
    ).fetchone()


async def test_host_can_update_goal_from_worker_thread(store):
    objective = Objective(store, USER)
    objective.update("chat", {"goal": "Original"})
    async with ContextManager(store=store, principal=USER, objective=objective) as manager:
        await manager.next_context("chat")
        await asyncio.to_thread(
            objective.update, "chat", {"goal": "Worker update"}, expected_revision=1
        )
        frame = await manager.next_context("chat")
        assert frame.part(objective.key).value["objective"]["goal"] == "Worker update"
