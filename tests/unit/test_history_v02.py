import asyncio

import pytest

from reactive_context.history import (
    CompactionRejected,
    HistoryBudgetExceeded,
    HistoryGroup,
    HistoryManager,
    HistoryPolicy,
    SummaryDraft,
)
from reactive_context.store import ContextStore, Principal

HOST = Principal("host", admin=True)


@pytest.fixture
def store(tmp_path):
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        yield store


def groups(count, *, size=1000, revisions=None):
    return tuple(
        HistoryGroup(
            ({"kind": "fixture", "id": i, "content": "x" * size},), revisions=revisions or {}
        )
        for i in range(count)
    )


def summarize(events):
    ids = []
    for event in events:
        if event.get("kind") == "historical_summary":
            ids.extend(event["summary"]["evidence_ids"])
        else:
            ids.append(HistoryManager.event_id(event))
    return SummaryDraft(text="Task progress preserved", evidence_ids=ids[:1])


def manager(store, **kwargs):
    return HistoryManager(
        store,
        HOST,
        compactor=summarize,
        validator=lambda source, draft: True,
        policy=HistoryPolicy(recent_groups=2, segment_groups=3, max_compactions_per_boundary=20),
        **kwargs,
    )


async def test_incremental_summaries_reuse_cache_and_archive_is_reversible(store):
    calls = []

    def compact(events):
        calls.append(events)
        return summarize(events)

    history = manager(store)
    history.compactor = compact
    original = groups(10)
    result = await history.project("chat", original, max_tokens=4000)
    assert result.compacted_groups > 0
    count = len(calls)
    again = await history.project("chat", original, max_tokens=4000)
    assert len(calls) == count and again.compacted_groups == 0
    ids = [history.event_id(e) for g in original for e in g.events]
    assert history.retrieve("chat", ids) == [e for g in original for e in g.events]
    assert store.db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] == 10
    assert all(r["validation"] == "host_semantic_validator_passed" for r in result.summaries)


async def test_summary_rollup_keeps_long_history_projection_bounded(store):
    history = manager(store)
    for count in (10, 20, 40):
        projection = await history.project("chat", groups(count), max_tokens=5000)
        assert projection.estimated_tokens <= 5000
    assert (
        store.db.execute(
            "SELECT COUNT(*) FROM compaction_records WHERE status='absorbed'"
        ).fetchone()[0]
        > 0
    )
    assert store.db.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] == 40
    ids = [history.event_id(e) for g in groups(40) for e in g.events]
    assert len(history.retrieve("chat", ids)) == 40


async def test_changed_resource_withdraws_summary_body_but_preserves_original_archive(store):
    history = manager(store)
    original = groups(10, revisions={"rules": "1"})
    first = await history.project("chat", original, max_tokens=4000, revisions={"rules": "1"})
    assert first.summaries
    second = await history.project("chat", original, max_tokens=4000, revisions={"rules": "2"})
    assert all(r["summary"] is None for r in second.summaries)
    assert all(r["status"] == "WITHDRAWN_SOURCE_CHANGED" for r in second.summaries)
    event_id = history.event_id(original[0].events[0])
    assert history.retrieve("chat", [event_id])[0]["content"] == "x" * 1000


async def test_required_constraints_and_fabricated_evidence_fail_closed(store):
    history = manager(store)
    history.policy = HistoryPolicy(recent_groups=1, required_constraints=("Do not deploy",))
    with pytest.raises(CompactionRejected, match="constraint"):
        await history.project("chat", groups(8), max_tokens=2000)
    history.policy = HistoryPolicy(recent_groups=1)
    history.compactor = lambda events: SummaryDraft(text="bad", evidence_ids=["invented"])
    with pytest.raises(CompactionRejected, match="evidence"):
        await history.project("chat", groups(8), max_tokens=2000)
    assert store.db.execute("SELECT COUNT(*) FROM compaction_records").fetchone()[0] == 0


async def test_semantic_validation_is_required_unless_host_explicitly_opts_out(store):
    history = manager(store)
    history.validator = None
    with pytest.raises(CompactionRejected, match="semantic validator"):
        await history.project("chat", groups(8), max_tokens=2000)
    history.validator = lambda source, draft: False
    with pytest.raises(CompactionRejected, match="rejected"):
        await history.project("chat", groups(8), max_tokens=2000)


async def test_protected_protocol_groups_are_not_silently_deleted(store):
    history = manager(store)
    protected = tuple(HistoryGroup(g.events, True) for g in groups(4))
    with pytest.raises(HistoryBudgetExceeded):
        await history.project("chat", protected, max_tokens=1000)
    assert store.db.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] == 4
    assert store.db.execute("SELECT COUNT(*) FROM compaction_records").fetchone()[0] == 0


async def test_cancelled_compactor_never_commits_partial_summary(store):
    entered = asyncio.Event()
    history = manager(store)

    async def pause(events):
        entered.set()
        await asyncio.Event().wait()

    history.compactor = pause
    task = asyncio.create_task(history.project("chat", groups(8), max_tokens=2000))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.db.execute("SELECT COUNT(*) FROM compaction_records").fetchone()[0] == 0
    history.compactor = summarize
    assert (await history.project("chat", groups(8), max_tokens=4000)).summaries


async def test_host_compaction_ownership_blocks_double_compaction(store):
    history = manager(store)
    history.policy = HistoryPolicy(host_compaction=True)
    with pytest.raises(CompactionRejected, match="host owns"):
        history.request_compaction("chat")
    with pytest.raises(HistoryBudgetExceeded, match="host must"):
        await history.project("chat", groups(8), max_tokens=1000)


async def test_summary_failure_rolls_back_record_and_keeps_archive(store, monkeypatch):
    history = manager(store)

    def fail(*args, **kwargs):
        raise OSError("simulated commit failure")

    monkeypatch.setattr(store, "_emit", fail)
    with pytest.raises(OSError):
        await history.project("chat", groups(8), max_tokens=2000)
    assert store.db.execute("SELECT COUNT(*) FROM compaction_records").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] == 8


async def test_scope_access_is_checked_on_archive_retrieval(store):
    history = manager(store)
    await history.project("chat", groups(2), max_tokens=10000)
    outsider = HistoryManager(store, Principal("outsider"))
    with pytest.raises(Exception, match="access_revoked"):
        outsider.retrieve("chat", [history.event_id(groups(2)[0].events[0])])


async def test_large_history_objects_survive_gc_and_detect_corruption(store):
    from reactive_context.store import StoreCorruption

    history = manager(store)
    source = groups(2, size=10000)
    await history.project("chat", source, max_tokens=100000)
    ids = [history.event_id(g.events[0]) for g in source]
    assert store.gc(HOST)["orphan_objects_removed"] == 0
    assert len(history.retrieve("chat", ids)[0]["content"]) == 10000
    row = store.db.execute("SELECT object_path FROM history_events LIMIT 1").fetchone()
    (store.directory / row[0]).write_bytes(b"damaged")
    with pytest.raises(StoreCorruption):
        history.retrieve("chat", ids)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_context_ratio_rejects_invalid_values(ratio):
    with pytest.raises(ValueError):
        HistoryPolicy(context_ratio=ratio)


def test_history_limits_count_reservations_against_total_context():
    policy = HistoryPolicy(context_ratio=0.8)
    assert policy.history_limits(128000, 10000) == (118000, 92400)
    assert policy.history_limits(256000, 10000) == (246000, 194800)
    assert policy.history_limits(10000, 9000) == (1000, 0)


@pytest.mark.parametrize("history_size,expected_calls", [(7000, 0), (7001, 1)])
async def test_summary_and_validation_start_only_above_total_context_ratio(
    store, history_size, expected_calls
):
    calls = []

    def compact(events):
        calls.append("summary")
        return summarize(events)

    def validate(events, draft):
        calls.append("validation")
        return True

    history = HistoryManager(
        store,
        HOST,
        compactor=compact,
        validator=validate,
        policy=HistoryPolicy(context_ratio=0.8, recent_groups=1),
    )
    raw = (
        HistoryGroup(({"id": 1, "tokens": history_size - 1000},)),
        HistoryGroup(({"id": 2, "tokens": 1000},)),
    )

    def measure(summaries, raw):
        return len(summaries) * 500 + sum(e["tokens"] for g in raw for e in g.events)

    await history.project("chat", raw, total_tokens=10000, reserved_tokens=1000, measure=measure)
    assert calls == ["summary", "validation"] * expected_calls


async def test_same_history_does_not_compact_in_larger_context_window(store):
    store.create_scope(HOST, "large")
    calls = []

    def compact(events):
        calls.append(1)
        return summarize(events)

    history = HistoryManager(
        store,
        HOST,
        compactor=compact,
        validator=lambda events, draft: True,
        policy=HistoryPolicy(context_ratio=0.8, recent_groups=1, segment_groups=3),
    )
    raw = tuple(HistoryGroup(({"id": i, "tokens": 2500},)) for i in range(4))

    def measure(summaries, raw):
        return len(summaries) * 500 + sum(e["tokens"] for g in raw for e in g.events)

    await history.project("chat", raw, total_tokens=10000, reserved_tokens=1000, measure=measure)
    assert len(calls) == 1
    larger = await history.project(
        "large", raw, total_tokens=20000, reserved_tokens=1000, measure=measure
    )
    assert len(calls) == 1
    assert larger.compacted_groups == 0


async def test_protected_history_can_exceed_soft_trigger_but_not_hard_limit(store):
    history = HistoryManager(store, HOST, policy=HistoryPolicy(context_ratio=0.8))
    raw = (HistoryGroup(({"id": 1},), protected=True),)
    result = await history.project(
        "chat", raw, total_tokens=10000, reserved_tokens=3000, measure=lambda *_: 6000
    )
    assert result.estimated_tokens == 6000 and not result.summaries
    with pytest.raises(HistoryBudgetExceeded):
        await history.project(
            "chat", raw, total_tokens=10000, reserved_tokens=3000, measure=lambda *_: 7001
        )
