from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from reactive_context import ContextStore
from reactive_context.core.snapshot import digest
from reactive_context.errors import AccessRevoked
from reactive_context.evidence import (
    BudgetUnsatisfiable,
    CompilationBudget,
    EvidenceReader,
    EvidenceRequirement,
    EvidenceResearcher,
    EvidenceStatus,
    ExecutionSelector,
    IndexedNavigator,
    ResearchSession,
    ResearchStopped,
    SessionBudget,
    SourceBinding,
    SourceSnapshot,
)
from reactive_context.store import Principal, VersionConflict


async def empty():
    return ()


async def validated(result):
    return result


def kwargs(**changes):
    return dict(
        slots=("dinner",),
        scope="s",
        index_revision="1",
        policy_revision="v1",
        operation=empty,
        validate=validated,
        **changes,
    )


async def test_negative_cache_is_versioned_expires_and_never_resets_total_budget():
    now = [0.0]
    session = ResearchSession(
        "task",
        "snapshot",
        "alice:s",
        clock=lambda: now[0],
        budget=SessionBudget(max_steps=3, max_wall_time_s=100),
    )
    await session.execute("  ZOE dinner", **kwargs())
    with pytest.raises(ResearchStopped, match="negative_cache"):
        await session.execute("zoe   dinner", **kwargs())
    changed = kwargs()
    changed["index_revision"] = "2"
    await session.execute("zoe dinner", **changed)
    now[0] = 31
    await session.execute("zoe dinner", **kwargs())
    changed["index_revision"] = "3"
    with pytest.raises(ResearchStopped) as exc:
        await session.execute("zoe dinner", **changed)
    assert exc.value.status == EvidenceStatus.BUDGET_EXHAUSTED
    assert session.steps == 3


def test_fingerprints_isolate_tasks_acl_snapshot_filters_slots_and_policies():
    a = ResearchSession("a", "snap", "alice:s")
    args = ("Q", ("dinner",), "s", "r1", "p1", ())
    identities = {a.fingerprint(*args)}
    for session in (
        ResearchSession("b", "snap", "alice:s"),
        ResearchSession("a", "snap2", "alice:s"),
        ResearchSession("a", "snap", "bob:s"),
    ):
        identities.add(session.fingerprint(*args))
    assert len(identities) == 4
    for index, value in ((1, ("hotel",)), (2, "other"), (3, "r2"), (4, "p2"), (5, ("filter",))):
        other = list(args)
        other[index] = value
        assert a.fingerprint(*other) not in identities


async def test_reordered_evidence_is_not_progress_and_new_text_does_not_reset_slot():
    session = ResearchSession("a", "b", "c")

    async def hits():
        return ("h1", "h2")

    k = kwargs()
    k["operation"] = hits
    await session.execute("one", **k)

    async def reordered():
        return ("h2", "h1")

    k["operation"] = reordered
    await session.execute("two", **k)
    await session.execute("three", **k)
    with pytest.raises(ResearchStopped, match="no_new_validated_evidence"):
        await session.execute("four", **k)
    assert session.trace[-1]["new_evidence_hashes"] == []
    with pytest.raises(ValueError):
        session.unrecoverable(requirement_id="x", proof_kind="searched_many_times", proof_ref="log")
    assert (
        session.unrecoverable(
            requirement_id="x",
            proof_kind="source_artifact_missing",
            proof_ref="immutable-record-sha",
        )
        == EvidenceStatus.UNRECOVERABLE
    )


async def test_actor_tool_actually_stops_identical_searches():
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from reactive_context.adapters.operations import context_tools

    calls = []

    async def search(*args, **kw):
        calls.append(args)
        return []

    history = SimpleNamespace(event_id=lambda event: digest(str(event)))
    manager = SimpleNamespace(history=history, search_history=search)
    toolset = context_tools(
        manager,
        "s",
        research_session=ResearchSession("a", "b", "p:s"),
        index_revision=lambda: "index1:acl1",
    )
    model_calls = []

    def model(messages, info):
        model_calls.append(1)
        return ModelResponse(
            parts=[ToolCallPart("context_history_search", {"intent": "Zoe day 5 dinner"})]
        )

    agent = Agent(FunctionModel(model), toolsets=[toolset])
    with pytest.raises(ResearchStopped, match="negative_cache"):
        await agent.run("Find the missing dinner")
    assert len(calls) == 1
    assert len(model_calls) == 2  # not 15 paid cycles


async def bundle():
    text = "Alice dinner A"
    source = SourceSnapshot(
        "one", "r1", "s", text, (SourceBinding("Alice.day1.dinner", 0, len(text), "A"),)
    )
    nav = IndexedNavigator([source], authorize=lambda *_: True)
    result = await EvidenceResearcher(nav).run(
        [EvidenceRequirement("Alice.day1.dinner", "Alice.day1.dinner", scope="s")], source.hits()
    )
    return source, nav, result


async def test_execution_budget_unknowns_dedup_conflict_and_read_acl():
    source, nav, report = await bundle()
    selector = ExecutionSelector(CompilationBudget(500, 500, 100, 1000), len)
    selected = selector.select(replace(report, excerpts=report.excerpts * 2))
    assert len(selected["evidence"]) == 1
    assert "trace" not in selected
    assert any(t["decision"] == "duplicate" for t in selector.trace)
    reader = EvidenceReader(nav, report)
    ref = selected["evidence"][0]["ref"]
    assert (await reader.read(ref, "s"))["text"] == source.text
    with pytest.raises(AccessRevoked):
        await reader.read(ref, "different")
    conflict = replace(report, conflict_ids=("Alice.day1.dinner",))
    selected = selector.select(conflict)
    assert not selected["evidence"]
    assert selected["unresolved"] == ("Alice.day1.dinner",)
    with pytest.raises(BudgetUnsatisfiable):
        ExecutionSelector(CompilationBudget(1, 1000, 0, 1000), len).select(report)
    with pytest.raises(BudgetUnsatisfiable):
        selector.select(report, reserved_actor_tokens=1000)


def test_missing_entity_is_retained_in_ledger_without_gold():
    from reactive_context.benchmarking.intelligence_pilot import requirement_ledger

    records = [
        {"name": "Alice", "daily_plans": [{"days": 1, "dinner": "A"}]},
        {"query": "I am Zoe.\nPlease plan dinner", "generated_plan": "truncated"},
    ]
    ledger = requirement_ledger(records, "Travel with Zoe")
    zoe = [r for r in ledger.records if r.entity == "Zoe"]
    assert len(zoe) == 7
    assert all(r.source_status == "source_artifact_missing" for r in zoe)
    assert ledger.metrics()["required_count"] == 14
    assert ledger.metrics()["evidence_validity"] == "NOT_MEASURED"
    assert len(type(ledger)([replace(zoe[0], status="revoked")]).requirements()) == 0


def test_fact_confirmation_is_host_only_append_only_and_revision_specific(tmp_path):
    class Note(BaseModel):
        text: str

    host, actor = Principal("host", admin=True), Principal("actor")
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "s")
        store.grant_scope(host, "s", actor.id)
        store.register_schema("note", Note)
        evidence = store.observe_evidence(actor, "s", "raw-record", "r1")
        store.record_evidence_support(actor, evidence, span=(0, 4), content_hash="hash")
        with pytest.raises(AccessRevoked):
            store.record_evidence_support(
                actor,
                evidence,
                span=(0, 4),
                content_hash="hash",
                support_status="validated",
                rule_ref="self-claim",
            )
        proposal = store.propose_fact(actor, "s", "note", "key", {"text": "a"}, [evidence])
        record = store.commit(actor, proposal)
        assert record.lifecycle == "candidate"
        with pytest.raises(AccessRevoked):
            store.confirm_fact(actor, record.id, expected_revision=1, confirmation_ref="self")
        confirmed = store.confirm_fact(
            host, record.id, expected_revision=1, confirmation_ref="trusted-rule:v1"
        )
        assert confirmed.revision == 2 and confirmed.lifecycle == "confirmed"
        assert store.load(actor, record.id, revision=1).lifecycle == "candidate"
        with pytest.raises(VersionConflict):
            store.confirm_fact(host, record.id, expected_revision=1, confirmation_ref="stale")
        proposal = store.propose_fact(
            actor, "s", "note", "key", {"text": "b"}, [evidence], expected_revision=2
        )
        assert store.commit(actor, proposal).lifecycle == "candidate"
        assert store.load(actor, record.id, revision=2).lifecycle == "confirmed"
        store.retire(actor, record.id, status="forgotten")
        assert not store.db.execute("SELECT * FROM fact_confirmations").fetchall()


async def test_shared_final_validation_bounded_repair_preserves_raw():
    from reactive_context.adapters.output import RepairBudget, validate_output
    from reactive_context.benchmarking.plan_projection import final_plan_view, project_plan

    def parse(text):
        return [{"day": 1, "dinner": "A"}]

    raw = "=== Zoe's Plan ===\nDay 1:\nDinner: A\nDinner: B"
    assert project_plan(raw, parse, required_fields=("dinner",)).final_plan is not None
    p = project_plan(
        raw, parse, required_fields=("dinner",), strict_slots=True, expected_entity="Alice"
    )
    assert {"duplicate_slot", "unexpected_entity"} <= set(p.audit["errors"])
    reservations = []

    async def reserve(cost):
        reservations.append(cost)

    async def repair(text, errors):
        return "=== Zoe's Plan ===\nDay 1:\nDinner: unknown"

    result = await validate_output(
        "analysis only",
        final_view=final_plan_view,
        validate=lambda _: (),
        repair=repair,
        budget=RepairBudget(1, 0.01),
        reserve=reserve,
        attempt_cost_upper_bound=0.01,
    )
    assert result["raw"] == "analysis only"
    assert len(result["versions"]) == 2 and reservations == [0.01]
    assert result["semantic_success"] == "NOT_MEASURED"


def test_ablation_single_variables_and_default_fallback():
    from dataclasses import asdict

    from reactive_context.benchmarking.intelligence_ablation import EXPERIMENTS, combined, fallback

    assert sum(asdict(EXPERIMENTS["E0"]).values()) == 0
    assert all(sum(asdict(EXPERIMENTS[name]).values()) == 1 for name in list(EXPERIMENTS)[1:])
    with pytest.raises(ValueError):
        combined({"E1": "NOT_MEASURED"})
    assert combined({"E1": "PASS"}).stop
    assert fallback(EXPERIMENTS["E1"], no_progress=True) == EXPERIMENTS["E0"]


async def test_session_deadline_and_cost_reserve_prevent_unbounded_work():
    import asyncio

    calls = []

    async def operation():
        calls.append(1)
        await asyncio.sleep(0.02)
        return ()

    session = ResearchSession("a", "b", "c", budget=SessionBudget(max_wall_time_s=0.001))
    options = kwargs()
    options["operation"] = operation
    with pytest.raises(ResearchStopped) as exc:
        await session.execute("q", **options)
    assert exc.value.status == EvidenceStatus.BUDGET_EXHAUSTED
    session = ResearchSession("a", "b", "c", budget=SessionBudget(max_cost_usd=0.01))
    calls.clear()
    with pytest.raises(ResearchStopped):
        await session.execute("q", cost_usd=0.02, **options)
    assert not calls


async def test_session_budget_counts_all_research_queries():
    from reactive_context.evidence import ResearchBudget

    nav = IndexedNavigator([], authorize=lambda *_: True)
    session = ResearchSession("task", "snapshot", "principal", budget=SessionBudget(max_steps=1))
    engine = EvidenceResearcher(
        nav, budget=ResearchBudget(max_rounds=10), session=session, index_revision=lambda: "v1"
    )
    report = await engine.run([EvidenceRequirement("missing", "missing")], mode="research")
    assert report.search_rounds == 1
    assert session.steps == 1
    with pytest.raises(ResearchStopped):
        await engine.run([EvidenceRequirement("other", "other")], mode="research")


async def test_frozen_ablation_changes_only_selected_variable():
    from reactive_context.benchmarking.intelligence_ablation import replay_evidence
    from reactive_context.evidence import ResearchBudget

    source, nav, report = await bundle()
    args = dict(
        requirements=report.requirements,
        candidates=source.hits(),
        navigator=nav,
        research_budget=ResearchBudget(),
        compilation_budget=CompilationBudget(1000, 1000, 100, 4096),
        session_budget=SessionBudget(),
        index_revision="r1",
        estimator=len,
    )
    e0 = await replay_evidence("E0", **args)
    e1 = await replay_evidence("E1", **args)
    e3 = await replay_evidence("E3", **args)
    e5 = await replay_evidence("E5", **args)
    assert e0["input_hash"] == e1["input_hash"] == e5["input_hash"]
    assert e3["input_hash"] != e0["input_hash"]
    assert e3["candidate_hash"] == e5["candidate_hash"] == e0["candidate_hash"]


def test_current_user_requirements_are_not_mistaken_for_failed_past_artifact():
    from reactive_context.benchmarking.intelligence_pilot import requirement_ledger

    rows = requirement_ledger(
        [{"name": "Alice", "daily_plans": [{"days": 1}]}], "I am Bob.\nJoin Alice"
    ).records
    current = [r for r in rows if r.entity == "Bob"]
    assert len(current) == 7
    assert all(r.source_status == "unlocated" for r in current)
