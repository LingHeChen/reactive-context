import copy

import pytest

from reactive_context import ContextManager, ContextStore
from reactive_context.benchmarking import memoryarena_pilot as base
from reactive_context.benchmarking.direction_pilot import DirectionIntegration, compact_catalog
from reactive_context.core.snapshot import thaw


@pytest.mark.parametrize("direction", ["control", "fact-state"])
async def test_classified_state_survives_real_admission_only_in_candidate(tmp_path, direction):
    integration = DirectionIntegration(tmp_path, direction)
    source = {"query": "I am Alice.", "generated_plan": "Hotel A", "execution_status": "completed"}
    draft = base.FactDraft(
        subject="Alice",
        topic="day-1",
        content="Hotel A",
        anchors=[base.SourceAnchor(path="/generated_plan")],
    )
    facts, _ = base.classify_fact_drafts([draft], source, [])
    assert facts[0].record_state == "submitted_plan"
    with ContextStore(tmp_path / "store") as store:
        store.create_scope(base.HOST, base.SCOPE)
        store.grant_scope(base.HOST, base.SCOPE, base.ACTOR.id)
        store.register_schema(base.SCHEMA, base.TravelFact, check=integration.check)
        async with ContextManager(store=store, principal=base.ACTOR) as manager:
            result = await integration.commit(manager, facts, source, 1)
            record = store.load(base.ACTOR, result[0]["id"])
            assert record.value["record_state"] == (
                "submitted_plan" if direction == "fact-state" else "observation"
            )
            catalog = [{"key": record.key, "value": thaw(record.value)}]
            accepted, rejected = base.classify_fact_drafts(
                [draft], {**source, "execution_status": "incomplete"}, catalog
            )
            assert bool(rejected) == (direction == "fact-state")
            assert bool(accepted) == (direction == "control")
            assert record.value["content"] == "Hotel A"


def test_catalog_preserves_keys_current_subject_and_does_not_mutate_host_values():
    catalog = [
        {
            "id": p,
            "key": f"{p}/day-1",
            "value": {
                "subject": p,
                "topic": "day-1",
                "record_state": "submitted_plan",
                "content": "OLD_VALUE",
                "quotes": ["SOURCE"],
            },
        }
        for p in ["Alice", "Bob"]
    ]
    original = copy.deepcopy(catalog)
    result = compact_catalog(catalog, {"query": "I am Alice."})
    assert result[0] == catalog[0]
    assert result[1]["id"] == "Bob" and result[1]["key"] == "Bob/day-1"
    assert "content" not in result[1]["value"] and "quotes" not in result[1]["value"]
    assert catalog == original
    assert compact_catalog(catalog, {"query": "ambiguous"}) == catalog


def test_directions_are_independent_and_reference_hints_contain_no_seed_values(tmp_path):
    records = [
        {"name": "Base", "daily_plans": [{"day": 1, "dinner": "SEED_SECRET"}]},
        {
            "query": "I am Peer.",
            "generated_plan": "=== Peer's Plan ===\nDay 1:\nDinner: OTHER_SECRET",
        },
    ]
    query = "I am Alice.\nFor day 2 dinner, choose a place cheaper than Peer's first-day lunch."
    for direction in ["control", "fact-state", "reference-hints", "writer-catalog"]:
        integration = DirectionIntegration(tmp_path, direction)
        result = integration.recovery_context(records, query)
        assert integration.preserve_record_state == (direction == "fact-state")
        assert ("comparison_references" in result) == (direction == "reference-hints")
        if direction == "reference-hints":
            assert result["comparison_references"]["hints"] == [
                {"person": "Peer", "day": 1, "field": "lunch"}
            ]
            assert "SECRET" not in str(result)
    with pytest.raises(ValueError):
        DirectionIntegration(tmp_path, "fact-state+reference-hints")
