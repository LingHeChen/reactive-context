"""Domain adapter: explicit visible base, preserved fields, no reference-answer rules."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reactive_context import ContextStore, Principal, StateUpdateRejected
from reactive_context.benchmarking import memoryarena as upstream
from reactive_context.benchmarking.travel_state import (
    TravelStateAdapter,
    final_plan_text,
    keyed_plan,
)
from reactive_context.errors import SchemaInvalid

HOST, ACTOR = Principal("host", True), Principal("planner")
PLAN = [
    {
        "day": 1,
        "current_city": "City",
        "transportation": "Taxi, duration: 2 hours, cost: 30",
        "breakfast": "A",
        "attraction": "Museum",
        "lunch": "B",
        "dinner": "C",
        "accommodation": "Hotel",
    }
]


def setup(path):
    store = ContextStore(path)
    store.create_scope(HOST, "trip")
    store.grant_scope(HOST, "trip", ACTOR.id)
    e = store.observe_evidence(ACTOR, "trip", "visible-plan-and-current-request", "1")
    adapter = TravelStateAdapter(store, ACTOR)
    adapter.initialize("trip", PLAN, [e], host=HOST)
    return store, adapter, e


@pytest.mark.skipif(not upstream.TRAVEL.exists(), reason="optional pinned upstream not downloaded")
def test_valid_dinner_patch_renders_full_plan_without_dropping_transport_or_attractions(tmp_path):
    store, adapter, e = setup(tmp_path)
    with store:
        token = adapter.authorize_fields(
            "trip", [(1, "dinner", "replace")], [e], expected_revision=1, host=HOST
        )
        candidate = deepcopy(PLAN)
        candidate[0]["dinner"] = "New dinner"
        text = adapter.render("trip", "Person").replace("Dinner: C", "Dinner: New dinner")
        text = "Analysis contains Day 1: this is not the result.\n" + text
        result = adapter.accept_output(
            "trip", text, upstream.parser(), expected_revision=1, permit_id=token
        )
        assert result["decision"] == "updated"
        assert adapter.plan("trip") == candidate
        parsed = upstream.parser()(adapter.render("trip", "Person"))
        assert parsed == candidate


def test_transport_rewrite_and_attraction_clear_fail_atomically(tmp_path):
    store, adapter, e = setup(tmp_path)
    with store:
        token = adapter.authorize_fields(
            "trip", [(1, "dinner", "replace")], [e], expected_revision=1, host=HOST
        )
        candidate = deepcopy(PLAN)
        candidate[0].update(
            dinner="New dinner", transportation="Taxi, Departure Time: -", attraction="-"
        )
        with pytest.raises(StateUpdateRejected) as error:
            adapter.state.replace(
                "trip", keyed_plan(candidate), expected_revision=1, permit_id=token
            )
        assert set(error.value.reasons) == {"unpermitted_replace", "unpermitted_clear"}
        assert adapter.plan("trip") == PLAN
        assert adapter.state.read("trip")["revision"] == 1


def test_explicit_clear_uses_domain_sentinel_and_requires_its_own_grant(tmp_path):
    store, adapter, e = setup(tmp_path)
    with store:
        token = adapter.authorize_fields(
            "trip", [(1, "attraction", "clear")], [e], expected_revision=1, host=HOST
        )
        adapter.state.patch(
            "trip",
            [{"op": "set", "path": "/days/1/attraction", "value": "-"}],
            expected_revision=1,
            permit_id=token,
        )
        assert adapter.plan("trip")[0]["attraction"] == "-"
        assert adapter.plan("trip")[0]["transportation"] == PLAN[0]["transportation"]


def test_ambiguous_final_blocks_and_duplicate_days_are_rejected():
    with pytest.raises(SchemaInvalid):
        final_plan_text("=== A's Plan ===\nDay 1:\n=== B's Plan ===\nDay 1:")
    with pytest.raises(SchemaInvalid):
        keyed_plan(PLAN + PLAN)
    assert "Analysis" not in final_plan_text("Analysis Day 7:\n=== A's Plan ===\nDay 1:")


@pytest.mark.skipif(
    not Path("results/memoryarena-generic-rerun/fact-small/36/stages.jsonl").exists(),
    reason="optional prior paid-run capture unavailable",
)
def test_recorded_transport_retention_failure_is_blocked_without_gold(tmp_path):
    # Public initial plan + actually generated output only; never row['answers'].
    row = next(row for row in upstream.tasks(3) if row["id"] == 36)
    source = Path("results/memoryarena-generic-rerun/fact-small/36/stages.jsonl")
    output = json.loads(source.read_text().splitlines()[0])["output"]
    candidate = keyed_plan(upstream.parser()(final_plan_text(output)))
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "trip")
        store.grant_scope(HOST, "trip", ACTOR.id)
        e = store.observe_evidence(ACTOR, "trip", "visible-initial-plan", "1")
        adapter = TravelStateAdapter(store, ACTOR)
        adapter.initialize("trip", row["base_person"]["daily_plans"], [e], host=HOST)
        # The change boundary supplied by this fixture allows a meal update, not
        # transport rewrites. It makes no claim to infer all natural-language intents.
        token = adapter.authorize_fields(
            "trip", [(2, "dinner", "replace")], [e], expected_revision=1, host=HOST
        )
        preview = adapter.state.replace(
            "trip", candidate, expected_revision=1, permit_id=token, preview=True
        )
        transport = [c for c in preview["changes"] if c["path"] == "/days/1/transportation"]
        assert preview["decision"] == "reject" and transport
        assert transport[0]["before"] != transport[0]["after"]
        assert adapter.state.read("trip")["revision"] == 1
