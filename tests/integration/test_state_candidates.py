"""Checked candidates and incremental memory are independent of benchmark answers."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    Principal,
    StateIssue,
    StateUpdateRejected,
    StateValidation,
    WorkState,
)
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
from reactive_context.adapters.travel_validation import (
    RestaurantObservation,
    RestaurantRequirement,
)
from reactive_context.benchmarking.travel_state import TravelStateAdapter
from reactive_context.errors import AccessRevoked, SchemaInvalid

HOST, AGENT = Principal("host", True), Principal("agent")
SCOPE, FIELD = "trip", "/days/1/breakfast"
BASE = [
    {
        "day": 1,
        "current_city": "Minneapolis",
        "transportation": "KEEP_DURATION_DISTANCE_COST",
        "breakfast": "Old Choice, Minneapolis",
        "attraction": "KEEP_MUSEUM",
        "lunch": "KEEP_LUNCH",
        "dinner": "KEEP_DINNER",
        "accommodation": "KEEP_HOTEL",
    }
]
BAD = {
    "name": "Soni Bakers",
    "city": "Minneapolis",
    "price": 42,
    "rating": 3.4,
    "cuisines": ["Pizza", "Seafood"],
}
GOOD = {
    "name": "Observed Alternative",
    "city": "Minneapolis",
    "price": 50,
    "rating": 3.8,
    "cuisines": ["Pizza", "Seafood"],
}
RULE = {
    "city": "Minneapolis",
    "price_min": 42.6,
    "price_max": 61.4,
    "rating_min": 3.5,
    "rating_max": 4.1,
    "cuisines": ["Pizza", "Seafood"],
}


def provision(path):
    store = ContextStore(path)
    store.create_scope(HOST, SCOPE)
    store.grant_scope(HOST, SCOPE, AGENT.id)
    user = store.observe_evidence(AGENT, SCOPE, "user-current-request", "1")
    source = store.observe_evidence(AGENT, SCOPE, "RestaurantSearch:Minneapolis", "1")
    adapter = TravelStateAdapter(store, AGENT, validate_restaurants=True)
    adapter.initialize(SCOPE, BASE, [user], host=HOST)
    validator = adapter.restaurant_validator
    validator.observe(SCOPE, [BAD, GOOD], source, host=HOST)
    validator.bind(adapter.state, SCOPE, {FIELD: RULE}, [user], expected_revision=0, host=HOST)
    permit = adapter.authorize_fields(
        SCOPE, [(1, "breakfast", "replace")], [user], expected_revision=1, host=HOST
    )
    return store, adapter, user, source, permit


def operations(label):
    return [{"op": "set", "path": FIELD, "value": label}]


def propose(adapter, label, source, permit):
    return adapter.state.candidates.propose(
        SCOPE, operations(label), [source], expected_revision=1, permit_id=permit
    )


def test_bad_candidate_reports_exact_violations_and_never_activates_or_becomes_fact(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Soni Bakers, Minneapolis", source, permit)
        assert candidate["status"] == "rejected"
        issues = candidate["last_check"]["validation"]["issues"]
        assert {issue["code"] for issue in issues} == {
            "price_below_minimum",
            "rating_below_minimum",
        }
        price = next(issue for issue in issues if issue["code"] == "price_below_minimum")
        assert price["expected"] == 42.6 and price["actual"] == 42
        committed = adapter.state.candidates.commit(SCOPE, candidate["candidate_id"])
        assert committed["status"] == "rejected"
        assert adapter.plan(SCOPE) == BASE
        assert adapter.state.deltas(SCOPE) == []
        assert store.query(AGENT, SCOPE, kind="fact") == []
        with pytest.raises(StateUpdateRejected) as error:
            adapter.state.patch(
                SCOPE,
                operations("Soni Bakers, Minneapolis"),
                expected_revision=1,
                permit_id=permit,
                evidence=[source],
            )
        assert error.value.validation["status"] == "failed"
        audit = json.dumps(
            [dict(row) for row in store.db.execute("SELECT * FROM work_state_audit")]
        )
        assert "Soni" not in audit and "42.6" not in audit


def test_valid_candidate_commits_once_and_only_changed_fields_enter_memory(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Observed Alternative, Minneapolis", source, permit)
        same = propose(adapter, "Observed Alternative, Minneapolis", source, permit)
        assert same["candidate_id"] == candidate["candidate_id"]
        assert candidate["status"] == "ready" and adapter.plan(SCOPE) == BASE
        result = adapter.state.candidates.commit(SCOPE, candidate["candidate_id"])
        assert result["status"] == "committed" and result["committed_revision"] == 2
        assert (
            adapter.state.candidates.commit(SCOPE, candidate["candidate_id"])["committed_revision"]
            == 2
        )
        assert adapter.plan(SCOPE)[0]["transportation"] == BASE[0]["transportation"]
        assert adapter.plan(SCOPE)[0]["attraction"] == BASE[0]["attraction"]
        deltas = adapter.state.deltas(SCOPE)
        assert len(deltas) == 1 and deltas[0]["revision"] == 2
        assert deltas[0]["requirements_revision"] == 1
        assert deltas[0]["validation"]["status"] == "passed"
        assert [c["path"] for c in deltas[0]["changes"]] == [FIELD]
        assert "KEEP_MUSEUM" not in json.dumps(deltas)
        assert set(deltas[0]["evidence"]) == {source, user}
        assert adapter.state.deltas(SCOPE, after_revision=2) == []
        assert (
            store.query(AGENT, SCOPE, kind="fact") == []
        )  # explicit state memory, no automatic promotion


def test_unknown_option_or_missing_source_stays_pending(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        unknown = propose(adapter, "Invented Choice, Minneapolis", source, permit)
        assert unknown["status"] == "pending"
        assert unknown["last_check"]["validation"]["issues"][0]["code"] == "selection_not_observed"
        known = propose(adapter, "Observed Alternative, Minneapolis", user, permit)
        assert known["status"] == "pending"
        assert known["last_check"]["validation"]["issues"][0]["code"] == "observed_source_not_bound"
        assert (
            adapter.state.candidates.commit(SCOPE, unknown["candidate_id"])["status"] == "pending"
        )
        assert adapter.plan(SCOPE) == BASE


def test_commit_rechecks_source_even_after_ready(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Observed Alternative, Minneapolis", source, permit)
        newer = store.observe_evidence(AGENT, SCOPE, "RestaurantSearch:Minneapolis", "2")
        adapter.restaurant_validator.observe(SCOPE, [{**GOOD, "price": 100}], newer, host=HOST)
        checked = adapter.state.candidates.commit(SCOPE, candidate["candidate_id"])
        assert checked["status"] == "pending" and adapter.plan(SCOPE) == BASE
        current = propose(adapter, "Observed Alternative, Minneapolis", newer, permit)
        assert current["status"] == "rejected"
        assert current["last_check"]["validation"]["issues"][0]["code"] == "price_above_maximum"


def test_requirement_change_invalidates_candidate_and_old_permits(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Observed Alternative, Minneapolis", source, permit)
        adapter.restaurant_validator.bind(
            adapter.state,
            SCOPE,
            {FIELD: {**RULE, "rating_min": 4.0}},
            [user],
            expected_revision=1,
            host=HOST,
        )
        assert (
            adapter.state.candidates.commit(SCOPE, candidate["candidate_id"])["status"] == "stale"
        )
        assert adapter.state.read(SCOPE)["permits"] == []
        with pytest.raises(StateUpdateRejected):
            adapter.state.patch(
                SCOPE,
                operations("Observed Alternative, Minneapolis"),
                expected_revision=1,
                permit_id=permit,
                evidence=[source],
            )
        assert adapter.state.read(SCOPE)["revision"] == 1


def test_missing_checker_cannot_be_bypassed_by_another_instance_or_restart(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Observed Alternative, Minneapolis", source, permit)
        reader = TravelStateAdapter(store, AGENT)
        assert reader.plan(SCOPE) == BASE
        with pytest.raises(SchemaInvalid, match="required state"):
            reader.state.patch(
                SCOPE,
                operations("Observed Alternative, Minneapolis"),
                expected_revision=1,
                permit_id=permit,
                evidence=[source],
            )
    with ContextStore(tmp_path) as store:
        reader = TravelStateAdapter(store, AGENT)
        with pytest.raises(SchemaInvalid):
            reader.state.candidates.commit(SCOPE, candidate["candidate_id"])
        restored = TravelStateAdapter(store, AGENT, validate_restaurants=True)
        assert (
            restored.state.candidates.commit(SCOPE, candidate["candidate_id"])["status"]
            == "committed"
        )


def test_atomic_rollback_includes_candidate_state_and_incremental_memory(tmp_path, monkeypatch):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Observed Alternative, Minneapolis", source, permit)
        write = adapter.state._write

        def fail(*args, **kwargs):
            write(*args, **kwargs)
            raise OSError("simulated failure after delta insert")

        monkeypatch.setattr(adapter.state, "_write", fail)
        with pytest.raises(OSError):
            adapter.state.candidates.commit(SCOPE, candidate["candidate_id"])
        assert adapter.state.candidates.read(SCOPE, candidate["candidate_id"])["status"] == "ready"
        assert adapter.plan(SCOPE) == BASE and adapter.state.deltas(SCOPE) == []


def test_candidate_owner_scope_revocation_and_discard_are_enforced(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Observed Alternative, Minneapolis", source, permit)
        other = Principal("other")
        store.grant_scope(HOST, SCOPE, other.id)
        stranger = TravelStateAdapter(store, other, validate_restaurants=True)
        with pytest.raises(AccessRevoked):
            stranger.state.candidates.read(SCOPE, candidate["candidate_id"])
        assert (
            adapter.state.candidates.discard(SCOPE, candidate["candidate_id"])["status"]
            == "discarded"
        )
        assert (
            adapter.state.candidates.commit(SCOPE, candidate["candidate_id"])["status"]
            == "discarded"
        )
        store.revoke_scope(HOST, SCOPE, AGENT.id)
        with pytest.raises(AccessRevoked):
            adapter.state.candidates.read(SCOPE, candidate["candidate_id"])
        with pytest.raises(AccessRevoked):
            adapter.state.deltas(SCOPE)


def test_concurrent_candidates_only_one_commits(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        adapter.restaurant_validator.observe(
            SCOPE, [{**GOOD, "name": "Other Alternative"}], source, host=HOST
        )
        candidates = [
            propose(adapter, name + ", Minneapolis", source, permit)
            for name in ["Observed Alternative", "Other Alternative"]
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(
                pool.map(
                    lambda c: adapter.state.candidates.commit(SCOPE, c["candidate_id"]), candidates
                )
            )
        assert sorted(item["status"] for item in result) == ["committed", "stale"]
        assert len(adapter.state.deltas(SCOPE)) == 1


def test_sources_are_immutable_host_only_and_scope_bound(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        with pytest.raises(AccessRevoked):
            adapter.restaurant_validator.observe(SCOPE, [{**BAD, "price": 50}], source, host=AGENT)
        with pytest.raises(SchemaInvalid):
            adapter.restaurant_validator.observe(SCOPE, [{**BAD, "price": 50}], source, host=HOST)
        store.create_scope(HOST, "other")
        e = store.observe_evidence(HOST, "other", "source", "1")
        with pytest.raises(AccessRevoked):
            adapter.restaurant_validator.observe(SCOPE, [GOOD], e, host=HOST)
        with pytest.raises(AccessRevoked):
            propose(adapter, "Observed Alternative, Minneapolis", e, permit)


def test_validator_bounds_and_no_checker_means_pending_not_verified(tmp_path):
    with pytest.raises(ValueError):
        RestaurantRequirement(price_min=50, price_max=40)
    with pytest.raises(ValueError):
        RestaurantObservation(**{**GOOD, "price": float("nan")})
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, SCOPE)
        store.grant_scope(HOST, SCOPE, AGENT.id)
        e = store.observe_evidence(AGENT, SCOPE, "source", "1")
        state = WorkState(store, AGENT)
        state.initialize(SCOPE, {"a": 1}, [e], host=HOST)
        permit = state.authorize(
            SCOPE, [{"path": "/a", "action": "replace"}], [e], expected_revision=1, host=HOST
        )
        c = state.candidates.propose(
            SCOPE,
            [{"op": "set", "path": "/a", "value": 2}],
            [e],
            expected_revision=1,
            permit_id=permit,
        )
        assert c["status"] == "pending"
        assert state.candidates.commit(SCOPE, c["candidate_id"])["status"] == "pending"
        assert state.read(SCOPE)["value"] == {"a": 1}
        state.set_requirements(SCOPE, {"minimum": 5}, [e], expected_revision=0, host=HOST)
        permit = state.authorize(
            SCOPE, [{"path": "/a", "action": "replace"}], [e], expected_revision=1, host=HOST
        )
        with pytest.raises(StateUpdateRejected) as error:
            state.patch(
                SCOPE,
                [{"op": "set", "path": "/a", "value": 6}],
                expected_revision=1,
                permit_id=permit,
            )
        assert error.value.validation["status"] == "pending"


@pytest.mark.parametrize(
    "attrs,code",
    [
        ({"city": "Elsewhere"}, "city_mismatch"),
        ({"cuisines": ["Pizza"]}, "required_cuisines_missing"),
        ({"rating": 4.2}, "rating_above_maximum"),
    ],
)
def test_declared_city_cuisines_and_upper_rating_are_checked(tmp_path, attrs, code):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        observed = {**GOOD, "name": "Test Alternative", **attrs}
        adapter.restaurant_validator.observe(SCOPE, [observed], source, host=HOST)
        c = propose(adapter, RestaurantObservation.model_validate(observed).label, source, permit)
        assert c["status"] == "rejected"
        assert code in {i["code"] for i in c["last_check"]["validation"]["issues"]}


def test_inclusive_numeric_boundaries_and_case_insensitive_cuisines(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        adapter.restaurant_validator.observe(
            SCOPE,
            [
                {
                    **GOOD,
                    "name": "Boundary",
                    "price": 42.6,
                    "rating": 3.5,
                    "cuisines": ["PIZZA", "seafood"],
                }
            ],
            source,
            host=HOST,
        )
        c = propose(adapter, "Boundary, Minneapolis", source, permit)
        assert c["status"] == "ready"


def test_host_check_input_is_immutable_and_invalid_return_fails_closed(tmp_path):
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, SCOPE)
        store.grant_scope(HOST, SCOPE, AGENT.id)
        e = store.observe_evidence(AGENT, SCOPE, "source", "1")

        def check(req):
            with pytest.raises(TypeError):
                req.after["a"] = 99
            return StateValidation("passed")

        state = WorkState(store, AGENT, check=check, check_id="test.check.v1")
        state.initialize(SCOPE, {"a": 1}, [e], host=HOST)
        permit = state.authorize(
            SCOPE, [{"path": "/a", "action": "replace"}], [e], expected_revision=1, host=HOST
        )
        state.patch(
            SCOPE, [{"op": "set", "path": "/a", "value": 2}], expected_revision=1, permit_id=permit
        )
        state.check = lambda req: True
        with pytest.raises(TypeError):
            state.replace(SCOPE, {"a": 2}, expected_revision=2)
        assert state.read(SCOPE)["revision"] == 2
    with pytest.raises(ValueError):
        StateValidation("passed", (StateIssue("/a", "bad"),))


async def test_native_propose_fix_commit_has_precise_feedback_and_no_early_activation(tmp_path):
    store, adapter, user, source, permit = provision(tmp_path)
    step = 0
    with store:

        def reply(messages, info):
            nonlocal step
            step += 1
            if step <= 2:
                if step == 2:
                    assert adapter.plan(SCOPE) == BASE
                    assert "price_below_minimum" in str(messages) and "42.6" in str(messages)
                args = {
                    "operations": operations(
                        "Soni Bakers, Minneapolis"
                        if step == 1
                        else "Observed Alternative, Minneapolis"
                    ),
                    "evidence": [source],
                    "expected_revision": 1,
                    "permit_id": permit,
                }
                return ModelResponse(
                    parts=[ToolCallPart("context_propose_state_candidate", args, f"propose-{step}")]
                )
            if step == 3:
                returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
                value = returns[-1].content
                assert value["status"] == "ready" and adapter.plan(SCOPE) == BASE
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "context_commit_state_candidate",
                            {"candidate_id": value["candidate_id"]},
                            "commit",
                        )
                    ]
                )
            assert adapter.plan(SCOPE)[0]["breakfast"] == "Observed Alternative, Minneapolis"
            return ModelResponse(parts=[TextPart("done")])

        async with ContextManager(
            store=store,
            principal=AGENT,
            work_state=adapter.state,
            compiler_policy=CompilerPolicy(max_tokens=128000),
        ) as manager:
            await Agent(
                FunctionModel(reply),
                capabilities=[ReactiveContextCapability(manager, scope=SCOPE, cache=None)],
            ).run("Select a compliant breakfast")
        assert step == 4 and len(adapter.state.deltas(SCOPE)) == 1


async def test_candidates_do_not_appear_in_default_context_and_uncovered_changes_stay_pending(
    tmp_path,
):
    store, adapter, user, source, permit = provision(tmp_path)
    with store:
        candidate = propose(adapter, "Soni Bakers, Minneapolis", source, permit)
        async with ContextManager(
            store=store, principal=AGENT, work_state=adapter.state
        ) as manager:
            snapshot = await manager.next_context(SCOPE)
            text = str(manager.compile(snapshot).provider_messages)
            assert "Soni Bakers" not in text and candidate["candidate_id"] not in text
            assert "Old Choice" in text
        # Permission to edit a field is not a substitute for a value validator.
        token = adapter.authorize_fields(
            SCOPE, [(1, "transportation", "replace")], [user], expected_revision=1, host=HOST
        )
        receipt = adapter.state.patch(
            SCOPE,
            [{"op": "set", "path": "/days/1/transportation", "value": "different"}],
            expected_revision=1,
            permit_id=token,
            evidence=[source],
            preview=True,
        )
        assert receipt["decision"] == "reject"
        assert receipt["validation"]["status"] == "pending"
        assert "changed_field_has_no_validator" in {
            i["code"] for i in receipt["validation"]["issues"]
        }
