"""Offline review: candidate -> observed-source validation -> commit -> memory delta.

Run: .venv/bin/python examples/checked_state_candidates.py
The second restaurant is a synthetic fixture, not a benchmark answer.
"""

import json
from tempfile import TemporaryDirectory

from reactive_context import ContextStore, Principal
from reactive_context.benchmarking.travel_state import TravelStateAdapter


def main():
    host, agent = Principal("host", True), Principal("agent")
    scope, path = "breakfast-task", "/days/1/breakfast"
    with (
        TemporaryDirectory(prefix="checked-candidates-") as directory,
        ContextStore(directory) as store,
    ):
        store.create_scope(host, scope)
        store.grant_scope(host, scope, agent.id)
        request = store.observe_evidence(agent, scope, "current-user-requirements", "1")
        source = store.observe_evidence(agent, scope, "restaurant-tool-fixture", "1")
        adapter = TravelStateAdapter(store, agent, validate_restaurants=True)
        state = adapter.state
        adapter.initialize(
            scope,
            [
                {
                    "day": 1,
                    "current_city": "Minneapolis",
                    "transportation": "Taxi, duration: 2 hours, cost: 30",
                    "breakfast": "Existing Breakfast, Minneapolis",
                    "attraction": "Existing Museum",
                    "lunch": "Existing Lunch",
                    "dinner": "Existing Dinner",
                    "accommodation": "Existing Hotel",
                }
            ],
            [request],
            host=host,
        )
        # This mapping is supplied by the application from the current request.
        # Extracting it correctly from natural language is a separate responsibility.
        adapter.restaurant_validator.bind(
            state,
            scope,
            {
                path: {
                    "city": "Minneapolis",
                    "price_min": 42.6,
                    "price_max": 61.4,
                    "rating_min": 3.5,
                    "rating_max": 4.1,
                    "cuisines": ["Pizza", "Seafood"],
                }
            },
            [request],
            expected_revision=0,
            host=host,
        )
        # Only the trusted source adapter registers observations. The agent cannot
        # make an invalid restaurant pass by asserting a different price/rating.
        adapter.restaurant_validator.observe(
            scope,
            [
                {
                    "name": "Soni Bakers",
                    "city": "Minneapolis",
                    "price": 42,
                    "rating": 3.4,
                    "cuisines": ["Pizza", "Seafood"],
                },
                {
                    "name": "Example Compliant Restaurant",
                    "city": "Minneapolis",
                    "price": 50,
                    "rating": 3.8,
                    "cuisines": ["Pizza", "Seafood"],
                },
            ],
            source,
            host=host,
        )
        permit = adapter.authorize_fields(
            scope, [(1, "breakfast", "replace")], [request], expected_revision=1, host=host
        )

        def propose(label):
            return state.candidates.propose(
                scope,
                [{"op": "set", "path": path, "value": label}],
                [source],
                expected_revision=1,
                permit_id=permit,
            )

        rejected = propose("Soni Bakers, Minneapolis")
        assert rejected["status"] == "rejected" and state.read(scope)["revision"] == 1
        ready = propose("Example Compliant Restaurant, Minneapolis")
        assert ready["status"] == "ready" and state.read(scope)["revision"] == 1
        result = state.candidates.commit(scope, ready["candidate_id"])
        assert result["status"] == "committed"
        replay = state.candidates.commit(scope, ready["candidate_id"])
        assert replay["committed_revision"] == 2 and len(state.deltas(scope)) == 1
        print(
            json.dumps(
                {
                    "invalid_candidate": {
                        "status": rejected["status"],
                        "issues": rejected["last_check"]["validation"]["issues"],
                        "active_state_unchanged": True,
                    },
                    "valid_candidate_before_commit": {
                        "status": ready["status"],
                        "active_revision": 1,
                    },
                    "committed": {
                        "revision": result["committed_revision"],
                        "plan": adapter.plan(scope),
                    },
                    "incremental_memory": state.deltas(scope),
                    "replayed_commit": {"revision": replay["committed_revision"], "delta_count": 1},
                    "automatic_fact_promotions": len(store.query(agent, scope, kind="fact")),
                    "model_calls": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
