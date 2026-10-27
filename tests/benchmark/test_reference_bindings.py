import json

import pytest

from reactive_context import ContextManager, ContextStore
from reactive_context.benchmarking import memoryarena_pilot as base
from reactive_context.benchmarking.direction_pilot import DirectionIntegration
from reactive_context.benchmarking.reference_bindings import (
    ReferenceBindingIntegration,
    quoted_field,
    relations,
)
from reactive_context.core.snapshot import thaw
from reactive_context.errors import AccessRevoked


def test_multiple_targets_keep_shared_reference_and_raw_clause():
    query = (
        "For dinner on day 3, spend 20% more than John's day 4 breakfast.\n"
        "For dinner on day 4, spend 30% more than John's day 4 breakfast."
    )
    edges = relations(query, ["John"])
    assert len(edges) == 2
    assert [e["target"]["day"] for e in edges] == [3, 4]
    assert all(e["reference"] == {"person": "John", "day": 4, "field": "breakfast"} for e in edges)
    assert [e["source_clause"] for e in edges] == query.splitlines()
    unknown = relations("I'd like to join John for breakfast on day 4.", ["John"])
    assert unknown[0]["target"] is None


def test_quotes_are_day_scoped_and_ambiguity_never_guesses():
    assert quoted_field({"quotes": ["Breakfast: A, City"]}, 4, "breakfast") is None
    assert quoted_field({"quotes": ["Day 3:\nBreakfast: A, City"]}, 4, "breakfast") is None
    assert quoted_field({"quotes": ["Day 4:\nBreakfast: A, City"]}, 4, "breakfast")[0] == "A, City"
    assert (
        quoted_field(
            {"quotes": ["Day 4:\nBreakfast: A, City", "Day 4:\nBreakfast: B, Other"]},
            4,
            "breakfast",
        )
        is None
    )
    assert (
        quoted_field({"quotes": [json.dumps({"days": 4, "breakfast": "A, City"})]}, 4, "breakfast")[
            0
        ]
        == "A, City"
    )


async def test_live_fact_version_binding_and_revocation(tmp_path, monkeypatch):
    async def no_shadow(*args, **kwargs):
        pass

    monkeypatch.setattr(DirectionIntegration, "prepare_context", no_shadow)
    integration = ReferenceBindingIntegration(tmp_path)
    integration.edges = relations("For dinner on day 3, join John's day 4 breakfast.", ["John"])
    with ContextStore(tmp_path / "store") as store:
        store.create_scope(base.HOST, base.SCOPE)
        store.grant_scope(base.HOST, base.SCOPE, base.ACTOR.id)
        store.register_schema(base.SCHEMA, base.TravelFact)
        evidence = store.observe_evidence(base.ACTOR, base.SCOPE, "observed", "1")

        def fact(venue):
            return base.TravelFact(
                subject="John",
                topic="day-4",
                content=f"Breakfast {venue}",
                quotes=[json.dumps({"day": 4, "breakfast": venue})],
            ).model_dump()

        record = store.commit(
            base.ACTOR,
            store.propose_fact(
                base.ACTOR, base.SCOPE, base.SCHEMA, "John/day-4", fact("Venue, City"), [evidence]
            ),
        )
        async with ContextManager(store=store, principal=base.ACTOR) as manager:
            await manager.activate(base.SCOPE, record.id)
            await integration.prepare_context(manager, "", [record.id], [], 1)

            def binding(snapshot):
                return thaw(snapshot.part("comparison_bindings").value)["relations"][0]["evidence"]

            first = binding(await manager.next_context(base.SCOPE))
            assert first["part_id"] == record.id and first["value"] == "Venue, City"
            revised = store.commit(
                base.ACTOR,
                store.propose_fact(
                    base.ACTOR,
                    base.SCOPE,
                    base.SCHEMA,
                    "John/day-4",
                    fact("Other, Elsewhere"),
                    [evidence],
                    expected_revision=record.revision,
                ),
            )
            second = binding(await manager.next_context(base.SCOPE))
            assert second["value"] == "Other, Elsewhere"
            assert second["revision"] == str(revised.revision) != first["revision"]
            store.revoke_scope(base.HOST, base.SCOPE, base.ACTOR.id)
            with pytest.raises(AccessRevoked):
                await manager.next_context(base.SCOPE)


async def test_unselected_fact_stays_unknown_without_extra_recall(tmp_path, monkeypatch):
    async def no_shadow(*args, **kwargs):
        pass

    monkeypatch.setattr(DirectionIntegration, "prepare_context", no_shadow)
    integration = ReferenceBindingIntegration(tmp_path)
    integration.edges = relations("For dinner on day 3, join John's day 4 breakfast.", ["John"])
    with ContextStore(tmp_path / "store") as store:
        store.create_scope(base.HOST, base.SCOPE)
        store.grant_scope(base.HOST, base.SCOPE, base.ACTOR.id)
        store.register_schema(base.SCHEMA, base.TravelFact)
        evidence = store.observe_evidence(base.ACTOR, base.SCOPE, "observed", "1")
        f = base.TravelFact(subject="John", topic="day-4", content="secret", quotes=["secret"])
        store.commit(
            base.ACTOR,
            store.propose_fact(
                base.ACTOR, base.SCOPE, base.SCHEMA, "John/day-4", f.model_dump(), [evidence]
            ),
        )
        async with ContextManager(store=store, principal=base.ACTOR) as manager:
            await integration.prepare_context(manager, "", [], [], 1)
            snap = await manager.next_context(base.SCOPE)
            value = thaw(snap.part("comparison_bindings").value)
            assert value["relations"][0]["status"] == "unknown"
            assert "secret" not in str(value)
            assert not store.active_refs(base.ACTOR, base.SCOPE)
