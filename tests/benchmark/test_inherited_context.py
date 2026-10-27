"""Baseline delivery is opt-in and uses stored Facts, not reference answers."""

from reactive_context import ContextManager, ContextStore
from reactive_context.benchmarking.intelligence_pilot import CoverageIntegration
from reactive_context.benchmarking.memoryarena_pilot import (
    ACTOR,
    HOST,
    SCHEMA,
    SCOPE,
    TravelFact,
    select_facts,
)


async def test_retained_baseline_does_not_displace_explicit_dependencies(tmp_path):
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, SCOPE)
        store.grant_scope(HOST, SCOPE, ACTOR.id)
        store.register_schema(SCHEMA, TravelFact)
        evidence = store.observe_evidence(ACTOR, SCOPE, "observed", "1")
        keys = [f"Base/day-{i}" for i in range(1, 8)] + [f"Peer/day-{i}" for i in range(1, 11)]
        for key in keys:
            person, topic = key.split("/")
            fact = TravelFact(subject=person, topic=topic, content=key, quotes=[key])
            store.commit(
                ACTOR, store.propose_fact(ACTOR, SCOPE, SCHEMA, key, fact.model_dump(), [evidence])
            )
        retained = keys[:7]
        required = keys[7:12]
        async with ContextManager(store=store, principal=ACTOR) as manager:
            selected = await select_facts(
                manager, "Peer", required_keys=required, retained_keys=retained
            )
            selected_keys = {store.metadata(ACTOR, id)["key"] for id in selected}
            assert set(retained + required) <= selected_keys
            assert len(selected) == len(set(selected)) == 12
            assert {r.part_id for r in store.active_refs(ACTOR, SCOPE)} == set(selected)
            # Re-entering the unmodified control arm restores its original top-k size.
            baseline = await select_facts(manager, "Peer", required_keys=required)
            assert len(baseline) == 8


def test_inheritance_policy_is_explicit_opt_in_without_injecting_seed_values(tmp_path):
    records = [{"name": "Base", "daily_plans": [{"day": 1, "dinner": "SEED_VALUE"}]}]
    query = "Use Peer day 2 dinner"
    control = CoverageIntegration("dx-baseline-shadow", tmp_path)
    assert control.recovery_context(records, query) == {}
    assert not control.retained_fact_keys
    candidate = CoverageIntegration("dx-baseline-shadow", tmp_path, retain_initial_plan=True)
    context = candidate.recovery_context(records, query)
    assert candidate.retained_fact_keys == ("Base/day-1",)
    assert context["plan_inheritance"]["baseline_person"] == "Base"
    assert "SEED_VALUE" not in str(context)
    assert "comparison evidence" in context["plan_inheritance"]["policy"]
    full = CoverageIntegration("full-history", tmp_path, retain_initial_plan=True)
    assert full.recovery_context(records, query) == {}


async def test_absent_baseline_fact_is_not_manufactured_or_restored_from_raw_seed(tmp_path):
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, SCOPE)
        store.grant_scope(HOST, SCOPE, ACTOR.id)
        async with ContextManager(store=store, principal=ACTOR) as manager:
            selected = await select_facts(manager, "dinner", retained_keys=("Base/day-1",))
            assert selected == []
            assert not store.active_refs(ACTOR, SCOPE)
