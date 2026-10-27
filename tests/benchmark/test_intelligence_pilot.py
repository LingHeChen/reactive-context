import json

from reactive_context import CompilerPolicy, ContextStore
from reactive_context.benchmarking import memoryarena_pilot as base
from reactive_context.benchmarking.intelligence_pilot import CoverageIntegration, observed_sources
from reactive_context.sources.config import ConfigSource


def seed():
    return {
        "name": "Alice",
        "daily_plans": [{"days": 1, "breakfast": "A"}, {"days": 2, "breakfast": "B"}],
    }


def test_source_index_accepts_public_seed_days_and_only_declared_prior_finals():
    initial = seed()
    sources, _ = observed_sources([initial, {"generated_plan": "No final plan"}])
    assert {s.bindings[0].subject for s in sources} == {
        "Alice.day1.breakfast",
        "Alice.day2.breakfast",
    }
    output = "Analysis Day 99: secret\n=== Bob's Plan ===\nDay 1:\nBreakfast: C"
    sources, _ = observed_sources([initial, {"generated_plan": output}])
    assert len(sources) == 3
    assert "secret" not in str(sources)


async def test_new_research_reaches_model_and_shadow_does_not(tmp_path):
    with ContextStore(tmp_path / "store") as store:
        store.create_scope(base.HOST, base.SCOPE)
        store.grant_scope(base.HOST, base.SCOPE, base.ACTOR.id)
        for variant in ("dx-baseline-shadow", "dx-coverage-research"):
            folder = tmp_path / variant
            folder.mkdir()
            integration = CoverageIntegration(variant, folder)
            objective = {"current_query": "Travel with Alice", "initial_traveler": "Alice"}
            parts = [
                ConfigSource("work_objective", objective).part,
                *integration.context_parts(store),
            ]
            async with integration.manager_factory(
                parts,
                initial_values={"work_objective": objective},
                store=store,
                principal=base.ACTOR,
                mandatory=[p.key for p in parts],
                compiler_policy=CompilerPolicy(),
            ) as manager:
                before = await manager.compile(scope=base.SCOPE)
                await integration.prepare_context(
                    manager, objective["current_query"], [], [seed()], 1
                )
                compiled = await manager.compile(scope=base.SCOPE)
                assert compiled.manifest["evidence"]["required_count"] == 14
                if variant == "dx-baseline-shadow":
                    assert before.messages == compiled.messages
                else:
                    payload = json.loads(
                        next(
                            m.content
                            for m in compiled.messages
                            if m.part_id == "researched_evidence"
                        )
                    )["data"]
                    assert "Alice.day1.breakfast" in payload
                    assert compiled.manifest["evidence"]["covered_count"] == 2
                    assert compiled.manifest["evidence"]["sufficiency"] == "incomplete"
