"""DX experiment exercises real declarations with no new model-visible task information."""

import json

from reactive_context import CompilerPolicy, ContextStore
from reactive_context.benchmarking import memoryarena_pilot as base
from reactive_context.benchmarking.memoryarena_dx_pilot import Integration
from reactive_context.sources.config import ConfigSource


async def test_same_objective_payload_and_no_hidden_declarations_in_messages(tmp_path):
    current = {"current_query": "public current query", "initial_traveler": "Base"}
    payloads = []
    probes = []
    with ContextStore(tmp_path) as store:
        store.create_scope(base.HOST, base.SCOPE)
        store.grant_scope(base.HOST, base.SCOPE, base.ACTOR.id)
        for variant in ("legacy-fact-small", "dx-fact-small"):
            integration = Integration(variant)
            async with integration.manager_factory(
                [ConfigSource("work_objective", current).part],
                initial_values={"work_objective": current},
                store=store,
                principal=base.ACTOR,
                mandatory=("work_objective",),
                compiler_policy=CompilerPolicy(),
            ) as manager:
                result = await manager.compile(scope=base.SCOPE)
                assert len(result.messages) == 1
                payloads.append(json.loads(result.messages[0].content)["data"])
                probes.append(integration.compilations)
                assert result.messages[0].part_id == "work_objective"
                if variant.startswith("dx-"):
                    assert set(manager.describe()) == {
                        "objective_input",
                        "derived_objective",
                        "work_objective",
                    }
    assert payloads[0] == payloads[1]
    assert probes[0][0]["objective_payload_hash"] == probes[1][0]["objective_payload_hash"]
