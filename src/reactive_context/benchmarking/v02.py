"""Real feature interventions for v02 comparisons; no name-only ablation aliases.

This module configures the existing runtime. The dataset adapter and original evaluator remain
external and must retain their official contracts. Fixture outcomes have mode='protocol'.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..adapters.pydantic_ai import ReactiveContextCapability
from ..context import Context
from ..core.snapshot import thaw


@dataclass(frozen=True)
class Variant:
    name: str
    refresh: bool = True
    active_selection: bool = True
    history_compaction: bool = True
    dependency_invalidation: bool = True
    withdraw_history: bool = True


VARIANTS = {
    "B0": Variant("B0", False, False, False, False, False),
    "B1": Variant("B1", True, False, False, True, False),
    "B2": Variant("B2", True, True, True, False, True),
    "B3": Variant("B3"),
    "B3-no-refresh": Variant("B3-no-refresh", refresh=False),
    "B3-no-active-selection": Variant("B3-no-active-selection", active_selection=False),
    "B3-no-history-compaction": Variant("B3-no-history-compaction", history_compaction=False),
    "B3-no-dependency-invalidation": Variant(
        "B3-no-dependency-invalidation", dependency_invalidation=False
    ),
    "only-history-compaction": Variant("only-history-compaction", False, False, True, False, False),
    "only-active-selection": Variant("only-active-selection", False, True, False, False, False),
}


async def configure_variant(manager, scope, name, *, selected_part_ids, history=None):
    variant = VARIANTS[name]
    if not variant.dependency_invalidation:
        computed = [p for p in manager.registry.definitions.values() if p.computed]
        if computed:
            manager._authorize_keys(scope, [p.key for p in computed])
            initial = await Context.next_context(manager, scope, selected=[p.key for p in computed])
            for part in computed:
                value, revision = initial.part(part.key).value, initial.part(part.key).revision

                async def fixed_value(ctx, value=value):
                    return thaw(value)

                async def fixed_revision(ctx, revision=revision):
                    return revision

                manager.registry.definitions[part.key] = replace(
                    part, computed=False, deps=(), resolve=fixed_value, revision=fixed_revision
                )
                # Avoid retaining an instance with the old dependency graph.
                for pair in [p for p in manager.registry.instances if p[1] == part.key]:
                    del manager.registry.instances[pair]
    ids = set(selected_part_ids)
    if not variant.active_selection:
        ids, after_id = set(), None
        while True:
            page = await manager.query(scope, limit=100, after_id=after_id)
            ids.update(row["id"] for row in page)
            if len(page) < 100:
                break
            after_id = page[-1]["id"]
    for ref in manager.store.active_refs(manager.principal, scope):
        if ref.part_id not in ids:
            await manager.deactivate(scope, ref.part_id)
    for part_id in sorted(ids):
        await manager.activate(scope, part_id)
    if variant.history_compaction and history is None:
        raise ValueError("this variant requires an actual HistoryManager")
    if history is not None:
        history.policy = replace(history.policy, projection="compact")
    return ReactiveContextCapability(
        manager,
        scope=scope,
        consistency="live" if variant.refresh else "pinned",
        history=history if variant.history_compaction else False,
        withdraw_history=variant.withdraw_history,
    )


def freeze_plan(
    directory,
    *,
    config,
    dataset_manifests,
    scorer_manifests,
    seeds=(17, 29, 43),
    mode="extension",
    variants=tuple(VARIANTS),
):
    if mode not in {"official", "extension", "protocol"}:
        raise ValueError("unknown experiment mode")
    if mode == "official":
        raise ValueError(
            "formal official runs require a verified official runtime adapter; "
            "the generic v02 interventions are research extensions"
        )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide distinct paired random seeds")
    if not dataset_manifests or not scorer_manifests:
        raise ValueError("pinned dataset and scorer manifests are required")
    plan = {
        "version": "0.2",
        "mode": mode,
        "seeds": list(seeds),
        "variants": [asdict(VARIANTS[name]) for name in variants],
        "configuration": config.public(),
        "datasets": dataset_manifests,
        "scorers": scorer_manifests,
        "gold_available_to_inference": False,
        "official_leaderboard_comparable": False,
        "cache_protocol": (
            "paired randomized order; cold/warm strata separately; no "
            "repair-only causal comparisons"
        ),
    }
    path = Path(directory) / "experiment-plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("experiment plan is frozen; choose a separate output directory")
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan
