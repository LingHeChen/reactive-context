"""Predeclared independent variables; unmeasured modules cannot enter E6/defaults."""

from dataclasses import asdict, dataclass

from ..core.snapshot import digest, freeze


@dataclass(frozen=True)
class IntelligenceFlags:
    stop: bool = False
    index: bool = False
    citations: bool = False
    budgeted_research: bool = False
    utility_selection: bool = False


EXPERIMENTS = {
    "E0": IntelligenceFlags(),
    "E1": IntelligenceFlags(stop=True),
    "E2": IntelligenceFlags(index=True),
    "E3": IntelligenceFlags(citations=True),
    "E4": IntelligenceFlags(budgeted_research=True),
    "E5": IntelligenceFlags(utility_selection=True),
}


def combined(validated):
    if not validated or any(name not in EXPERIMENTS or name == "E0" for name in validated):
        raise ValueError("E6 requires independently validated components")
    if any(result != "PASS" for result in validated.values()):
        raise ValueError("NOT_MEASURED or failed components cannot enter E6")
    values = {
        key: any(asdict(EXPERIMENTS[name])[key] for name in validated)
        for key in asdict(IntelligenceFlags())
    }
    return IntelligenceFlags(**values)


def fallback(flags, *, structural_regression=False, budget_runaway=False, no_progress=False):
    if structural_regression or budget_runaway or no_progress:
        return EXPERIMENTS["E0"]
    return flags


def preregistration(*, source_hashes, fixture_hashes, model_parameters, thresholds):
    if not thresholds:
        raise ValueError("declare release thresholds before evaluation")
    content = {
        "experiments": {k: asdict(v) for k, v in EXPERIMENTS.items()},
        "source_hashes": source_hashes,
        "fixture_hashes": fixture_hashes,
        "model_parameters": model_parameters,
        "thresholds": thresholds,
        "order": ["shadow", "frozen_replay", "repeated_complete_episodes"],
        "release": "NOT_MEASURED",
        "default": "E0",
        "metrics": {
            k: "NOT_MEASURED"
            for k in (
                "PS",
                "SPS",
                "semantic_TSR",
                "valid_final_structure",
                "inherited_fields",
                "source_coverage",
                "evidence_validity",
                "requirement_coverage",
                "answer_preservation",
                "pass_at_k",
                "pass_power_k",
                "unknown_fabrications",
                "stale_evidence",
                "conflict_merges",
                "cache_read_ratio",
                "latency_p50_p95",
            )
        },
        "accounting_roles": ["research", "fact_writer", "compiler", "actor"],
    }
    return freeze({**content, "manifest_hash": digest(freeze(content))})


async def replay_evidence(
    experiment,
    *,
    requirements,
    candidates,
    navigator,
    research_budget,
    compilation_budget,
    session_budget,
    index_revision,
    estimator,
):
    """Frozen candidate replay of one variable. No model, gold, writer or paid calls.

    This is an interface experiment, NOT an extraction or task-quality benchmark.
    E5 uses exactly E0's candidate set and renderer; only admission differs.
    """
    from dataclasses import replace

    from ..compiler.compiler import Compiler
    from ..compiler.ir import ContextIRSegment
    from ..compiler.policy import CompilerPolicy
    from ..core.snapshot import ContextSnapshot, PartSnapshot, thaw
    from ..evidence import EvidenceResearcher, ExecutionSelector, ResearchSession

    flags = EXPERIMENTS[experiment]
    session = (
        ResearchSession("frozen-replay", index_revision, "host:default", budget=session_budget)
        if flags.stop or flags.budgeted_research
        else None
    )
    engine = EvidenceResearcher(
        navigator,
        budget=replace(research_budget, max_rounds=1) if flags.index else research_budget,
        estimator=estimator,
        session=session,
        index_revision=lambda: index_revision,
    )
    bundle = await engine.run(
        requirements,
        candidates,
        mode=("dynamic" if flags.index else "research" if flags.budgeted_research else "shadow"),
    )
    excerpts = thaw(bundle.excerpts)
    selection_trace = ()
    unresolved = sorted(set(bundle.missing_ids) | set(bundle.conflict_ids))
    if flags.utility_selection:
        selector = ExecutionSelector(compilation_budget, estimator)
        selected = selector.select(bundle)
        keys = {(item["subject"], item["value"]) for item in selected["evidence"]}
        excerpts = [e for e in excerpts if (e["subject"], e["text"]) in keys]
        selection_trace = selector.trace
        unresolved = thaw(selected["unresolved"])
    # Same minimal renderer across E0/E1/E2/E4/E5. E3 changes ONLY citation representation.
    facts = [{"subject": e["subject"], "value": e["text"]} for e in excerpts]
    if flags.citations:
        facts = [
            {
                **fact,
                "source_ref": e["source_id"],
                "revision": e["source_revision"],
                "span": [e["span_start"], e["span_end"]],
                "content_hash": e["content_hash"],
            }
            for fact, e in zip(facts, excerpts, strict=True)
        ]
    payload = freeze(
        {
            "requirements": [r.id for r in requirements if r.required],
            "facts": facts,
            "unresolved": unresolved,
        }
    )
    policy = CompilerPolicy(
        max_tokens=compilation_budget.actor_input_token_cap, output_reserve=0, estimator=estimator
    )
    revision = digest(payload)
    part = PartSnapshot("execution", "default", "execution", revision, 0, payload, 0, 0)
    segment = ContextIRSegment(
        id="execution",
        part_id="execution",
        resource_id="execution",
        source_revision=revision,
        scope_id="default",
        kind="evidence",
        requested_role="user",
        required=True,
        payload=payload,
    )
    snapshot = ContextSnapshot(
        revision, "default", (part,), (segment,), policy.fingerprint, "independent-v1", "live"
    )
    result = Compiler(policy).compile(snapshot)
    return freeze(
        {
            "experiment": experiment,
            "flags": asdict(flags),
            "candidate_hash": digest(freeze(tuple(candidates))),
            "payload": payload,
            "input_hash": digest(freeze(result.messages)),
            "estimated_tokens": result.token_estimate,
            "model_calls": 0,
            "source_coverage": {
                "covered": len(bundle.covered_ids),
                "total": len(bundle.requirements),
            },
            "selection_trace": selection_trace,
            "research": bundle.manifest(),
            "semantic_TSR": "NOT_MEASURED",
        }
    )
