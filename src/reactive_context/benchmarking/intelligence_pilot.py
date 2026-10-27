"""End-to-end coverage pilot. Public observed plans only; no gold enters this adapter."""

import argparse
import asyncio
import fcntl
import json
import random
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ..core.snapshot import digest, thaw
from ..evidence import (
    EvidenceRequirement,
    EvidenceResearcher,
    IndexedNavigator,
    ResearchBudget,
    SourceBinding,
    SourceSnapshot,
)
from ..sources.config import ConfigSource
from . import memoryarena as upstream
from . import memoryarena_pilot as base
from .config import BenchmarkConfig
from .history_regression import append, write_json
from .memoryarena_dx_pilot import DXManager, Integration, hashes
from .official import ROOT
from .plan_projection import PROJECTION_REVISION, final_plan_view, score_tracks
from .runtime import BudgetExceeded, BudgetLedger, make_model

VARIANTS = {
    "full-history": "full-history",
    "retrieval-compression": "retrieval-compression",
    "dx-baseline-shadow": "fact-archive",
    "dx-coverage-research": "fact-archive",
}


def observed_sources(records):
    """Canonical JSON subtrees of actual seed/outputs; one field per immutable span.

    Binding means 'this person previously output this field', not business correctness.
    It is not an authoritative Fact promotion. Raw records persist alongside this index.
    """
    sources, owners = [], {}
    parse = upstream.parser()
    for index, record in enumerate(records):
        if index == 0:
            person = record["name"]
            days = record.get("daily_plans", [])
        else:
            text = final_plan_view(record.get("generated_plan", ""))
            if not text:
                continue
            match = re.match(r"=== (.+)['’]s Plan ===", text)
            if match is None:
                continue
            person = match[1]
            days = parse(text) or []
        ids = [d.get("day", d.get("days")) for d in days]
        if len(ids) != len(set(ids)):
            continue
        for day in days:
            number = day.get("day", day.get("days"))
            if not isinstance(number, int) or number < 1:
                continue
            for key, value in sorted(day.items()):
                if key in {"day", "days"} or not isinstance(value, str) or not value.strip():
                    continue
                subject = f"{person}.day{number}.{key}"
                text = json.dumps({key: value}, ensure_ascii=False, sort_keys=True)
                source_id = f"observed:{index}:{person}:day{number}:{key}"
                source = SourceSnapshot(
                    source_id,
                    digest(text),
                    base.SCOPE,
                    text,
                    (SourceBinding(subject, 0, len(text), value),),
                )
                sources.append(source)
                owners[source_id] = {
                    "person": person,
                    "value": value,
                    "record": index,
                    "record_hash": digest(record),
                }
    return sources, owners


class CoverageManager(DXManager):
    def compile(
        self,
        snapshot=None,
        *,
        target="generic",
        history_tokens=0,
        scope="default",
        consistency="live",
        evidence=None,
    ):
        if snapshot is not None and evidence is None:
            evidence = getattr(self, "coverage_report", None)
        return super().compile(
            snapshot,
            target=target,
            history_tokens=history_tokens,
            scope=scope,
            consistency=consistency,
            evidence=evidence,
        )


class CoverageIntegration(Integration):
    def __init__(
        self,
        variant,
        directory,
        *,
        use_requirement_ledger=False,
        selector=None,
        stop_budget=None,
        strict_final=False,
        retain_initial_plan=False,
    ):
        super().__init__(variant)
        self.retain_initial_plan = retain_initial_plan and variant.startswith("dx-")
        self.retained_fact_keys = ()
        self.selection_audits = []
        self.use_requirement_ledger = use_requirement_ledger
        self.selector = selector
        self.stop_budget = stop_budget
        self.projection_options = {"strict_slots": strict_final}
        self.fallback_reason = None
        self.directory = directory
        self.research = []
        self.extra_source = None

    def context_tool_options(self, manager, stage):
        if self.stop_budget is None:
            return {}
        from ..evidence import ResearchSession

        session = ResearchSession(
            str(self.directory),
            f"stage:{stage}",
            manager.principal.id + ":" + base.SCOPE,
            budget=self.stop_budget,
        )
        self.actor_session = session

        return {"research_session": session, "requirement_key": self.retrieval_slots}

    def retrieval_slots(self, intent):
        from .travel_dependencies import dependencies

        hints = dependencies(intent, getattr(self, "known_people", ()))
        return tuple(f"{p}.day{day}.{field}" for p, day, field in hints) or (
            "current_task_unresolved_context",
        )

    def dependency_keys(self, query):
        from .travel_dependencies import dependencies

        return tuple(
            sorted(
                {
                    f"{p}/day-{day}"
                    for p, day, _ in dependencies(query, getattr(self, "known_people", ()))
                }
            )
        )

    def recovery_context(self, records, query):
        """Failure status is public task state, distinct from uncommitted tool observations."""
        self.known_people = {records[0]["name"]}
        self.retained_fact_keys = (
            tuple(
                f"{records[0]['name']}/day-{d.get('day', d.get('days'))}"
                for d in records[0].get("daily_plans", ())
            )
            if self.retain_initial_plan
            else ()
        )
        missing = []
        for index, record in enumerate(records[1:], 1):
            match = re.match(r"I am ([^\.\n]+)\.", record.get("query", ""))
            if not match:
                continue
            self.known_people.add(match[1])
            if not re.search(r"\b" + re.escape(match[1]) + r"\b", query, re.I):
                continue
            if record.get("final_status") == "incomplete" or not final_plan_view(
                record.get("generated_plan", "")
            ):
                missing.append(
                    {
                        "entity": match[1],
                        "stage": index,
                        "status": "source_artifact_missing",
                        "reason": "no_valid_committed_final_plan",
                        "partial_observations_available": bool(record.get("actual_interaction")),
                        "instruction": "Do not search for a confirmed plan that was never "
                        "produced. Observations are not selected decisions.",
                    }
                )
        context: dict[str, object] = (
            {"unresolved_prior_plans": missing} if missing and self.stop_budget else {}
        )
        if self.retain_initial_plan:
            context["plan_inheritance"] = {
                "baseline_person": records[0]["name"],
                "baseline_fact_keys": self.retained_fact_keys,
                "policy": "Use the initial traveler's fixed plan as the baseline. Preserve "
                "fields the current traveler has not changed. Other travelers' plans are "
                "comparison evidence for explicitly referenced requirements, not replacements "
                "for the entire baseline. If baseline information is missing, retrieve it; "
                "do not invent unchanged fields or treat missing fields as '-'.",
            }
        return context

    def on_actor_failure(self, error):
        from ..evidence import ResearchStopped

        if isinstance(error, ResearchStopped):
            self.fallback_reason = error.reason
            append(
                self.directory / "fallback.jsonl",
                {
                    "reason": error.reason,
                    "next_policy": "fresh_guarded_session",
                    "session_usage": self.actor_session.usage(),
                    "session_trace": self.actor_session.trace,
                },
            )

    def on_output_projection(self, projection):
        if self.projection_options["strict_slots"] and not projection.audit["valid"]:
            self.fallback_reason = "invalid_final_structure"
            self.variant = "dx-baseline-shadow"
            self.selector = None
            append(
                self.directory / "fallback.jsonl",
                {
                    "reason": self.fallback_reason,
                    "next_policy": "dx-baseline-shadow",
                    "errors": thaw(projection.audit["errors"]),
                },
            )

    def context_parts(self, store):
        if self.variant != "dx-coverage-research":
            return ()
        self.extra_source = ConfigSource("researched_evidence", {"citations": []})
        return (self.extra_source.part,)

    def manager_factory(self, parts=(), *, initial_values, **kwargs):
        if not initial_values or not self.variant.startswith("dx-"):
            return super().manager_factory(parts, initial_values=initial_values, **kwargs)
        parts = [p for p in parts if p.key != "work_objective"]
        kwargs["compiler_policy"] = replace(
            kwargs["compiler_policy"],
            research_mode="shadow" if self.variant == "dx-baseline-shadow" else "schema_coverage",
        )
        return CoverageManager(
            parts,
            objective_input=initial_values["work_objective"],
            benchmark_probe=self.compilations,
            **kwargs,
        )

    async def prepare_context(self, manager, query, selected, records, stage):
        started = perf_counter()
        sources, owners = observed_sources(records)
        people = {records[0]["name"]}
        people.update(
            info["person"]
            for info in owners.values()
            if re.search(r"\b" + re.escape(info["person"]) + r"\b", query, re.I)
        )
        selected_facts = [manager.store.load(base.ACTOR, key) for key in selected]
        if self.retain_initial_plan:
            keys = {fact.key for fact in selected_facts}
            self.selection_audits.append(
                {
                    "stage": stage,
                    "retained_fact_keys": self.retained_fact_keys,
                    "selected_baseline_keys": sorted(keys & set(self.retained_fact_keys)),
                    "missing_baseline_keys": sorted(set(self.retained_fact_keys) - keys),
                    "dependency_keys": self.dependency_keys(query),
                    "selected_dependency_keys": sorted(keys & set(self.dependency_keys(query))),
                    "selected_fact_count": len(selected_facts),
                    "selected_fact_tokens_proxy": base.token_count(
                        base.encoded([thaw(f.value) for f in selected_facts])
                    ),
                    "semantic_retention": "NOT_MEASURED",
                }
            )
        candidates = []
        for source in sources:
            owner = owners[source.id]
            if owner["person"] not in people:
                continue
            # Conservatively recognize exact quoted field values, bound to the Fact's owner.
            if any(
                f.value["subject"].casefold() == owner["person"].casefold()
                and any(owner["value"] in q for q in f.value["quotes"])
                for f in selected_facts
            ):
                candidates.extend(source.hits())
        # Public output structure defines the requested slots; values are not requirements.
        fields = (
            "current_city",
            "transportation",
            "breakfast",
            "attraction",
            "lunch",
            "dinner",
            "accommodation",
        )
        reqs = tuple(
            EvidenceRequirement(
                f"{person}.day{day.get('day', day.get('days'))}.{field}",
                f"{person}.day{day.get('day', day.get('days'))}.{field}",
                scope=base.SCOPE,
            )
            for person in sorted(people)
            for day in records[0]["daily_plans"]
            for field in fields
        )

        def authorize(scope, source_id):
            manager.store._scope_access(base.ACTOR, scope)
            return scope == base.SCOPE and source_id in owners

        engine = EvidenceResearcher(
            IndexedNavigator(sources, authorize=authorize),
            budget=ResearchBudget(
                max_rounds=3,
                max_io=512,
                max_tokens=6000,
                timeout_seconds=3,
                initial_k=128,
                max_k=512,
            ),
            estimator=base.token_count,
            policy_revision="observed-plan-slots-v1",
        )
        ledger = requirement_ledger(records, query) if self.use_requirement_ledger else None
        if ledger is not None:
            reqs = ledger.requirements()
        shadow = await engine.run(reqs, candidates, mode="shadow")
        bundle = shadow
        if self.variant == "dx-coverage-research":
            # Reuse B1 findings, but the actual research invocation starts from the same candidates.
            bundle = await engine.run(reqs, candidates, mode="research")
            assert self.extra_source is not None
            self.extra_source.set(
                thaw(self.selector.select(bundle))
                if self.selector
                else {
                    "citations": thaw(bundle.excerpts),
                    "sufficiency": bundle.sufficiency,
                    "missing_ids": bundle.missing_ids,
                }
            )
        manager.coverage_report = bundle
        audit = {
            "stage": stage,
            "requirement_ledger": ledger.metrics() if ledger else None,
            "selected_fact_ids": selected,
            "selected_fact_revisions": {f.id: f.revision for f in selected_facts},
            "shadow": thaw(shadow.manifest()),
            "research": thaw(bundle.manifest()),
            "index_sources": len(sources),
            "observed_record_count": len(records),
            "index_and_research_ms": (perf_counter() - started) * 1000,
            "raw_record_hashes": [digest(r) for r in records],
            "body_added": self.variant == "dx-coverage-research",
        }
        self.research.append(audit)
        append(self.directory / "evidence-trace.jsonl", audit)
        write_json(self.directory / f"observed-stage-{stage}.json", records)

    def audit(self):
        return {
            **super().audit(),
            "evidence": self.research,
            "retain_initial_plan": self.retain_initial_plan,
            "selection_audits": self.selection_audits,
        }


async def run(args):
    directory = Path(args.output)
    auth = json.loads((directory / "authorization.json").read_text())
    ledger_path = ROOT / "results/history-regression-budget.json"
    with (ROOT / "results/v02/pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = json.loads(ledger_path.read_text())
        if start["requests"] != auth["ledger_start"]["requests"]:
            raise ValueError("ledger changed after authorization allocation")
        config = replace(
            BenchmarkConfig.load(args.env),
            max_requests=auth["max_requests"],
            max_cost_usd=auth["max_cost_usd"],
            max_output_tokens=6000,
            pricing_basis="DeepSeek Flash peak upper bound, verified 2026-10-26",
        )
        rows = upstream.tasks(3)
        jobs = [
            (trial, variant, row) for trial in range(1, 4) for row in rows for variant in VARIANTS
        ]
        random.Random(83).shuffle(jobs)
        manifest = {
            **upstream.manifest(),
            "variants": VARIANTS,
            "retain_initial_plan": args.retain_initial_plan,
            "trials": 3,
            "config": config.public(),
            "authorization": auth,
            "groups": [{"id": r["id"], "stages": len(r["questions"])} for r in rows],
            "job_order": [[t, v, r["id"]] for t, v, r in jobs],
            "source_hashes": hashes(),
            "projection_revision": PROJECTION_REVISION,
            "planned_stage_executions": 252,
            "scope": "end-to-end writer; combined coverage+raw research intervention",
            "not_isolated": ["B2 versus B3 versus B4", "LLM variance", "Fact IDs"],
            "gate": {
                "tolerance_pp": 0,
                "primary": "validated official PS",
                "blocking": "known authority violation or invalid final structure",
                "statistical_verdict": "inconclusive if CI overlaps zero",
            },
            "unmeasured": [
                "semantic TSR",
                "semantic Fact precision/recall",
                "coverage classifier precision/recall",
                "physical IOPS",
            ],
            "task_budget": {
                "actor_requests": 30,
                "output_tokens": 4096,
                "shared_all_phase_usd": 5,
                "shared_all_phase_calls": 2000,
            },
            "cache": "isolated application stores; provider cache may be shared",
            "requirements": "public seed schema for base and explicitly named prior people",
        }
        write_json(directory / "manifest.json", manifest)
        if args.preflight:
            print(
                json.dumps(
                    {
                        "credential_configured": bool(config.api_key),
                        "planned": len(jobs),
                        "calls_allowance": 2000,
                        "cost_allowance": 5,
                        "model": config.model_id,
                    }
                )
            )
            return
        for name in manifest["source_hashes"]:
            dest = directory / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        ledger = BudgetLedger(config, ledger_path)
        engine = await asyncio.to_thread(upstream.executor)
        semaphore, exhausted = asyncio.Semaphore(2), asyncio.Event()
        async with make_model(config) as model:

            async def job(trial, variant, row):
                async with semaphore:
                    if exhausted.is_set():
                        return
                    arm = VARIANTS[variant]
                    folder = directory / f"trial-{trial}" / variant / arm / str(row["id"])
                    folder.mkdir(parents=True, exist_ok=True)
                    if (folder / "integration.json").exists():
                        raise ValueError("no duplicate paid runs")
                    integration = CoverageIntegration(
                        variant, folder, retain_initial_plan=args.retain_initial_plan
                    )
                    try:
                        await base.episode(
                            row, arm, folder, model, ledger, engine, integration=integration
                        )
                    except BudgetExceeded:
                        exhausted.set()
                        append(
                            directory / "failures.jsonl",
                            {
                                "trial": trial,
                                "variant": variant,
                                "group": row["id"],
                                "error": "budget_exceeded",
                            },
                        )
                    except Exception as exc:
                        append(
                            directory / "failures.jsonl",
                            {
                                "trial": trial,
                                "variant": variant,
                                "group": row["id"],
                                "error": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
                    finally:
                        write_json(folder / "integration.json", integration.audit())

            await asyncio.gather(*(job(*job_args) for job_args in jobs))
        write_json(directory / "budget-end.json", json.loads(ledger_path.read_text()))
        scores = {}
        for trial in range(1, 4):
            for variant, arm in VARIANTS.items():
                folders = [
                    directory / f"trial-{trial}" / variant / arm / str(r["id"]) for r in rows
                ]
                if not all((d / "checkpoint.json").exists() for d in folders):
                    continue
                states = [json.loads((d / "checkpoint.json").read_text()) for d in folders]
                if not all(
                    s["completed"] == len(r["questions"]) for s, r in zip(states, rows, strict=True)
                ):
                    continue
                path = directory / f"trial-{trial}" / variant / "submission.jsonl"
                path.write_text(
                    "".join(
                        json.dumps({"id": r["id"], "persons": s["persons"]}) + "\n"
                        for s, r in zip(states, rows, strict=True)
                    )
                )
                scores[f"{trial}/{variant}"] = score_tracks(rows, path, folders)
        write_json(directory / "scores.json", scores)
        print(json.dumps(scores), flush=True)


def requirement_ledger(records, query, *, scope=base.SCOPE):
    """Public observed identities, including failed outputs; never consult task gold.

    This domain adapter intentionally over-approximates mentioned people's public trip
    fields. Actual task-requirement precision remains NOT_MEASURED.
    """
    from ..evidence.ledger import RequirementLedger, RequirementRecord
    from .plan_projection import REQUIRED_FIELDS

    sources, _ = observed_sources(records)
    by_subject = {}
    for source in sources:
        by_subject.setdefault(source.bindings[0].subject, []).append(source.id)
    known = {records[0]["name"]: "public_seed"}
    for index, record in enumerate(records[1:], 1):
        match = re.match(r"I am ([^\.\n]+)\.", record.get("query", ""))
        if match:
            known[match[1]] = f"observed_user_request:{index}:{digest(record['query'])}"
    current = re.match(r"I am ([^\.\n]+)\.", query)
    if current:
        known.setdefault(current[1], "current_user_task")
    mentioned = {name for name in known if re.search(r"\b" + re.escape(name) + r"\b", query, re.I)}
    mentioned.add(records[0]["name"])
    count = len(records[0].get("daily_plans", ()))
    rows = []
    for name in sorted(mentioned):
        artifact = any(subject.startswith(name + ".") for subject in by_subject)
        for day in range(1, count + 1):
            for field in REQUIRED_FIELDS:
                key = f"{name}.day{day}.{field}"
                refs = tuple(by_subject.get(key, ()))
                rows.append(
                    RequirementRecord(
                        key,
                        name,
                        field,
                        f"day{day}",
                        scope,
                        known[name],
                        observed_sources=refs,
                        source_status="located"
                        if refs
                        else "unlocated"
                        if artifact or known[name] == "current_user_task"
                        else "source_artifact_missing",
                        value_status="observed" if refs else "unresolved",
                    )
                )
    return RequirementLedger(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/intelligence-benchmark")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retain-initial-plan", action="store_true")
    asyncio.run(run(parser.parse_args()))
