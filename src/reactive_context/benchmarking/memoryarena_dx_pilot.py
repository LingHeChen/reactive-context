"""Paid paired DX regression experiment. Explicit authorization file is mandatory.

Same task stream, tools, Fact writer/checks and scorer as GenericIntegration. No business
validation gates, future questions or gold references enter any actor/writer call.
"""

import argparse
import asyncio
import fcntl
import json
import random
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from .. import ContextManager, computed, part, state
from ..core.snapshot import ContextSnapshot, digest, thaw
from . import memoryarena as upstream
from . import memoryarena_pilot as base
from .config import BenchmarkConfig
from .history_regression import append, write_json
from .memoryarena_generic_pilot import GenericIntegration
from .official import ROOT, file_hash
from .plan_projection import PROJECTION_REVISION, score_tracks
from .runtime import BudgetExceeded, BudgetLedger, make_model

VARIANTS = {
    "full-history": "full-history",
    "retrieval-compression": "retrieval-compression",
    "legacy-fact-small": "fact-small",
    "dx-fact-small": "fact-small",
    "legacy-fact-archive": "fact-archive",
    "dx-fact-archive": "fact-archive",
}


def objective_revision(reader):
    return digest(reader.read("derived_objective"))


class ObservedManager(ContextManager):
    def __init__(self, *args, benchmark_probe, **kwargs):
        self.benchmark_probe = benchmark_probe
        super().__init__(*args, **kwargs)

    def compile(
        self,
        snapshot: ContextSnapshot | None = None,
        *,
        target="generic",
        history_tokens=0,
        scope="default",
        consistency="live",
        evidence=None,
    ) -> Any:
        if snapshot is None:
            return super().compile(
                target=target,
                history_tokens=history_tokens,
                scope=scope,
                consistency=consistency,
                evidence=evidence,
            )
        start = perf_counter()
        result = super().compile(
            snapshot, target=target, history_tokens=history_tokens, evidence=evidence
        )
        objective = next((p for p in snapshot.parts if p.part_id == "work_objective"), None)
        self.benchmark_probe.append(
            {
                "duration_ms": (perf_counter() - start) * 1000,
                "token_estimate": result.token_estimate,
                "objective_payload_hash": digest(objective.value) if objective else None,
                "snapshot_id": snapshot.id,
                "declared_parts": thaw(self.describe()),
            }
        )
        return result


class DXManager(ObservedManager):
    objective_input: dict = state(default_factory=dict, expose=False)

    @computed(expose=False)
    def derived_objective(self):
        return self.objective_input

    @part(revision=objective_revision)
    async def work_objective(self, reader):
        return reader.read("derived_objective")


class Integration(GenericIntegration):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.compilations = []

    def manager_factory(self, parts=(), *, initial_values, **kwargs):
        if not self.variant.startswith("dx-fact") or "work_objective" not in initial_values:
            return ObservedManager(parts, benchmark_probe=self.compilations, **kwargs)
        if any(p.key == "work_objective" for p in parts):
            parts = [p for p in parts if p.key != "work_objective"]
        return DXManager(
            parts,
            objective_input=initial_values["work_objective"],
            benchmark_probe=self.compilations,
            **kwargs,
        )

    def audit(self):
        return {**super().audit(), "variant": self.variant, "compilations": self.compilations}


def hashes():
    return {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in sorted((ROOT / "src/reactive_context").rglob("*.py"))
    }


async def run(args):
    directory = Path(args.output)
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    with (ROOT / "results/v02/pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auth_path = directory / "authorization.json"
        auth = json.loads(auth_path.read_text())  # No auto-created authorization or budget.
        ledger_path = ROOT / "results/history-regression-budget.json"
        start = json.loads(ledger_path.read_text())
        config = replace(
            BenchmarkConfig.load(args.env),
            max_requests=auth["max_requests"],
            max_cost_usd=auth["max_cost_usd"],
            max_output_tokens=6000,
        )
        rows = upstream.tasks(3)
        jobs = [(variant, row) for variant in VARIANTS for row in rows]
        random.Random(83).shuffle(jobs)
        manifest = {
            **upstream.manifest(),
            "mode": "dx-paired-regression",
            "scoring_projection_revision": PROJECTION_REVISION,
            "seed": 83,
            "variants": VARIANTS,
            "groups": [{"id": row["id"], "stages": len(row["questions"])} for row in rows],
            "job_order": [[variant, row["id"]] for variant, row in jobs],
            "config": config.public(),
            "harness_sha256": hashes(),
            "budget_ledger": "results/history-regression-budget.json",
            "new_authorization": auth,
            "controlled": [
                "actor/writer prompts",
                "business tools",
                "Fact schema and checks",
                "extraction schedule",
                "history projection",
                "scorer",
                "model settings",
            ],
            "dx_difference": (
                "Same current objective through hidden state, hidden computed and async part. "
                "Wrapper resource IDs/revisions differ; payloads must match public stage inputs."
            ),
            "excluded": "No state hard gates, travel constraint translation or gold-derived rules",
            "quality_scope": "One paired run of three travel groups; not statistical proof",
        }
        path = directory / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("resume manifest mismatch")
        write_json(path, manifest)
        if args.preflight:
            print(
                json.dumps(
                    {
                        "groups": manifest["groups"],
                        "variants": VARIANTS,
                        "remaining_requests": config.max_requests - start["requests"],
                        "remaining_usd": config.max_cost_usd - start["cost_upper_bound"],
                        "credential_configured": bool(config.api_key),
                    }
                )
            )
            return
        for name, checksum in manifest["harness_sha256"].items():
            destination = directory / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
            assert file_hash(destination) == checksum
        ledger = BudgetLedger(config, ledger_path)
        engine = await asyncio.to_thread(upstream.executor)
        semaphore = asyncio.Semaphore(2)
        exhausted = asyncio.Event()
        async with make_model(config) as model:

            async def job(variant, row):
                async with semaphore:
                    if exhausted.is_set():
                        return variant, None
                    arm = VARIANTS[variant]
                    folder = directory / variant / arm / str(row["id"])
                    integration = Integration(variant)
                    try:
                        result = await base.episode(
                            row,
                            arm,
                            folder,
                            model,
                            ledger,
                            engine,
                            integration=integration,
                        )
                        return variant, result
                    except BudgetExceeded:
                        exhausted.set()
                        append(
                            directory / "failures.jsonl",
                            {
                                "variant": variant,
                                "group": row["id"],
                                "error": "budget_exceeded",
                            },
                        )
                    except Exception as exc:
                        append(
                            directory / "failures.jsonl",
                            {
                                "variant": variant,
                                "group": row["id"],
                                "error": type(exc).__name__,
                                "message": str(exc)[:1000],
                            },
                        )
                    finally:
                        audit_path = folder / "integration.json"
                        audit = integration.audit()
                        if audit_path.exists():
                            raise RuntimeError("This experiment does not resume partial audit data")
                        write_json(audit_path, audit)
                    return variant, None

            results = await asyncio.gather(*(job(variant, row) for variant, row in jobs))
        scores = {}
        for variant, arm in VARIANTS.items():
            records = [r for v, r in results if v == variant and r is not None]
            submission = directory / variant / f"{arm}-submission.jsonl"
            write_json(submission.with_suffix(".records.json"), records)
            submission.write_text("".join(base.encoded(r) + "\n" for r in records))
            if len(records) == len(rows):
                scores[variant] = score_tracks(
                    rows, submission, [directory / variant / arm / str(r["id"]) for r in rows]
                )
        write_json(directory / "scores.json", scores)
        write_json(directory / "validation-budget-end.json", json.loads(ledger_path.read_text()))
        print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env")
    parser.add_argument("--output", default="results/dx-v2-benchmark")
    parser.add_argument("--preflight", action="store_true")
    asyncio.run(run(parser.parse_args()))
