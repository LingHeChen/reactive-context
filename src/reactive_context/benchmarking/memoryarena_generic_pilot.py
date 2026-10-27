"""Frozen MemoryArena integration rerun of generic admission and history projections.

No answer labels, new travel-solving rules, numeric filters or altered extraction
schedule enter the actor. Old model prompts, business tools and scorer are reused.
"""

import argparse
import asyncio
import fcntl
import json
import random
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .. import FactVerdict, HistoryManager
from . import memoryarena as upstream
from . import memoryarena_pilot as base
from .config import BenchmarkConfig
from .history_regression import write_json
from .official import ROOT, file_hash
from .plan_projection import final_plan_view, score_tracks
from .runtime import BudgetLedger, make_model

ARMS = ("fact-small", "fact-archive")


class GenericIntegration:
    """One instance per isolated episode: evidence registry and declared output projection."""

    def __init__(self):
        self.sources = {}
        self.decisions = Counter()
        self.projections = {}

    def final_view(self, text):
        # This boundary is explicitly declared by the upstream output contract.
        # It neither interprets field values nor compares against reference plans.
        result = final_plan_view(text)
        self.projections[sha256(text.encode()).hexdigest()] = {
            "original_token_proxy": base.token_count(text),
            "projected_token_proxy": base.token_count(result) if result else 0,
            "has_declared_boundary": result is not None,
        }
        return result

    def manager_options(self, store):
        return {"history": HistoryManager(store, base.ACTOR, final_projector=self.final_view)}

    def check(self, request):
        value = request.value
        if request.key != f"{value['subject']}/{value['topic']}":
            return FactVerdict("reject", ("key_value_identity_mismatch",))
        supports = [self.sources[e] for e in request.evidence if e in self.sources]
        if not supports:
            return FactVerdict("reject", ("unobserved_source",))
        for quote in value["quotes"]:
            if not any(
                quote in canonical or any(quote in text for text in texts)
                for texts, canonical in supports
            ):
                return FactVerdict("reject", ("unsupported_source_anchor",))
        return FactVerdict("accept", ("source_anchor_supported",))

    @staticmethod
    def source_support(source):
        texts = base.flatten(source)
        canonical = set()

        def visit(value):
            if not isinstance(value, str):
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if len(encoded) <= 3500:
                    canonical.add(encoded)
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(source)
        return texts, canonical

    async def commit(self, manager, facts, source, stage):
        support = self.source_support(source)

        async def persist(manager, key, fact, paths, evidence, previous):
            self.sources[evidence] = support
            result = await manager.save_fact(
                base.SCOPE,
                base.SCHEMA,
                key,
                fact.model_dump(),
                [evidence],
                expected_revision=previous.revision if previous else None,
            )
            self.decisions[result["decision"]] += 1
            receipt = {
                "key": key,
                "id": result.get("part_id"),
                "revision": result.get("revision"),
                "admission": result,
                "source_paths": paths,
                "evidence": evidence,
            }
            if result["decision"] in {"created", "updated"}:
                await manager.activate(base.SCOPE, result["part_id"])
                receipt.update(committed=True, activated=True)
            elif result["decision"] == "unchanged":
                receipt["unchanged"] = True
            else:
                receipt.update(status="rejected" if result["decision"] == "reject" else "skipped")
            return receipt

        return await base.commit_batch(
            manager,
            facts,
            source,
            stage,
            persist=persist,
            preserve_record_state=getattr(self, "preserve_record_state", False),
        )

    def audit(self):
        return {
            "save_decisions": dict(self.decisions),
            "projected_results": self.projections,
            "check_scope": (
                "schema, scoped evidence, exact source anchors and key identity; not business truth"
            ),
            "extraction_schedule": "unchanged: initial input and every completed stage",
        }


def source_hashes():
    paths = [
        Path(__file__),
        Path(base.__file__),
        Path(upstream.__file__),
        ROOT / "src/reactive_context/benchmarking/plan_projection.py",
        ROOT / "src/reactive_context/store.py",
        ROOT / "src/reactive_context/manager.py",
        ROOT / "src/reactive_context/history.py",
        ROOT / "src/reactive_context/history_content.py",
        ROOT / "src/reactive_context/fact_checks.py",
        ROOT / "src/reactive_context/adapters/history.py",
        ROOT / "src/reactive_context/adapters/operations.py",
        ROOT / "src/reactive_context/adapters/pydantic_ai.py",
    ]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}


async def run(args):
    directory = Path(args.output)
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    lock_path = ROOT / "results/v02/pilot.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = ROOT / "results/history-regression-budget.json"
        current = json.loads(ledger_path.read_text())
        authorization_path = directory / "authorization.json"
        if authorization_path.exists():
            authorization = json.loads(authorization_path.read_text())
        else:
            authorization = {
                "additional_requests": 500,
                "additional_cost_usd": 2,
                "ledger_start": current,
                "max_requests": current["requests"] + 500,
                "max_cost_usd": current["cost_upper_bound"]
                + current.get("outstanding_reservation", 0)
                + 2,
            }
            write_json(authorization_path, authorization)
        config = replace(
            BenchmarkConfig.load(args.env),
            max_requests=authorization["max_requests"],
            max_cost_usd=authorization["max_cost_usd"],
            max_output_tokens=6000,
        )
        rows = upstream.tasks(3)
        manifest = {
            **upstream.manifest(),
            "mode": "extension",
            "seed": 83,
            "arms": list(ARMS),
            "groups": [{"id": row["id"], "stages": len(row["questions"])} for row in rows],
            "config": config.public(),
            "harness_sha256": source_hashes(),
            "budget_ledger": "results/history-regression-budget.json",
            "baseline": "results/memoryarena-fact-fix-expanded",
            "integration": "generic checked save + clean index + declared final-result view",
            "unchanged": [
                "actor and writer instructions",
                "business tools",
                "questions",
                "initial inputs",
                "30 actor calls per stage",
                "extraction timing",
                "scorer",
            ],
            "check_limits": "provenance only; no new travel business constraints",
        }
        path = directory / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("resume manifest mismatch")
        write_json(path, manifest)
        write_json(directory / "budget-start.json", authorization["ledger_start"])
        ledger = BudgetLedger(config, ledger_path)
        engine = await asyncio.to_thread(upstream.executor)
        if args.preflight:
            print(json.dumps(manifest["groups"]))
            return
        sem = asyncio.Semaphore(2)
        async with make_model(config) as model:
            jobs = [(row, arm) for row in rows for arm in ARMS]
            random.Random(83).shuffle(jobs)

            async def job(row, arm):
                async with sem:
                    integration = GenericIntegration()
                    folder = directory / arm / str(row["id"])
                    try:
                        result = await base.episode(
                            row, arm, folder, model, ledger, engine, integration=integration
                        )
                        return arm, result
                    finally:
                        audit_path = folder / "integration.json"
                        audit = integration.audit()
                        if audit_path.exists():
                            previous = json.loads(audit_path.read_text())
                            audit["save_decisions"] = dict(
                                Counter(previous["save_decisions"])
                                + Counter(audit["save_decisions"])
                            )
                            audit["projected_results"] = {
                                **previous["projected_results"],
                                **audit["projected_results"],
                            }
                        write_json(audit_path, audit)

            results = await asyncio.gather(*(job(row, arm) for row, arm in jobs))
        scores = {}
        for arm in ARMS:
            submission = directory / f"{arm}-submission.jsonl"
            submission.write_text(
                "".join(base.encoded(record) + "\n" for name, record in results if name == arm)
            )
            scores[arm] = score_tracks(
                rows, submission, [directory / arm / str(r["id"]) for r in rows]
            )
        write_json(directory / "scores.json", scores)
        write_json(directory / "validation-budget-end.json", json.loads(ledger_path.read_text()))
        print(json.dumps(scores), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env")
    parser.add_argument("--output", default="results/memoryarena-generic-rerun")
    parser.add_argument("--preflight", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
