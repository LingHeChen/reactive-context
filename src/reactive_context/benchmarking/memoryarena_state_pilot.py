"""Paid state-retention pilot with model-proposed, source-anchored edit boundaries.

Uses the remaining existing authorization, never a fresh budget. No gold-derived
grants, numerical solving rules, or hardcoded shared-plan inheritance are supplied.
"""

import argparse
import asyncio
import fcntl
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, ToolOutput
from pydantic_ai.capabilities import Caching
from pydantic_ai.usage import UsageLimits

from .. import StateOperation, StateUpdateRejected
from ..core.snapshot import digest
from ..errors import SchemaInvalid
from . import memoryarena as upstream
from . import memoryarena_pilot as base
from .config import BenchmarkConfig
from .history_regression import append, write_json
from .memoryarena_generic_pilot import ARMS, GenericIntegration
from .official import ROOT, file_hash
from .runtime import AuditedModel, BudgetExceeded, BudgetLedger, CallTrace, make_model
from .travel_state import FIELDS, TravelStateAdapter


class EditIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day: int = Field(ge=1, le=60)
    field: Literal[
        "current_city",
        "transportation",
        "breakfast",
        "attraction",
        "lunch",
        "dinner",
        "accommodation",
    ]
    action: Literal["replace", "clear"] = "replace"
    quote: str = Field(min_length=4, max_length=1500)


class PlanIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(min_length=1, max_length=100)
    subject_quote: str = Field(min_length=4, max_length=1000)
    template_person: str = Field(min_length=1, max_length=100)
    template_reason: str = Field(min_length=1, max_length=1000)
    edits: list[EditIntent] = Field(min_length=1, max_length=100)


INTENT_INSTRUCTIONS = (
    "You extract edit boundaries for a structured work-state update; do not solve the trip. "
    "Only CURRENT QUERY, already-visible person names, day IDs, and field names are supplied. "
    "Identify the current traveler and cite an exact substring identifying them. "
    "Choose one available prior person's plan as the preservation template, based on the query. "
    "There is no hardcoded default inheritance rule. Explain your choice, including ambiguity. "
    "List ONLY day/field slots whose values the CURRENT traveler explicitly asks to change. "
    "A reference person's day/field used in a comparison is NOT itself a target to edit. "
    "Do not authorize transport, cities or attractions merely because a generic output template "
    "mentions them. Cite the exact current-query clause for every target. 'stay' means "
    "accommodation. Expand requests covering multiple target days. Use action replace for a new "
    "choice, clear only for an explicit cancellation or removal. Do not invent preferences, "
    "prices, cities, target values, answers, or future questions."
)


def validate_intent(value, query, available, days):
    intent = PlanIntent.model_validate(value)
    if intent.template_person not in available:
        raise ValueError("template must be an already visible person")
    if intent.subject_quote not in query or intent.subject not in intent.subject_quote:
        raise ValueError("subject quote must be an exact identifying source substring")
    targets = set()
    for edit in intent.edits:
        if edit.quote not in query or edit.day not in days:
            raise ValueError("edit requires a current-query source anchor and existing target day")
        target = (edit.day, edit.field)
        if target in targets:
            raise ValueError("duplicate target slot")
        targets.add(target)
    return intent


class IntentCache:
    """One sequential public-query extraction per stage, shared identically by both arms."""

    def __init__(self, directory, model, ledger):
        self.directory, self.model, self.ledger = directory, model, ledger
        self.locks = {}

    def previous(self, group, index):
        result = []
        for stage in range(1, index):
            path = self.directory / str(group) / f"{stage}.json"
            result.append(PlanIntent.model_validate_json(path.read_text()))
        return result

    async def get(self, group, initial, query, index):
        async with self.locks.setdefault((group, index), asyncio.Lock()):
            folder = self.directory / str(group)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{index}.json"
            available = [initial["name"], *(p.subject for p in self.previous(group, index))]
            days = [day.get("day", day.get("days")) for day in initial["daily_plans"]]
            if path.exists():
                return validate_intent(json.loads(path.read_text()), query, available, days)
            trace = CallTrace(capture_directory=folder / "requests")

            def checked(value: PlanIntent):
                try:
                    return validate_intent(value, query, available, days)
                except ValueError as error:
                    raise ModelRetry(str(error)) from None

            settings: Any = {**base.SETTINGS, "max_tokens": 2500}
            agent = Agent(
                AuditedModel(self.model, self.ledger, trace, "edit-intent"),
                output_type=PlanIntent,
                instructions=INTENT_INSTRUCTIONS,
                model_settings=settings,
                retries=1,
                capabilities=[Caching(messages=False)],
            )
            agent.output_validator(checked)
            try:
                result = await agent.run(
                    base.encoded(
                        {
                            "current_query": query,
                            "available_prior_people": available,
                            "existing_day_ids": days,
                            "field_names": FIELDS,
                        }
                    ),
                    usage_limits=UsageLimits(request_limit=2),
                )
                intent = result.output
                write_json(path, intent.model_dump())
                return intent
            finally:
                for call in trace.calls:
                    append(folder / "calls.jsonl", asdict(call))


class StateIntegration(GenericIntegration):
    actor_retries = 2
    actor_instructions = (
        "\nSTRUCTURED WORK STATE: The current travel.plan part is the selected preservation "
        "template plus host-issued edit permissions extracted from the CURRENT user query. "
        "Preserve EVERY unrequested field, exact spelling and transport details. Only edit "
        "permitted day/field paths; a missing field is not a cancellation. Use search tools "
        "and historical Facts to solve the requested changes. Finish by calling "
        "submit_plan_changes with ONE complete batch of operations on permitted fields. "
        "The host renders the complete final plan from committed state in the required "
        "format; do not regenerate untouched fields or output a free-text full plan. "
        "If you already applied your batch through context_update_work_state, finish "
        "with operations=[]. A rejected batch does not update state; repair only it. "
        "Do not request extra grants."
    )

    def __init__(self, group, folder, intents):
        super().__init__()
        self.group, self.folder, self.intents = group, folder, intents
        self.adapter = None
        self.intent = None
        self.permit = None
        self.state_events = Counter()

    def manager_options(self, store):
        result: dict[str, Any] = dict(super().manager_options(store))
        if self.adapter is not None:
            result["work_state"] = self.adapter.state
        return result

    async def prepare_stage(self, store, initial, query, index, persons):
        intent = await self.intents.get(self.group, initial, query, index)
        available = {initial["name"]: initial["daily_plans"]}
        for stage, (prior, person) in enumerate(
            zip(self.intents.previous(self.group, index), persons, strict=True), 1
        ):
            saved = TravelStateAdapter(store, base.ACTOR, key=f"travel.plan.stage-{stage}")
            previous_state = saved.state.read(base.SCOPE)
            # Structured state is authoritative after an observed update even if
            # the old free-text acknowledgement/formatting failed. Do not silently
            # substitute a seed-only state for a failed, never-updated person's plan.
            available[prior.subject] = (
                saved.plan(base.SCOPE)
                if previous_state["revision"]
                and (person["plan"] is not None or previous_state["revision"] > 1)
                else person["plan"]
            )
        visible_plan = available[intent.template_person]
        self.adapter = TravelStateAdapter(store, base.ACTOR, key=f"travel.plan.stage-{index}")
        state = self.adapter.state
        if state.read(base.SCOPE)["revision"] == 0:
            e = store.observe_evidence(
                base.ACTOR, base.SCOPE, f"visible-template:{index}", digest(visible_plan)
            )
            self.adapter.initialize(base.SCOPE, visible_plan, [e], host=base.HOST)
        current = state.read(base.SCOPE)
        # Interrupted stages may have committed state before output validation;
        # read its exact revision rather than resetting it or overwriting it.
        e = store.observe_evidence(
            base.ACTOR, base.SCOPE, f"current-request:{index}", digest(query)
        )
        self.permit = self.adapter.authorize_fields(
            base.SCOPE,
            [(edit.day, edit.field, edit.action) for edit in intent.edits],
            [e],
            expected_revision=current["revision"],
            host=base.HOST,
        )
        self.intent = intent
        write_json(
            self.folder / f"state-stage-{index}-input.json",
            {
                "intent": intent.model_dump(),
                "initial_state": state.read(base.SCOPE),
                "validation": (
                    "schema and exact source anchors; semantic boundaries are model-proposed"
                ),
            },
        )

    def output_type(self, store, index):
        adapter, intent, permit = self.adapter, self.intent, self.permit
        assert adapter is not None and intent is not None

        def submit_plan_changes(
            operations: Annotated[list[StateOperation], Field(max_length=100)],
        ) -> str:
            """Commit requested field edits and return the complete, host-rendered plan.
            Only supply changed fields. Omitted fields are retained exactly.
            Use [] after an already successful native state update, or if no values change.
            """
            before = adapter.state.read(base.SCOPE)
            try:
                receipt = (
                    adapter.state.patch(
                        base.SCOPE,
                        operations,
                        expected_revision=before["revision"],
                        permit_id=permit,
                    )
                    if operations
                    else {
                        "decision": "unchanged",
                        "revision": before["revision"],
                        "reasons": [],
                        "changes": [],
                    }
                )
            except (StateUpdateRejected, SchemaInvalid, ValueError) as error:
                self.state_events["output_rejected"] += 1
                message: dict[str, Any] = {"error": type(error).__name__}
                if isinstance(error, StateUpdateRejected):
                    message["changes"] = [
                        {"path": c.path, "action": c.action} for c in error.changes
                    ]
                    message["reasons"] = list(error.reasons)
                else:
                    message["reason"] = "use a valid nonoverlapping batch of permitted state paths"
                append(self.folder / "state-output-checks.jsonl", {"stage": index, **message})
                raise ModelRetry(
                    "Patch rejected. Retain unpermitted fields; submit only allowed changes. "
                    + base.encoded(message)
                ) from None
            self.state_events["output_accepted"] += 1
            append(self.folder / "state-output-checks.jsonl", {"stage": index, **receipt})
            write_json(
                self.folder / f"state-stage-{index}-final.json", adapter.state.read(base.SCOPE)
            )
            return adapter.render(base.SCOPE, intent.subject)

        return ToolOutput(submit_plan_changes, name="submit_plan_changes")

    def audit(self):
        return {**super().audit(), "state_events": dict(self.state_events)}


def hashes():
    paths = [*sorted((ROOT / "src/reactive_context").rglob("*.py"))]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


async def run(args, *, integration_factory=StateIntegration, manifest_extra=None):
    directory = Path(args.output)
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    with (ROOT / "results/v02/pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allocation_path = await asyncio.to_thread(Path(args.authorization).resolve)
        allocation = json.loads(allocation_path.read_text())
        ledger_path = ROOT / "results/history-regression-budget.json"
        start = json.loads(ledger_path.read_text())
        config = replace(
            BenchmarkConfig.load(args.env),
            max_requests=allocation["max_requests"],
            max_cost_usd=allocation["max_cost_usd"],
            max_output_tokens=6000,
        )
        rows = upstream.tasks(3)
        if args.group_ids:
            requested = set(args.group_ids)
            rows = [row for row in rows if row["id"] in requested]
            if {row["id"] for row in rows} != requested:
                raise ValueError("group IDs must come from the existing paired pilot")
        manifest = {
            **upstream.manifest(),
            "mode": "state-retention-extension",
            "seed": 83,
            "arms": list(ARMS),
            "groups": [{"id": r["id"], "stages": len(r["questions"])} for r in rows],
            "config": config.public(),
            "budget_ledger": "results/history-regression-budget.json",
            "authorization_source": str(allocation_path.relative_to(ROOT)),
            "harness_sha256": hashes(),
            "baseline": "results/memoryarena-generic-rerun",
            "intent_extraction": "once per current query, shared across arms, all calls charged",
            "intent_validation": (
                "schema+source anchors; no semantic guarantee or gold-derived permits"
            ),
            "template_selection": (
                "model chooses an available prior person; no fixed initial-person rule"
            ),
            "output_protocol": (
                "atomic final patch ToolOutput then host render; no full-text rewrite"
            ),
            "intent_cache": str(Path(args.intent_cache)) if args.intent_cache else None,
            "actor_output_retries": 2,
            "actor_request_limit": 30,
            "business_tools_and_scorer": "unchanged",
            "extraction_schedule": "unchanged",
            **(manifest_extra or {}),
        }
        path = directory / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("resume manifest mismatch")
        write_json(path, manifest)
        if not (directory / "budget-start.json").exists():
            write_json(directory / "budget-start.json", start)
        write_json(
            directory / "authorization.json",
            {
                "reuses_existing_allocation": True,
                "source": str(allocation_path.relative_to(ROOT)),
                "max_requests": config.max_requests,
                "max_cost_usd": config.max_cost_usd,
            },
        )
        if args.preflight:
            print(
                json.dumps(
                    {
                        "groups": manifest["groups"],
                        "remaining_requests": config.max_requests - start["requests"],
                        "remaining_usd": config.max_cost_usd - start["cost_upper_bound"],
                    }
                )
            )
            return
        ledger = BudgetLedger(config, ledger_path)
        engine = await asyncio.to_thread(upstream.executor)
        sem = asyncio.Semaphore(2)
        exhausted = asyncio.Event()
        async with make_model(config) as model:
            intents = IntentCache(
                Path(args.intent_cache) if args.intent_cache else directory / "intents",
                model,
                ledger,
            )

            async def job(row, arm):
                async with sem:
                    if exhausted.is_set():
                        return None
                    folder = directory / arm / str(row["id"])
                    folder.mkdir(parents=True, exist_ok=True)
                    integration = integration_factory(row["id"], folder, intents)
                    try:
                        return await base.episode(
                            row, arm, folder, model, ledger, engine, integration=integration
                        )
                    except BudgetExceeded:
                        exhausted.set()
                        append(
                            directory / "failures.jsonl",
                            {"arm": arm, "group": row["id"], "error": "budget_exceeded"},
                        )
                    except Exception as error:
                        append(
                            directory / "failures.jsonl",
                            {
                                "arm": arm,
                                "group": row["id"],
                                "error": type(error).__name__,
                                "message": str(error)[:1000],
                            },
                        )
                    finally:
                        audit = integration.audit()
                        audit_path = folder / "integration.json"
                        if audit_path.exists():
                            previous = json.loads(audit_path.read_text())
                            for key in ("save_decisions", "state_events"):
                                audit[key] = dict(
                                    Counter(previous.get(key, {})) + Counter(audit.get(key, {}))
                                )
                            audit["projected_results"] = {
                                **previous["projected_results"],
                                **audit["projected_results"],
                            }
                        write_json(audit_path, audit)

            # Finish the formerly regressing small-history arm before archive jobs.
            # Every job still uses the same isolated groups and sequential stages.
            for arm in ARMS:
                results = await asyncio.gather(*(job(row, arm) for row in rows))
                submission = directory / f"{arm}-submission.jsonl"
                submission.write_text(
                    "".join(base.encoded(r) + "\n" for r in results if r is not None)
                )
        scores = {}
        for arm in ARMS:
            complete = all(
                (directory / arm / str(row["id"]) / "checkpoint.json").exists()
                and json.loads((directory / arm / str(row["id"]) / "checkpoint.json").read_text())[
                    "completed"
                ]
                == len(row["questions"])
                for row in rows
            )
            if complete:
                scores[arm] = upstream.score(rows, directory / f"{arm}-submission.jsonl")
        write_json(directory / "scores.json", scores)
        write_json(directory / "validation-budget-end.json", json.loads(ledger_path.read_text()))
        print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env")
    parser.add_argument("--output", default="results/memoryarena-state-patch-rerun")
    parser.add_argument(
        "--authorization", default="results/memoryarena-generic-rerun/authorization.json"
    )
    parser.add_argument("--intent-cache")
    parser.add_argument("--group-ids", type=int, nargs="+")
    parser.add_argument("--preflight", action="store_true")
    asyncio.run(run(parser.parse_args()))
