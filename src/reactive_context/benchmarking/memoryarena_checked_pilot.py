"""Checked-candidate benchmark with real observations and deterministic delta Facts."""

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import Agent, ModelRetry, ToolOutput
from pydantic_ai.capabilities import Caching
from pydantic_ai.usage import UsageLimits

from .. import StateOperation, StateUpdateRejected, source_part
from ..core.snapshot import digest
from ..errors import SchemaInvalid
from . import memoryarena_pilot as base
from . import memoryarena_state_pilot as state_pilot
from .history_regression import append, write_json
from .official import ROOT, file_hash
from .runtime import AuditedModel, CallTrace
from .travel_constraints import ConstraintBundle, ObservedTravelConstraints
from .travel_state import TravelStateAdapter

CONSTRAINT_INSTRUCTIONS = (
    "Translate ONLY the provided current query and edit slots into executable constraints. "
    "Do not solve the task or select venues. Every target must appear exactly once and every "
    "clause must quote an exact current-query substring. Target fields refer to the CURRENT "
    "traveler; comparison references refer to already available prior people. Attributes: "
    "choice (identity/join), price, rating, cuisines (restaurant set), room_type, rules (hotel "
    "set), capacity. Operators ge/le/gt/lt/eq/ne, contains_all, overlaps. The right operand is "
    "either literal value or reference {person,day,field,attribute}; numeric result is "
    "operand*scale+offset. All clauses are conjunctive. 'within 20%' => ge with scale .8 AND "
    "le with scale 1.2; 'within $50' => ge offset -50 AND le offset 50; 'at least 30% more' "
    "=> ge scale 1.3; do not round. 'join X' => choice eq reference choice. Hotel room types "
    "are Entire home/apt, Private room, Shared room. House rules include No pets, No smoking, "
    "No parties, No visitors, No children under 10. 'serves A and B' => contains_all; 'shares "
    "at least one cuisine/house rule' => overlaps reference set. Capacity fits N => ge N. "
    "Explicit cancellation => choice eq '-'. For a numerical interval output both bounds. "
    "Do not add unstated constraints, infer future requirements, or default inheritance."
)


class ConstraintCache:
    def __init__(self, directory, model, ledger):
        self.directory, self.model, self.ledger = directory, model, ledger

    async def get(self, group, index, intent, query, available, days):
        folder = self.directory / str(group)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{index}.json"
        targets = {(e.day, e.field) for e in intent.edits}

        def validate(bundle: ConstraintBundle):
            slots = [(s.day, s.field) for s in bundle.slots]
            if set(slots) != targets or len(slots) != len(targets):
                raise ModelRetry("Cover exactly the supplied target slots once each.")
            for slot in bundle.slots:
                for clause in slot.clauses:
                    if clause.quote not in query:
                        raise ModelRetry("Every clause needs an exact CURRENT QUERY quote.")
                    allowed = (
                        {"choice", "price", "rating", "room_type", "rules", "capacity"}
                        if slot.field == "accommodation"
                        else {"choice", "price", "rating", "cuisines"}
                    )
                    if clause.attribute not in allowed:
                        raise ModelRetry("Attribute does not apply to this target kind.")
                    if clause.reference and (
                        clause.reference.person not in available or clause.reference.day not in days
                    ):
                        raise ModelRetry(
                            "References must be already available people and valid days."
                        )
            return bundle

        if path.exists():
            return validate(ConstraintBundle.model_validate_json(path.read_text()))
        trace = CallTrace(capture_directory=folder / "requests")
        settings: Any = {**base.SETTINGS, "max_tokens": 6500}
        agent = Agent(
            AuditedModel(self.model, self.ledger, trace, "constraint-extraction"),
            output_type=ConstraintBundle,
            instructions=CONSTRAINT_INSTRUCTIONS,
            retries=1,
            model_settings=settings,
            capabilities=[Caching(messages=False)],
        )
        agent.output_validator(validate)
        try:
            result = await agent.run(
                base.encoded(
                    {
                        "current_query": query,
                        "target_slots": [e.model_dump() for e in intent.edits],
                        "available_prior_people": list(available),
                        "day_ids": days,
                    }
                ),
                usage_limits=UsageLimits(request_limit=2),
            )
            write_json(path, result.output.model_dump())
            return result.output
        finally:
            for call in trace.calls:
                append(folder / "calls.jsonl", base.asdict(call))


class RecordingExecutor:
    def __init__(self, integration, engine):
        self.integration, self.engine = integration, engine

    def execute(self, name, args):
        if name not in {"RestaurantSearch", "AccommodationSearch"}:
            return self.engine.execute(name, args)
        # The original API and formatter are executed unchanged. Only rows in the
        # actual returned DataFrame become trusted observations; no database oracle.
        api = self.engine.restaurants if name == "RestaurantSearch" else self.engine.accommodations
        raw = api.run(city=args["city"])
        text = self.engine._format_result(raw, name)
        integration = self.integration
        e = integration.store.observe_evidence(
            base.ACTOR, base.SCOPE, f"tool:{name}:{integration.index}", digest((args, text))
        )
        if hasattr(raw, "to_dict"):
            integration.checker.observe(
                "restaurant" if name == "RestaurantSearch" else "hotel",
                raw.to_dict(orient="records"),
                e,
                host=base.HOST,
            )
        return {"result": text, "evidence_id": e}


class CheckedIntegration(state_pilot.StateIntegration):
    actor_instructions = state_pilot.StateIntegration.actor_instructions.replace(
        "If you already applied your batch through context_update_work_state, finish "
        "with operations=[]. ",
        "",
    ) + (
        "\nVALUE CHECKS: submit_plan_changes stages a candidate, validates every declared "
        "constraint against observed tool rows, and commits only passed candidates. Failed "
        "or pending candidates never replace current state. Inspect exact issue expected/actual "
        "values and repair the relevant choices. Search the relevant city if a target or "
        "comparison option is not observed. Do not invent numeric attributes. Use one complete "
        "batch. Finalization binds actual observed source IDs automatically. Historical person "
        "Facts are plan-reference plus per-day deltas: follow template_person when a field is "
        "not in a person's changes. Initial plan Facts remain explicit."
    )

    def __init__(self, group, folder, intents):
        super().__init__(group, folder, intents)
        self.store, self.checker, self.index = None, None, 0
        self.constraints = ConstraintCache(
            intents.directory.parent / "constraints", intents.model, intents.ledger
        )

    def executor(self, engine):
        return RecordingExecutor(self, engine)

    def manager_options(self, store):
        options = super().manager_options(store)
        # This workflow owns its one atomic finalization tool. Installing the
        # complete alternate mutation API creates competing paths and unbounded
        # ordinary-tool retries; the same real candidate engine is used below.
        options.pop("work_state", None)
        return options

    def context_parts(self, store):
        assert self.adapter is not None
        state = self.adapter.state

        def payload(scope):
            value = state.read(scope)
            value.pop("maintenance", None)
            value["workflow"] = {
                "finalize_tool": "submit_plan_changes",
                "rule": (
                    "Submit one complete patch. Change failing choices on rejection; "
                    "do not repeat an identical patch."
                ),
            }
            return value

        async def resolve(ctx):
            return payload(ctx.scope_id)

        async def revision(ctx):
            return digest(payload(ctx.scope_id))

        return [
            source_part(
                key=state.key, schema=dict, resolve=resolve, revision=revision, kind="work_state"
            )
        ]

    async def prepare_stage(self, store, initial, query, index, persons):
        await super().prepare_stage(store, initial, query, index, persons)
        assert self.intent is not None
        self.store, self.index = store, index
        plans = {initial["name"]: initial["daily_plans"]}
        for stage, (intent, person) in enumerate(
            zip(self.intents.previous(self.group, index), persons, strict=True), 1
        ):
            saved = TravelStateAdapter(store, base.ACTOR, key=f"travel.plan.stage-{stage}")
            record = saved.state.read(base.SCOPE)
            plans[intent.subject] = (
                saved.plan(base.SCOPE)
                if record["revision"] and (person["plan"] is not None or record["revision"] > 1)
                else person["plan"]
            )
        bundle = await self.constraints.get(
            self.group,
            index,
            self.intent,
            query,
            plans,
            [day.get("day", day.get("days")) for day in initial["daily_plans"]],
        )
        self.checker = ObservedTravelConstraints(store, base.ACTOR, base.SCOPE, plans)
        self.adapter = TravelStateAdapter(
            store,
            base.ACTOR,
            key=f"travel.plan.stage-{index}",
            check=self.checker,
            check_id=self.checker.check_id,
        )
        state = self.adapter.state
        e = store.observe_evidence(
            base.ACTOR, base.SCOPE, f"quoted-constraints:{index}", digest(bundle.model_dump())
        )
        current = state.requirements(base.SCOPE)
        state.set_requirements(
            base.SCOPE,
            {"constraints": bundle.model_dump()},
            [e],
            expected_revision=current["revision"],
            host=base.HOST,
        )
        self.permit = self.adapter.authorize_fields(
            base.SCOPE,
            [(v.day, v.field, v.action) for v in self.intent.edits],
            [e],
            expected_revision=state.read(base.SCOPE)["revision"],
            host=base.HOST,
        )
        write_json(
            self.folder / f"state-stage-{index}-input.json",
            {
                "intent": self.intent.model_dump(),
                "initial_state": state.read(base.SCOPE),
                "constraint_translation": (
                    "model-extracted; exact quotes checked, semantics not assumed infallible"
                ),
            },
        )

    def output_type(self, store, index):
        adapter, intent, permit = self.adapter, self.intent, self.permit
        checker = self.checker
        assert adapter is not None and intent is not None and checker is not None

        def submit_plan_changes(
            operations: Annotated[list[StateOperation], Field(max_length=100)],
        ) -> str:
            state = adapter.state
            current = state.read(base.SCOPE)
            evidence = list(
                dict.fromkeys([*checker.sources(), *state.requirements(base.SCOPE)["evidence"]])
            )
            try:
                return finalize(operations, state, current, evidence)
            except (StateUpdateRejected, SchemaInvalid, ValueError) as error:
                self.state_events["output_rejected"] += 1
                details = error.validation if isinstance(error, StateUpdateRejected) else None
                append(
                    self.folder / "state-output-checks.jsonl",
                    {
                        "stage": index,
                        "error": type(error).__name__,
                        "validation": details,
                    },
                )
                raise ModelRetry(
                    "Invalid candidate patch. Use permitted paths and string slot "
                    "values; inspect current state. " + base.encoded(details)
                ) from None

        def finalize(operations, state, current, evidence):
            if operations:
                candidate = state.candidates.propose(
                    base.SCOPE,
                    operations,
                    evidence,
                    expected_revision=current["revision"],
                    permit_id=permit,
                )
                if candidate["status"] == "ready":
                    candidate = state.candidates.commit(base.SCOPE, candidate["candidate_id"])
                receipt = candidate["last_check"]
                status = candidate["status"]
            else:
                receipt = state.replace(
                    base.SCOPE,
                    current["value"],
                    expected_revision=current["revision"],
                    evidence=evidence,
                    preview=True,
                )
                status = (
                    "committed"
                    if receipt["validation"]["status"] == "passed"
                    and receipt["decision"] != "reject"
                    else "pending"
                )
            if status != "committed":
                self.state_events["output_rejected"] += 1
                append(
                    self.folder / "state-output-checks.jsonl",
                    {
                        "stage": index,
                        "error": "candidate_not_committed",
                        "status": status,
                        "receipt": receipt,
                    },
                )
                raise ModelRetry(
                    "Candidate rejected. Choose DIFFERENT values for the failing slots. "
                    "An identical patch cannot pass. Retrieve missing source rows if needed. "
                    + base.encoded(
                        {
                            "status": status,
                            "reasons": receipt.get("reasons", []),
                            "issues": receipt.get("validation", {}).get("issues", []),
                        }
                    )
                )
            self.state_events["output_accepted"] += 1
            append(self.folder / "state-output-checks.jsonl", {"stage": index, **receipt})
            write_json(self.folder / f"state-stage-{index}-final.json", state.read(base.SCOPE))
            return adapter.render(base.SCOPE, intent.subject)

        return ToolOutput(submit_plan_changes, name="submit_plan_changes")

    async def write_facts(self, manager, source, stage):
        """Project typed observed seed/deltas directly. Never calls the extraction model."""
        facts = []
        if stage == 0:
            initial = source["given_fixed_initial_plan"]
            for i, day in enumerate(initial["daily_plans"]):
                d = day.get("day", day.get("days"))
                facts.append(
                    base.FactDraft(
                        subject=initial["name"],
                        topic=f"day-{d}",
                        content=base.encoded(day),
                        anchors=[
                            base.SourceAnchor(path=f"/given_fixed_initial_plan/daily_plans/{i}")
                        ],
                    )
                )
        else:
            assert self.intent is not None and self.adapter is not None
            if not source["generated_plan"]:
                return []
            state = self.adapter.state
            reference = {
                "subject": self.intent.subject,
                "template_person": self.intent.template_person,
                "state_key": state.key,
                "revision": state.read(base.SCOPE)["revision"],
            }
            source["plan_reference"] = reference
            facts.append(
                base.FactDraft(
                    subject=self.intent.subject,
                    topic="plan-reference",
                    content=base.encoded(reference),
                    anchors=[base.SourceAnchor(path="/plan_reference")],
                )
            )
            by_day = defaultdict(list)
            for delta in state.deltas(base.SCOPE):
                for change in delta["changes"]:
                    parts = change["path"].split("/")
                    by_day[parts[2]].append(
                        {"field": parts[3], "action": change["action"], "value": change["after"]}
                    )
            source["day_deltas"] = {
                day: {"template_person": self.intent.template_person, "changes": changes}
                for day, changes in by_day.items()
            }
            for day, data in source["day_deltas"].items():
                facts.append(
                    base.FactDraft(
                        subject=self.intent.subject,
                        topic=f"day-{day}",
                        content=base.encoded(data),
                        anchors=[base.SourceAnchor(path=f"/day_deltas/{day}")],
                    )
                )
        result = await self.commit(manager, facts, source, stage)
        self.state_events["deterministic_memory_batches"] += 1
        return result


async def main(args):
    directory = Path(args.output)
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    cached = {}
    old = ROOT / Path(args.seed_run)
    for part in ("intents", "constraints"):
        for group in (218, 164, 36):
            for source in (old / part / str(group)).glob("[0-9]*.json"):
                destination = directory / part / str(group) / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
                cached[str(source.relative_to(ROOT))] = file_hash(source)
    args.intent_cache = None
    await state_pilot.run(
        args,
        integration_factory=CheckedIntegration,
        manifest_extra={
            "variant": "checked-candidates-and-structured-delta-memory",
            "prepaid_inputs": cached,
            "candidate_tool_surface": (
                "one final patch tool; alternate native state mutation tools not installed"
            ),
            "business_tools_and_scorer": (
                "original APIs, formatter, schemas and scorer; evidence metadata added"
            ),
            "extraction_schedule": (
                "deterministic initial seed and committed state deltas; zero LLM Fact extraction"
            ),
            "validation_scope": (
                "typed restaurant/hotel/identity/relative constraints from current quoted request"
            ),
            "translation_limits": (
                "quote provenance checked; natural-language expression semantics may be wrong"
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env")
    parser.add_argument("--output", default="results/memoryarena-checked-state-rerun")
    parser.add_argument(
        "--authorization", default="results/memoryarena-checked-state-rerun/allocation.json"
    )
    parser.add_argument("--group-ids", type=int, nargs="+")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seed-run", default="results/memoryarena-state-rerun")
    asyncio.run(main(parser.parse_args()))
