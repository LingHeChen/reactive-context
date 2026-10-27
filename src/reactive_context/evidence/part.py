"""Research through the existing Part lifecycle and compiler, with explicit opt-in."""

from dataclasses import replace

from ..compiler.ir import ContextIRSegment
from ..core.part import source_part
from ..core.snapshot import digest, freeze, thaw
from .model import EvidenceBundle, EvidenceHit, EvidenceInsufficient, EvidenceRequirement


def restore_bundle(value):
    data = thaw(value)
    data["requirements"] = tuple(EvidenceRequirement(**r) for r in data["requirements"])
    data["hits"] = tuple(EvidenceHit(**h) for h in data["hits"])
    return EvidenceBundle(**data)


class EvidencePart:
    """Host-owned binding. No model calls, imports or IO at declaration time.

    Unversioned navigation is intentionally best_effort. Every new snapshot rechecks
    source access; strict_fresh rejects it rather than pretending atomic external reads.
    Shadow emits no segments. Expansion emits only user data with source references.
    """

    def __init__(
        self,
        key,
        researcher,
        requirements,
        *,
        candidates=(),
        mode="shadow",
        required=True,
        selector=None,
    ):
        if mode not in {"shadow", "dynamic", "research"}:
            raise ValueError("unknown research mode")
        self.key, self.researcher = key, researcher
        self.requirements = tuple(requirements)
        self.candidates = tuple(candidates)
        self.mode = mode
        self.selector = selector
        self.part = source_part(
            key=key,
            schema=dict,
            resolve=self.resolve,
            project=self.project,
            memoize=False,
            expose=mode != "shadow",
            required=required,
            timeout=researcher.budget.timeout_seconds + 1,
            kind="evidence",
            configuration={"mode": mode},
        )

    async def resolve(self, reader):
        if any(r.scope != reader.scope_id for r in self.requirements):
            raise ValueError("evidence requirements must match the bound scope")
        bundle = await self.researcher.run(self.requirements, self.candidates, mode=self.mode)
        return thaw(freeze(bundle))

    def project(self, value, reader):
        if self.mode == "shadow":
            return []
        return [
            ContextIRSegment(
                id=self.key,
                part_id=self.key,
                resource_id=self.key,
                source_revision=reader.revision or digest(value),
                scope_id=reader.scope_id,
                kind="evidence",
                requested_role="user",
                required=self.part.required,
                payload=(
                    self.selector.select(restore_bundle(value))
                    if self.selector
                    else {
                        "citations": value["excerpts"],
                        "sufficiency": value["sufficiency"],
                    }
                ),
            )
        ]

    async def compile(self, context, *, scope="default", history_tokens=0):
        if context.compiler.policy.research_mode == "off":
            raise ValueError("enable shadow or schema_coverage explicitly")
        if self.mode != "shadow" and context.compiler.policy.research_mode == "shadow":
            raise ValueError("shadow policy cannot render expanded evidence")
        snapshot = await context.next_context(scope)
        bundle = restore_bundle(snapshot.part(self.key).value)
        # Always use the same compiler and its budget/authority validation.
        compiled = context.compile(snapshot, evidence=bundle, history_tokens=history_tokens)
        if (
            self.selector is not None
            and compiled.token_estimate + history_tokens
            > self.selector.budget.actor_input_token_cap
        ):
            from .selection import BudgetUnsatisfiable

            raise BudgetUnsatisfiable(detail="total actor input exceeds execution cap")
        delivered = any(t.part_id == self.key and t.included for t in compiled.trace)
        if self.mode != "shadow" and not delivered and context.compiler.policy.evidence_fail_closed:
            raise EvidenceInsufficient(detail="retrieved evidence omitted by render budget")
        manifest = dict(compiled.manifest)
        manifest["evidence_delivery"] = freeze(
            {
                "part_key": self.key,
                "included": delivered,
                "delivery_mode": "diagnostics_only" if self.mode == "shadow" else "quoted_spans",
                "bundle_hash": digest(snapshot.part(self.key).value),
            }
        )
        return replace(compiled, manifest=freeze(manifest))
