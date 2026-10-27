import json
from dataclasses import replace

from ..core.snapshot import digest, freeze, thaw
from ..errors import AuthorityViolation, CompilerBudget, ResourceConflict
from .ir import CompiledContext, ProviderMessage, TraceDecision
from .layout import canonical_catalog, placement, render_key
from .policy import CompilerPolicy


class Compiler:
    def __init__(self, policy: CompilerPolicy | None = None):
        self.policy = policy or CompilerPolicy()

    def compile(self, snapshot, *, history_tokens=0, evidence=None):
        policy = self.policy
        from ..evidence.model import EvidenceInsufficient

        if evidence is not None and any(
            r.scope != snapshot.scope_id for r in evidence.requirements
        ):
            raise AuthorityViolation(detail="evidence belongs to a different scope")
        if policy.evidence_fail_closed and (
            evidence is None or evidence.sufficiency != "sufficient"
        ):
            raise EvidenceInsufficient(detail="required evidence unresolved")
        evidence_manifest = (
            evidence.manifest() if evidence is not None and policy.research_mode != "off" else None
        )
        trace, messages, tools, admitted = [], [], [], []
        used = history_tokens + policy.output_reserve
        if used > policy.max_tokens:
            raise CompilerBudget(detail="history and output reserve exceed budget")
        resources, namespaces = {}, {}
        segments = sorted(snapshot.segments, key=lambda s: (not s.required, -s.priority, s.id))
        for segment in segments:
            resource = segment.resource_id or segment.part_id
            role = segment.requested_role
            grant = policy.grants.get(segment.part_id)
            if role in {"system", "developer"} and grant != role:
                raise AuthorityViolation(segment.part_id)
            if role not in {"system", "developer", "user"}:
                raise AuthorityViolation(segment.part_id, "standalone tool roles are forbidden")
            if segment.kind == "instruction" and grant != role:
                raise AuthorityViolation(segment.part_id, "instruction requires host grant")
            text = (
                segment.payload
                if isinstance(segment.payload, str)
                else json.dumps(thaw(segment.payload), sort_keys=True, ensure_ascii=False)
            )
            if segment.kind not in {"instruction", "history"}:
                text = json.dumps(
                    {
                        "resource": resource,
                        "revision": segment.source_revision,
                        "status": segment.status,
                        "data": text,
                    },
                    ensure_ascii=False,
                )
            cost = policy.estimator(text)
            if cost < 0:
                raise ValueError("token estimator must be nonnegative")
            reason = segment.inclusion_reason
            include = True
            identity = (segment.source_revision, role, segment.kind, text)
            if segment.status != "CURRENT":
                include = policy.include_uncertain and segment.status in {"STALE", "UNKNOWN"}
                reason = "explicit uncertain data policy" if include else "invalid active source"
                if segment.kind == "instruction":
                    include = False
                    reason = "invalid active source"
            if include:
                if resource in resources and resources[resource] != identity:
                    raise ResourceConflict(segment.part_id, "multiple active resource revisions")
                if resource in resources:
                    include, reason = False, "duplicate resource/revision"
                if segment.exclusive_namespace:
                    prior = namespaces.get(segment.exclusive_namespace)
                    if prior and prior != resource:
                        raise ResourceConflict(segment.part_id, "exclusive namespace conflict")
            if include and used + cost > policy.max_tokens:
                if segment.required:
                    raise CompilerBudget(segment.part_id, "required segment cannot fit")
                include, reason = False, "optional segment exceeds budget"
            if not include and segment.required and reason == "invalid active source":
                raise ResourceConflict(segment.part_id, "required source is not current")
            if include:
                resources[resource] = identity
                if segment.exclusive_namespace:
                    namespaces[segment.exclusive_namespace] = resource
                used += cost
                admitted.append((segment, text))
            trace.append(
                TraceDecision(
                    segment.part_id,
                    resource,
                    segment.source_revision,
                    segment.status,
                    include,
                    reason,
                    cost,
                    policy.fingerprint,
                )
            )
        # Admission priority controls the budget, never the final prompt position.
        positions = {key: index for index, key in enumerate(policy.part_order)}
        admitted.sort(key=lambda item: render_key(item[0], policy, positions))
        placements = {}
        for segment, text in admitted:
            placements[segment.id] = placement(segment, policy)
            if segment.kind == "tool_definition":
                tools.append(freeze(canonical_catalog(segment.payload, policy)))
            else:
                messages.append(
                    ProviderMessage(
                        segment.requested_role,
                        text,
                        segment.resource_id or segment.part_id,
                        segment.source_revision,
                        segment.kind,
                        segment.part_id,
                        placement=placements[segment.id],
                    )
                )
        rendered_parts = {
            segment.part_id: (index, placements[segment.id])
            for index, (segment, _) in enumerate(admitted)
        }
        trace = [
            replace(
                item,
                render_index=rendered_parts[item.part_id][0],
                placement=rendered_parts[item.part_id][1],
            )
            if item.included
            else item
            for item in trace
        ]
        return CompiledContext(
            snapshot.id,
            tuple(messages),
            tuple(tools),
            tuple(trace),
            used,
            tuple(segment.id for segment, _ in admitted),
            revision_vector=snapshot.revision_vector,
            manifest=freeze(
                {
                    "evidence": evidence_manifest,
                    "experiment_id": policy.experiment_id,
                    "tokenizer_revision": policy.tokenizer_revision,
                    "history_projection_revision": policy.history_projection_revision,
                    "exact_resource_order": tuple(s.resource_id or s.part_id for s, _ in admitted),
                    "rendered_bytes_hash": digest(tuple((m.role, m.content) for m in messages)),
                    "rendered_tools_hash": digest(tuple(tools)),
                    "snapshot_id": snapshot.id,
                    "compiler_version": "2",
                    "policy_revision": snapshot.policy_revision,
                    "compiler_config_revision": snapshot.compiler_config_revision,
                    "scope_id": snapshot.scope_id,
                    "resource_versions": dict(snapshot.revision_vector),
                    "materializations": tuple(
                        {
                            "part_key": p.part_id,
                            "resource_id": p.resource_id,
                            "hash": digest(p.value),
                            "revision": p.revision,
                            "provenance": p.provenance,
                        }
                        for p in snapshot.parts
                    ),
                    "declarations": snapshot.metadata,
                    "fragments": tuple(
                        {
                            "part_key": item.part_id,
                            "included": item.included,
                            "reason": item.inclusion_reason,
                        }
                        for item in trace
                    ),
                    "rendered_messages_hash": digest(
                        tuple((m.role, m.content, m.resource_id, m.revision) for m in messages)
                    ),
                    "estimated_tokens": used,
                    "actual_tokens_if_available": None,
                }
            ),
            diagnostics=tuple(
                freeze(
                    {
                        "part_key": item.part_id,
                        "code": "PART_OMITTED",
                        "reason": item.inclusion_reason,
                    }
                )
                for item in trace
                if not item.included
            )
            + tuple((snapshot.metadata or {}).get("omissions", ()))
            + (
                (
                    freeze(
                        {
                            "code": "EVIDENCE_INSUFFICIENT",
                            "sufficiency": evidence.sufficiency,
                            "stop_reason": evidence.stop_reason,
                        }
                    ),
                )
                if (
                    evidence is not None
                    and evidence_manifest
                    and evidence.sufficiency != "sufficient"
                )
                else ()
            ),
        )
