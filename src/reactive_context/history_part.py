"""History projection as a computed Part; source dependencies may remain input-only."""

import json
from collections.abc import Callable
from dataclasses import asdict

from .compiler.ir import ContextIRSegment
from .compiler.policy import CompilerPolicy
from .core.part import computed_part
from .core.scheduler import invoke
from .core.snapshot import digest, thaw


def history_part(
    key,
    manager,
    *,
    load_groups,
    dependencies=(),
    get_revision=None,
    max_tokens: int | None = None,
    context_policy: CompilerPolicy | Callable[[], CompilerPolicy] | None = None,
    reserved_tokens: int | Callable[[], int] = 0,
    configuration=None,
    **kwargs,
):
    if context_policy is None and max_tokens is None:
        raise ValueError("history_part requires context_policy or an explicit legacy max_tokens")

    def budget():
        policy = context_policy() if callable(context_policy) else context_policy
        extra = reserved_tokens() if callable(reserved_tokens) else reserved_tokens
        if type(extra) is not int or extra < 0:
            raise ValueError("reserved tokens must be a nonnegative integer")
        return policy, extra

    async def revision(ctx):
        policy, extra = budget()
        factor = await invoke(get_revision, ctx) if get_revision else None
        return digest(
            (
                factor,
                policy.fingerprint if policy is not None else None,
                extra,
                max_tokens,
                asdict(manager.policy),
            )
        )

    def payload(summaries, groups):
        return {
            "kind": "historical_context",
            "authority": "untrusted_historical_data",
            "summaries": [
                {k: v for k, v in record.items() if k != "source_ids"} for record in summaries
            ],
            "recent_events": [thaw(e) for group in groups for e in group.events],
        }

    async def compute(ctx):
        groups = await invoke(load_groups, ctx)
        revisions = {p.resource_id: p.revision for p in ctx.dependencies.values()}
        policy, extra = budget()

        def measure(summaries, groups):
            assert policy is not None
            return policy.estimator(json.dumps(payload(summaries, groups), ensure_ascii=False))

        projection = await manager.prepare(
            ctx.scope_id,
            groups,
            max_tokens=max_tokens,
            revisions=revisions,
            total_tokens=policy.max_tokens if policy is not None else None,
            reserved_tokens=policy.output_reserve + extra if policy is not None else extra,
            measure=measure if policy is not None else None,
        )
        return payload(projection.summaries, projection.raw_groups)

    def project(value, ctx):
        return [
            ContextIRSegment(
                key,
                key,
                ctx.revision,
                json.dumps(thaw(value), ensure_ascii=False),
                kind="history",
                required=True,
                requested_role="user",
            )
        ]

    return computed_part(
        key=key,
        schema=dict,
        dependencies=list(dependencies),
        compute=compute,
        revision=revision,
        project=project,
        kind="history",
        configuration=configuration or {},
        **kwargs,
    )
