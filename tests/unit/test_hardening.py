import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import BaseModel

from reactive_context import Context, computed_part, source_part
from reactive_context.errors import AccessRevoked, ConsistencyUnavailable
from reactive_context.sources.config import ConfigSource


async def test_polling_alone_advances_generation_and_dirties_computed():
    from dataclasses import replace

    source = ConfigSource("a", 1)

    async def compute(ctx):
        return ctx.dependencies["a"].value * 2

    derived = computed_part(key="double", schema=int, dependencies=["a"], compute=compute)
    async with Context([replace(source.part, subscribe=None), derived]) as context:
        first = await context.next_context()
        source.set(2)
        second = await context.next_context()
        assert second.part("a").generation > first.part("a").generation
        assert second.part("double").generation > first.part("double").generation
        assert second.part("double").value == 4


async def test_unversioned_ttl_expires_without_sliding_verification():
    calls = 0

    async def resolve(ctx):
        nonlocal calls
        calls += 1
        return calls

    async with Context([source_part(key="ttl", schema=int, resolve=resolve, ttl=0.03)]) as context:
        first = await context.next_context()
        second = await context.next_context()
        assert first.part("ttl").verified_at == second.part("ttl").verified_at
        assert calls == 1
        await asyncio.sleep(0.04)
        third = await context.next_context()
        assert third.part("ttl").value == 2


async def test_nested_models_and_immutable_scalar_types():
    class Nested(BaseModel):
        values: list[int]

    class Schema(BaseModel):
        nested: Nested
        timestamp: datetime
        price: Decimal

    source = ConfigSource(
        "a",
        {
            "nested": {"values": [1]},
            "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
            "price": Decimal("1.25"),
        },
        schema=Schema,
    )
    async with Context([source.part]) as context:
        snapshot = await context.next_context()
        assert snapshot.part("a").value.nested.values == (1,)
        assert snapshot.part("a").value.price == Decimal("1.25")
        context.compile(snapshot)


async def test_projection_race_retries_instead_of_mixing_versions():
    source = ConfigSource("a", "A")
    invoked = False

    async def project(value, ctx):
        nonlocal invoked
        if not invoked:
            invoked = True
            source.set("B")
        from reactive_context import ContextIRSegment

        return [ContextIRSegment("a", "a", ctx.revision, value)]

    from dataclasses import replace

    async with Context([replace(source.part, project=project)]) as context:
        snapshot = await context.next_context()
        assert snapshot.part("a").value == "B"
        assert '"data": "A"' not in str(context.compile(snapshot))


async def test_extend_keeps_host_revocation():
    async with Context([ConfigSource("a", "secret").part]) as context:
        context.revoke("a")
        async with context.extend(parts=[ConfigSource("b", "other").part]) as extended:
            with pytest.raises(AccessRevoked):
                await extended.next_context()


async def test_custom_revision_mismatch_hook_has_bounded_retry():
    calls = 0

    async def resolve(ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConsistencyUnavailable("a")
        return "B"

    async with Context(
        [source_part(key="a", schema=str, resolve=resolve, revision=lambda ctx: "B")]
    ) as context:
        assert (await context.next_context()).part("a").value == "B"
        assert calls == 2


async def test_close_while_resolving_never_commits():
    entered, release = asyncio.Event(), asyncio.Event()

    async def resolve(ctx):
        entered.set()
        await release.wait()
        return "A"

    context = Context([source_part(key="a", schema=str, resolve=resolve, revision=lambda ctx: "A")])
    pending = asyncio.create_task(context.next_context())
    await entered.wait()
    closing = asyncio.create_task(context.close())
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await closing
    assert not context.registry.instances


async def test_action_guard_detects_changes_and_revocation():
    source = ConfigSource("a", "A")
    async with Context([source.part]) as context:
        snapshot = await context.next_context()
        await context.guard(snapshot)
        source.set("B")
        with pytest.raises(ConsistencyUnavailable):
            await context.guard(snapshot)
        new = await context.next_context()
        context.revoke("a")
        with pytest.raises(AccessRevoked):
            await context.guard(new)


async def test_plugin_cannot_downgrade_host_required_segments():
    from reactive_context import CompilerPolicy, ContextIRSegment
    from reactive_context.errors import CompilerBudget

    def project(value, ctx):
        return [ContextIRSegment("a", "a", ctx.revision, value, required=False)]

    source = ConfigSource("a", "x" * 100, project=project)
    async with Context(
        [source.part], compiler_policy=CompilerPolicy(max_tokens=10, output_reserve=0)
    ) as context:
        with pytest.raises(CompilerBudget):
            context.compile(await context.next_context())


async def test_uncertain_policy_never_silently_drops_required_instruction():
    from dataclasses import replace

    from reactive_context import CompilerPolicy, ContextIRSegment
    from reactive_context.errors import ResourceConflict

    source = ConfigSource(
        "policy",
        "trusted-policy",
        project=lambda value, ctx: [
            ContextIRSegment(
                "policy", "policy", ctx.revision, value, kind="instruction", requested_role="system"
            )
        ],
    )
    async with Context(
        [source.part],
        compiler_policy=CompilerPolicy(grants={"policy": "system"}, include_uncertain=True),
    ) as context:
        snapshot = await context.next_context()
        uncertain = replace(
            snapshot,
            segments=tuple(replace(segment, status="STALE") for segment in snapshot.segments),
        )
        with pytest.raises(ResourceConflict):
            context.compile(uncertain)
        assert context.explain(snapshot.id)[0].status == "ERROR"


async def test_revocation_signal_before_initialization_covers_future_scopes():
    from reactive_context import ChangeSignal

    async with Context([ConfigSource("a", "secret").part]) as context:
        context.notify(ChangeSignal("a", revoked=True))
        with pytest.raises(AccessRevoked):
            await context.next_context("not-created-yet")
