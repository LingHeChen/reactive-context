import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from reactive_context import Context, computed_part
from reactive_context.sources.config import ConfigSource


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=30))
@settings(max_examples=40, deadline=None)
def test_event_sequence_replay_has_deterministic_vectors(values):
    async def replay():
        source = ConfigSource("a", 0)

        async def compute(ctx):
            return ctx.dependencies["a"].value * 2

        derived = computed_part(key="double", schema=int, dependencies=["a"], compute=compute)
        vectors = []
        async with Context([source.part, derived]) as context:
            for value in values:
                source.set(value)
                snapshot = await context.next_context()
                assert snapshot.part("double").value == value * 2
                vectors.append(dict(snapshot.revision_vector))
                for part in snapshot.parts:
                    assert part.status == "CURRENT"
        return vectors

    assert asyncio.run(replay()) == asyncio.run(replay())


@given(st.integers(min_value=2, max_value=25))
@settings(max_examples=20, deadline=None)
def test_diamond_dirty_propagation_reaches_all_descendants(width):
    async def scenario():
        source = ConfigSource("root", 1)
        parts = [source.part]
        for index in range(width):

            async def compute(ctx):
                return ctx.dependencies["root"].value + 1

            parts.append(
                computed_part(
                    key=f"leaf-{index}", schema=int, dependencies=["root"], compute=compute
                )
            )

        async def total(ctx):
            return sum(part.value for part in ctx.dependencies.values())

        parts.append(
            computed_part(
                key="total",
                schema=int,
                dependencies=[f"leaf-{i}" for i in range(width)],
                compute=total,
            )
        )
        async with Context(parts) as context:
            first = await context.next_context()
            source.set(2)
            assert all(context.inspect(f"leaf-{i}")["state"] == "DIRTY" for i in range(width))
            assert context.inspect("total")["state"] == "DIRTY"
            second = await context.next_context()
            assert first.part("total").value == 2 * width
            assert second.part("total").value == 3 * width

    asyncio.run(scenario())
