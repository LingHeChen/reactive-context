from reactive_context import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    HistoryGroup,
    HistoryManager,
    HistoryPolicy,
    Principal,
    SummaryDraft,
    history_part,
)
from reactive_context.core.snapshot import thaw
from reactive_context.sources.config import ConfigSource


async def test_input_only_history_source_compacts_without_summarizing_resource(tmp_path):
    host = Principal("host", True)
    raw = ConfigSource(
        "raw-history",
        [{"id": i, "content": "x" * 1000} for i in range(10)],
        kind="history",
        expose=False,
    )
    skill = ConfigSource("skill", "EXACT_CURRENT_RESOURCE_BODY")
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")
        calls = []

        def compact(events):
            calls.append(events)
            return SummaryDraft(
                text="Earlier historical notes", evidence_ids=[HistoryManager.event_id(events[0])]
            )

        history = HistoryManager(
            store,
            host,
            compactor=compact,
            validator=lambda events, draft: bool(draft.evidence_ids),
            policy=HistoryPolicy(projection="compact", recent_groups=2, segment_groups=4),
        )

        def load(ctx):
            return tuple(HistoryGroup((thaw(e),)) for e in ctx.dependencies["raw-history"].value)

        part = history_part(
            "history", history, load_groups=load, dependencies=["raw-history"], max_tokens=4000
        )
        async with ContextManager(
            [raw.part, skill.part, part],
            store=store,
            principal=host,
            mandatory=["history"],
            compiler_policy=CompilerPolicy(max_tokens=20000),
        ) as context:
            context.publish_catalog(host, "chat")
            row = next(r for r in await context.query("chat") if r["key"] == "skill")
            await context.activate("chat", row["id"])
            snapshot = await context.next_context("chat")
            compiled = context.compile(snapshot)
            assert "raw-history" not in [m.part_id for m in compiled.provider_messages]
            assert "EXACT_CURRENT_RESOURCE_BODY" in str(compiled.provider_messages)
            assert any(m.part_id == "history" for m in compiled.provider_messages)
            before = len(calls)
            await context.next_context("chat")
            assert len(calls) == before


async def test_context_policy_is_required_when_no_legacy_history_cap(tmp_path):
    import pytest

    with ContextStore(tmp_path) as store:
        history = HistoryManager(store, Principal("host", True))
        with pytest.raises(ValueError, match="context_policy"):
            history_part("history", history, load_groups=lambda ctx: ())


async def test_history_part_recomputes_when_context_budget_or_reservations_change(tmp_path):
    from dataclasses import replace

    host = Principal("host", True)
    raw = ConfigSource(
        "raw", [{"id": i, "content": "x" * 1000} for i in range(10)], kind="history", expose=False
    )
    calls = []
    extra = [0]
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")

        def compact(events):
            calls.append(events)
            return SummaryDraft(
                text="Earlier notes", evidence_ids=[HistoryManager.event_id(events[0])]
            )

        history = HistoryManager(
            store,
            host,
            compactor=compact,
            validator=lambda events, draft: True,
            policy=HistoryPolicy(
                projection="compact", context_ratio=0.8, recent_groups=1, segment_groups=9
            ),
        )
        part = history_part(
            "history",
            history,
            load_groups=lambda ctx: tuple(
                HistoryGroup((thaw(e),)) for e in ctx.dependencies["raw"].value
            ),
            dependencies=["raw"],
            context_policy=lambda: context.compiler.policy,
            reserved_tokens=lambda: extra[0],
        )
        async with ContextManager(
            [raw.part, part],
            store=store,
            principal=host,
            mandatory=["history"],
            compiler_policy=CompilerPolicy(max_tokens=30000, output_reserve=1000),
        ) as context:
            context.publish_catalog(host, "chat")
            before = await context.next_context("chat")
            assert not calls
            extra[0] = 20000
            smaller = await context.next_context("chat")
            assert len(calls) == 1
            assert before.part("history").revision != smaller.part("history").revision
            await context.next_context("chat")
            assert len(calls) == 1
            # With unchanged sources, a different total context still invalidates the bundle cache.
            context.compiler.policy = replace(context.compiler.policy, max_tokens=60000)
            larger = await context.next_context("chat")
            assert smaller.part("history").revision != larger.part("history").revision


async def test_history_part_uses_context_token_estimator_for_percentage_budget(tmp_path):
    host = Principal("host", True)
    raw = ConfigSource(
        "raw", [{"id": i, "content": "x" * 1000} for i in range(10)], kind="history", expose=False
    )
    policy = CompilerPolicy(
        max_tokens=1000, output_reserve=100, estimator=lambda text: len(text) // 100 + 1
    )
    with ContextStore(tmp_path) as store:
        store.create_scope(host, "chat")
        # No compactor: the context estimator says this fits, even though the default byte
        # estimator would greatly exceed the budget. Projection and compiler must agree.
        history = HistoryManager(store, host, policy=HistoryPolicy(recent_groups=10))
        part = history_part(
            "history",
            history,
            dependencies=["raw"],
            context_policy=policy,
            load_groups=lambda ctx: tuple(
                HistoryGroup((thaw(e),)) for e in ctx.dependencies["raw"].value
            ),
        )
        async with ContextManager(
            [raw.part, part],
            store=store,
            principal=host,
            mandatory=["history"],
            compiler_policy=policy,
        ) as context:
            context.publish_catalog(host, "chat")
            snapshot = await context.next_context("chat")
            assert len(snapshot.part("history").value["recent_events"]) == 10
            assert context.compile(snapshot).token_estimate < 800
