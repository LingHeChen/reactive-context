import json
from dataclasses import replace

import pytest

from reactive_context import CompilerPolicy, Context, ContextIRSegment, SkillRegistry, skill_part
from reactive_context.core.snapshot import thaw
from reactive_context.errors import AccessRevoked, AuthorityViolation
from reactive_context.sources.catalog import ToolCatalogSource
from reactive_context.sources.config import ConfigSource


def optional(key, priority):
    def project(value, ctx):
        return [ContextIRSegment(key, key, ctx.revision, value, required=False, priority=priority)]

    return ConfigSource(key, "value", project=project, required=False).part


async def test_budget_selection_is_independent_from_render_order():
    policy = CompilerPolicy(part_order=("b", "a"), max_tokens=10000, output_reserve=0)
    async with Context([optional("a", 100), optional("b", 1)], compiler_policy=policy) as context:
        snapshot = await context.next_context()
        first = context.compile(snapshot)
        reversed_priorities = replace(
            snapshot,
            segments=tuple(
                replace(segment, priority=-segment.priority) for segment in snapshot.segments
            ),
        )
        second = context.compile(reversed_priorities)
        assert [message.part_id for message in first.provider_messages] == ["b", "a"]
        assert first.provider_messages == second.provider_messages
        assert first.render_order == ("b", "a")
        assert {item.part_id: item.render_index for item in first.trace} == {"a": 1, "b": 0}
        # Admit just one item: priority still controls which one fits, not prompt order.
        constrained = replace(policy, max_tokens=first.trace[0].token_estimate)
        compiled = type(context.compiler)(constrained).compile(snapshot)
        assert [message.part_id for message in compiled.provider_messages] == ["a"]


async def test_trusted_skill_requires_separate_host_grant_and_prefix_data_is_not_promoted():
    source = ConfigSource("skill", "Do the approved workflow.")
    wrapped = skill_part(source.part)
    async with Context([wrapped]) as context:
        with pytest.raises(AuthorityViolation):
            context.compile(await context.next_context())
    data = ConfigSource("schema", {"table": "accounts"})
    policy = CompilerPolicy(grants={"skill": "developer"}, prefix_parts=frozenset({"schema"}))
    async with Context([wrapped, data.part], compiler_policy=policy) as context:
        compiled = context.compile(await context.next_context())
        instruction, reference = compiled.provider_messages
        assert instruction.role == "developer" and instruction.placement == "instructions"
        assert reference.role == "user" and reference.placement == "prefix"
        assert json.loads(instruction.content)["instructions"] == "Do the approved workflow."


async def test_catalog_order_and_schema_key_order_are_canonical():
    source = ToolCatalogSource(
        "tools",
        [
            {"name": "z", "parameters_json_schema": {"type": "object", "properties": {}}},
            {"name": "a", "parameters_json_schema": {"properties": {}, "type": "object"}},
        ],
    )
    async with Context([source.part]) as context:
        first = context.compile(await context.next_context()).provider_tools
        source.set(
            [
                {"parameters_json_schema": {"type": "object", "properties": {}}, "name": "a"},
                {"parameters_json_schema": {"properties": {}, "type": "object"}, "name": "z"},
            ]
        )
        second = context.compile(await context.next_context()).provider_tools
        assert first == second
        assert [tool["name"] for tool in thaw(first[0])] == ["a", "z"]


async def test_skills_resolve_only_on_activation_and_keep_first_activation_order():
    counts = {"a": 0, "b": 0}
    sources = {key: ConfigSource(key, f"{key}-body-v1") for key in counts}
    async with Context() as context:
        skills = SkillRegistry(context)
        for key, source in sources.items():

            async def resolve(ctx, key=key, source=source):
                counts[key] += 1
                return await source.resolve(ctx)

            skills.register(
                replace(source.part, resolve=resolve),
                name=key,
                description=f"Use {key}",
                trusted=True,
            )
        metadata = await context.next_context()
        assert len(metadata.parts) == 1
        assert counts == {"a": 0, "b": 0}
        skills.activate("b")
        first = context.compile(await context.next_context())
        assert counts == {"a": 0, "b": 1}
        skills.activate("a")
        second = context.compile(await context.next_context())
        assert skills.active_order == ("b", "a")
        assert first.provider_messages == second.provider_messages[:2]
        assert counts == {"a": 1, "b": 1}
        sources["a"].set("a-body-v2")
        third = context.compile(await context.next_context())
        assert second.provider_messages[:2] == third.provider_messages[:2]
        assert "a-body-v1" not in third.provider_messages[-1].content
        assert counts == {"a": 2, "b": 1}


async def test_deactivate_reactivate_preserves_slots_and_cannot_override_revoke():
    async with Context() as context:
        registry = SkillRegistry(context)
        for key in ["b", "a"]:
            registry.register(
                ConfigSource(key, key).part, name=key, description=key, trusted=True, active=True
            )
        await context.next_context()
        await registry.deactivate("b")
        assert [part.part_id for part in (await context.next_context()).parts] == [
            registry.index_key,
            "a",
        ]
        registry.activate("b")
        assert registry.active_order == ("b", "a")
        compiled = context.compile(await context.next_context())
        assert [message.part_id for message in compiled.provider_messages] == [
            registry.index_key,
            "b",
            "a",
        ]
        context.revoke("b")
        await registry.deactivate("b")
        registry.activate("b")
        with pytest.raises(AccessRevoked):
            await context.next_context()


async def test_registry_rejects_untrusted_authority_and_repin_is_explicit():
    async with Context() as context:
        skills = SkillRegistry(context)
        source = ConfigSource("skill", "body")
        with pytest.raises(AuthorityViolation):
            skills.register(source.part, name="skill", description="skill", trusted=False)
        await context.next_context(consistency="pinned")
        skills.register(source.part, name="skill", description="skill", trusted=True, active=True)
        with pytest.raises(ValueError, match="repin"):
            await context.next_context(consistency="pinned")
        context.repin()
        assert (await context.next_context(consistency="pinned")).part("skill").value == "body"


def test_layout_configuration_is_immutable_and_part_of_identity():
    order = ["a", "b"]
    policy = CompilerPolicy(part_order=order, prefix_parts={"a"})
    order.reverse()
    assert policy.part_order == ("a", "b")
    assert policy.fingerprint != replace(policy, part_order=("b", "a")).fingerprint
    assert policy.fingerprint != replace(policy, prefix_parts=frozenset()).fingerprint
    with pytest.raises(ValueError, match="duplicate"):
        CompilerPolicy(part_order=("a", "a"))


async def test_host_can_fix_skill_slots_across_different_registration_and_activation_orders():
    async def build(registration, activation):
        async with Context() as context:
            skills = SkillRegistry(context, order=("a", "b"))
            for key in registration:
                skills.register(
                    ConfigSource(key, key + "-BODY").part,
                    name=key,
                    description="Use " + key,
                    trusted=True,
                )
            for key in activation:
                skills.activate(key)
            assert skills.active_order == ("a", "b")
            return [
                message.content
                for message in context.compile(await context.next_context()).provider_messages
            ]

    assert await build(["a", "b"], ["b", "a"]) == await build(["b", "a"], ["a", "b"])


async def test_reserved_skill_slots_allow_incremental_registration_and_activation():
    async with Context() as context:
        skills = SkillRegistry(context, order=("a", "b"))
        for key in ["b", "a"]:
            skills.register(
                ConfigSource(key, key).part, name=key, description=key, trusted=True, active=True
            )
        assert skills.active_order == ("a", "b")
        assert context.compile(await context.next_context()).render_order == (
            skills.index_key,
            "a",
            "b",
        )
