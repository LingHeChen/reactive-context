"""DX acceptance and adversarial lifecycle coverage; no model API calls."""

import asyncio
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from reactive_context import (
    AmbiguousPartDeclaration,
    CompilerPolicy,
    ComputedSideEffectError,
    Context,
    ContextManager,
    ContextStore,
    InvalidStateDefault,
    LegacyPartAdapter,
    MissingPartProvider,
    PartHandle,
    PartMaterialization,
    PartSpec,
    Principal,
    ScopeBindingRequired,
    UnknownContextOption,
    computed,
    part,
    skill,
    state,
)
from reactive_context.errors import (
    CircularDependency,
    CompilerBudget,
    ConsistencyUnavailable,
    SchemaInvalid,
    SourceUnavailable,
)
from reactive_context.sources.config import ConfigSource


class Names(Context):
    first: str = state("Lin")
    last: str = state("Chen")
    items: list[str] = state(default_factory=list)

    @computed
    def full(self):
        return f"{self.first} {self.last}"


def test_discovery_isolation_validation_and_atomic_update():
    a, b = Names(), Names(first="Alex")
    assert a.full == "Lin Chen" and b.full == "Alex Chen"
    a.items.append("local")
    assert b.items == []
    before = a.state_revision
    with pytest.raises(SchemaInvalid):
        a.update(first="not committed", last={})
    assert a.first == "Lin" and a.state_revision == before
    a.update(first="Alex", expected_revision=before)
    assert a.full == "Alex Chen"
    with pytest.raises(ConsistencyUnavailable):
        a.update(first="stale", expected_revision=before)
    version = a.state_revision
    a.first = "Alex"
    assert a.state_revision == version
    with pytest.raises(UnknownContextOption):
        Names(frist="typo")
    with pytest.raises(UnknownContextOption):
        Names(full="no override")
    with pytest.raises(InvalidStateDefault):
        state({})
    with pytest.raises(InvalidStateDefault):
        state(default_factory=list, default=())
    with pytest.raises(FrozenInstanceError):
        Names.first.required = False
    with pytest.raises(TypeError):
        Names.first = state("changed")
    with pytest.raises(TypeError):
        del Names.first


def test_dynamic_edges_lazy_invalidation_and_failed_evaluation():
    class Switch(Context):
        choose: bool = state(True)
        left: int = state(1)
        right: int = state(2)

        @computed
        def selected(self):
            return self.left if self.choose else self.right

        @computed
        def final(self):
            return self.selected * 2

    ctx = Switch()
    assert ctx.explain("final")["status"] == "not_resolved"
    assert ctx.final == 2
    ctx.left = 3
    assert ctx.explain("final")["status"] == "dirty"
    assert ctx.final == 6
    ctx.choose = False
    assert ctx.final == 4
    count = ctx.explain("final")["computations"]
    ctx.left = 9
    assert ctx.explain("final")["status"] == "clean"
    assert ctx.final == 4 and ctx.explain("final")["computations"] == count
    assert set(ctx.explain("selected")["dependencies"]) == {"choose", "right"}


def test_cycles_side_effect_and_async_computed_rejected():
    class Cycle(Context):
        @computed
        def a(self):
            return self.b

        @computed
        def b(self):
            return self.a

    with pytest.raises(CircularDependency, match="a -> b -> a"):
        _ = Cycle().a

    class SideEffect(Context):
        count: int = state(0)

        @computed
        def wrong(self):
            self.count = 2
            return self.count

    ctx = SideEffect()
    with pytest.raises(ComputedSideEffectError):
        _ = ctx.wrong
    assert ctx.count == 0
    with pytest.raises(TypeError, match="synchronous"):

        class Invalid(Context):
            @computed
            async def wrong(self):
                return 1


def test_inheritance_shadow_conflict_and_duplicate_key():
    class Child(Names):
        first = state("Alex")
        full = "ordinary shadow"

    assert Child().first == "Alex"
    assert "full" not in Child.describe()
    assert Names().full == "Lin Chen"

    class Left(Context):
        field = state(1)

    class Right(Context):
        field = state(2)

    with pytest.raises(AmbiguousPartDeclaration):

        class Conflict(Left, Right):
            pass

    class Explicit(Left, Right):
        field = state(3)

    assert Explicit().field == 3
    with pytest.raises(AmbiguousPartDeclaration):

        class Duplicate(Context):
            one = state(1, key="same")
            two = state(2, key="same")


async def test_async_handle_no_hidden_io_and_optional_failure():
    calls = []

    class Remote(Context):
        task: str = state("hello")

        @part
        async def answer(self, reader):
            calls.append(reader.read("task"))
            return {"answer": reader.read("task")}

        @part(required=False)
        async def broken(self, reader):
            raise OSError("private information must not enter diagnostics")

    ctx = Remote()
    assert isinstance(ctx.answer, PartHandle)
    assert ctx.describe()["answer"]["dynamic_dependencies"] == "not_resolved"
    assert ctx.explain("answer")["status"] == "not_resolved"
    await ctx.flush()
    assert not calls
    assert await ctx.parts.answer.resolve() == {"answer": "hello"}
    compiled = await ctx.compile()
    assert compiled.messages and any(d["part_key"] == "broken" for d in compiled.diagnostics)
    assert "private information" not in repr(compiled)
    with pytest.raises(TypeError):
        compiled.manifest["snapshot_id"] = "bad"
    with pytest.raises(FrozenInstanceError):
        compiled.messages[0].content = "bad"
    old_content = repr(compiled.messages)
    ctx.task = "new"
    assert old_content == repr(compiled.messages)
    assert "new" in repr((await ctx.compile()).messages)
    await ctx.aclose()
    with pytest.raises(RuntimeError):
        await ctx.resolve("answer")


async def test_required_failure_and_budget_fail_closed():
    class Required(Context):
        @part
        async def broken(self, reader):
            raise OSError("unavailable")

    with pytest.raises(SourceUnavailable):
        await Required().compile()
    with pytest.raises(CompilerBudget):
        await Names(compiler_policy=CompilerPolicy(max_tokens=32, output_reserve=1)).compile()


async def test_async_stale_result_cannot_commit_and_task_reads_are_isolated():
    started, release = asyncio.Event(), asyncio.Event()
    reads = []

    class Remote(Context):
        value: int = state(1)

        @part
        async def answer(self, reader):
            value = reader.read("value")
            reads.append(value)
            started.set()
            await release.wait()
            return value

    a, b = Remote(), Remote(value=20)
    task = asyncio.create_task(a.compile())
    await started.wait()
    for value in range(2, 10):
        a.value = value
    release.set()
    result = await task
    assert "9" in next(m.content for m in result.messages if m.part_id == "answer")
    assert a._dx.nodes["answer"].value == 9 and reads[0] == 1
    assert a._dx.nodes["answer"].value != b._dx.nodes["value"].value
    assert (await b.resolve("answer")) == 20
    assert set(b.explain("answer")["dependencies"]) == {"value"}


async def test_singleflight_cancel_and_close():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Remote(Context):
        @part(revision=lambda reader: "fixed")
        async def answer(self, reader):
            calls.append(1)
            entered.set()
            await release.wait()
            return "done"

    ctx = Remote()
    a = asyncio.create_task(ctx.resolve("answer"))
    await entered.wait()
    b = asyncio.create_task(ctx.resolve("answer"))
    release.set()
    assert await a == await b == "done"
    assert len(calls) == 1
    other = Remote()
    release.clear()
    c = asyncio.create_task(other.resolve("answer"))
    await asyncio.sleep(0)
    await other.aclose()
    assert c.cancelled()
    assert other.explain("answer")["status"] == "not_resolved"


async def test_factory_scopes_revision_polling_fixed_hash_and_strict_consistency():
    class Provider:
        def __init__(self):
            self.version = "1"
            self.value = "rule A"
            self.calls = 0

        async def revision(self, reader, resource, **options):
            assert reader.scope_bindings["project"] == "demo"
            return self.version

        async def resolve(self, reader, resource, **options):
            self.calls += 1
            return PartMaterialization(self.value, sources=("file:" + resource,))

    class Project(Context):
        coding = skill("coding", scope="project")

    provider = Provider()
    with pytest.raises(ScopeBindingRequired):
        Project()
    with pytest.raises(MissingPartProvider):
        await Project(scope_bindings={"project": "demo"}).compile()
    ctx = Project(scope_bindings={"project": "demo"}, providers={"skill": provider})
    a, b = await ctx.compile(), await ctx.compile()
    assert a.snapshot_id == b.snapshot_id
    assert a.manifest["rendered_messages_hash"] == b.manifest["rendered_messages_hash"]
    assert provider.calls == 1
    provider.version, provider.value = "2", "rule B"
    c = await ctx.compile(consistency="strict_fresh")
    assert c.snapshot_id != a.snapshot_id and "rule B" in repr(c.messages)
    assert all(message.role == "user" for message in c.messages)
    assert c.manifest["declarations"]["parts"]["coding"]["sources"] == ("file:coding",)
    with pytest.raises(ValueError):
        PartMaterialization("inject", trust="trusted_policy")

    class BestEffort(Context):
        @part
        async def clock(self, reader):
            return "now"

    with pytest.raises(ConsistencyUnavailable):
        await BestEffort().compile(consistency="strict_fresh")


async def test_pinned_keeps_original_view_until_repin():
    ctx = Names()
    a = await ctx.compile(consistency="pinned")
    ctx.first = "Alex"
    b = await ctx.compile(consistency="pinned")
    assert b.snapshot_id == a.snapshot_id
    ctx.repin()
    c = await ctx.compile(consistency="pinned")
    assert c.snapshot_id != a.snapshot_id


async def test_legacy_and_protocol_use_same_compiler():
    source = ConfigSource("legacy", "old")

    class Plugin:
        spec = PartSpec("plugin", schema=str)

        def dependencies(self):
            return ("legacy",)

        async def revision(self, reader):
            return reader.dependencies["legacy"].revision

        async def resolve(self, reader):
            return "plugin:" + reader.dependencies["legacy"].value

        async def materialize(self, reader, value):
            return PartMaterialization(value, sources=("plugin:source",))

    ctx = Context([LegacyPartAdapter(source.part), Plugin()])
    first = await ctx.compile()
    assert "plugin:old" in repr(first.messages)
    source.set("new")
    second = await ctx.compile()
    assert "plugin:new" in repr(second.messages)


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=40))
def test_property_chain_matches_eager_value(updates):
    class Sum(Context):
        value: int = state(0)

        @computed
        def doubled(self):
            return self.value * 2

        @computed
        def final(self):
            return self.doubled + 1

    ctx = Sum()
    for value in updates:
        ctx.value = value
        assert ctx.final == value * 2 + 1


def test_hundred_node_dag_avoids_unrelated_recomputation():
    declarations = {"root": state(1), "other": state(4)}
    for index in range(100):

        def compute(self, index=index):
            return self.root + index

        declarations[f"node_{index}"] = computed(compute)
    declarations["unrelated"] = computed(lambda self: self.other * 3)
    cls = type("Large", (Context,), declarations)
    ctx = cls()
    assert [getattr(ctx, f"node_{i}") for i in range(100)] == list(range(1, 101))
    assert ctx.unrelated == 12
    ctx.root = 9
    assert all(ctx.explain(f"node_{i}")["computations"] == 1 for i in range(100))
    assert ctx.unrelated == 12 and ctx.explain("unrelated")["computations"] == 1
    assert [getattr(ctx, f"node_{i}") for i in range(100)] == list(range(9, 109))


async def test_same_version_concurrent_local_cas():
    ctx = Names()
    version = ctx.state_revision

    async def write(value):
        await asyncio.sleep(0)
        return ctx.update(first=value, expected_revision=version)

    results = await asyncio.gather(write("A"), write("B"), return_exceptions=True)
    assert sum(isinstance(v, ConsistencyUnavailable) for v in results) == 1


async def test_native_agent_with_declared_manager_preserves_history_and_tool_chain(tmp_path):
    from copy import deepcopy

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from reactive_context.adapters.pydantic_ai import ReactiveContextCapability, validate_protocol

    class Chat(ContextManager):
        goal: str = state("goal A")

        @computed
        def instruction(self):
            return "current " + self.goal

    host, actor = Principal("host", admin=True), Principal("actor")
    store = ContextStore(tmp_path)
    store.create_scope(host, "chat")
    store.grant_scope(host, "chat", actor.id)
    seen = []
    async with Chat(store=store, principal=actor) as ctx:

        def respond(messages, info):
            seen.append(deepcopy(messages))
            if len(seen) == 1:
                assert "goal A" in repr(messages)
                return ModelResponse(parts=[ToolCallPart("advance", {}, tool_call_id="call-1")])
            assert "goal B" in repr(messages)
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[ReactiveContextCapability(ctx, scope="chat")],
        )

        @agent.tool_plain
        def advance():
            ctx.goal = "goal B"
            return "updated"

        result = await agent.run("work")
        assert result.output == "done"
        validate_protocol(result.all_messages())
        original = deepcopy(result.all_messages())
        await ctx.compile(scope="chat")
        assert result.all_messages() == original
        assert store.db.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] > 0
    store.close()


async def test_file_factory_host_grant_and_scope_independent_instances(tmp_path):
    from reactive_context import FilePartProvider

    (tmp_path / "coding.md").write_text("Use tests", encoding="utf-8")
    provider = FilePartProvider(tmp_path)

    class Project(Context):
        coding = skill("coding.md", scope="project")

    policy = CompilerPolicy(
        grants={"coding": "system"},
        part_order=("coding",),
        prefix_parts=frozenset({"coding"}),
    )
    ctx = Project(
        scope_bindings={"project": "A"},
        providers={"skill": provider},
        compiler_policy=policy,
    )
    first = await ctx.compile()
    assert first.messages[0].role == "system" and first.messages[0].placement == "instructions"
    (tmp_path / "coding.md").write_text("Updated tests", encoding="utf-8")
    second = await ctx.compile()
    assert first.snapshot_id != second.snapshot_id
    assert "Updated tests" in second.messages[0].content
    assert "Updated tests" not in first.messages[0].content
    with pytest.raises(Exception) as exc:
        await provider.resolve(None, "../outside.md")
    from reactive_context.errors import AccessRevoked

    assert isinstance(exc.value, AccessRevoked)


async def test_nested_async_dependency_revalidates_and_readonly_containers():
    version = [1]

    class Chain(Context):
        @part(revision=lambda reader: str(version[0]))
        async def child(self, reader):
            return {"value": version[0]}

        @part(revision=lambda reader: "algorithm-v1")
        async def parent(self, reader):
            return {"parent": (await reader.resolve("child"))["value"]}

    ctx = Chain()
    assert (await ctx.resolve("parent"))["parent"] == 1
    with pytest.raises(TypeError):
        (await ctx.resolve("child"))["value"] = 99
    version[0] = 2
    assert (await ctx.resolve("parent"))["parent"] == 2
    assert (await ctx.compile()).manifest["declarations"]["parts"]["parent"]["dependencies"]


async def test_async_cycle_timeout_and_optional_missing_factory():
    class Cycle(Context):
        @part
        async def a(self, reader):
            return await reader.resolve("b")

        @part
        async def b(self, reader):
            return await reader.resolve("a")

    with pytest.raises(CircularDependency):
        await Cycle().resolve("a")

    class Optional(Context):
        absent = skill("missing-plugin", required=False)

    result = await Optional().compile()
    assert not result.messages
    assert result.diagnostics[0]["code"] == "MISSING_PART_PROVIDER"

    class Timeout(Context):
        @part(timeout=0.01, required=False)
        async def slow(self, reader):
            await asyncio.Event().wait()

    result = await Timeout().compile()
    assert result.diagnostics


async def test_final_seal_detects_lost_external_event_and_no_mixed_snapshot():
    external = [1]
    entered, release = asyncio.Event(), asyncio.Event()

    class Versioned(Context):
        @part(revision=lambda reader: str(external[0]))
        async def early(self, reader):
            return external[0]

        @part
        async def later(self, reader):
            entered.set()
            await release.wait()
            return "later"

    ctx = Versioned()
    task = asyncio.create_task(ctx.compile())
    await entered.wait()
    external[0] = 2
    release.set()
    result = await task
    early = next(message for message in result.messages if message.part_id == "early")
    assert '"data": "2"' in early.content


async def test_optional_dependency_cannot_remove_required_part():
    from reactive_context import computed_part, source_part

    async def broken(ctx):
        raise OSError("missing")

    optional = source_part(key="optional", schema=str, resolve=broken, required=False)
    dependent = computed_part(
        key="required",
        schema=str,
        dependencies=["optional"],
        compute=lambda ctx: "hidden",
    )
    with pytest.raises(SourceUnavailable, match="required"):
        await Context([optional, dependent]).compile()


def test_diamond_shadows_and_class_registry_cannot_be_overwritten():
    class Base(Context):
        value = state(1)

    class Left(Base):
        pass

    class Right(Base):
        pass

    class Diamond(Left, Right):
        pass

    assert Diamond().value == 1
    with pytest.raises(TypeError):
        Diamond.__parts__ = {}

    class Shadow(Base):
        value = "ordinary"

    class Child(Shadow):
        pass

    assert "value" not in Child.describe()


async def test_extend_rebinds_local_nodes_instead_of_sharing_descriptor_closures():
    original = Names(first="A")
    extended = original.extend(parts=[ConfigSource("extra", "extra data").part])
    assert extended.full == "A Chen"
    extended.first = "B"
    assert original.full == "A Chen" and extended.full == "B Chen"
    assert "B Chen" in repr((await extended.compile()).messages)
    await original.aclose()
    assert extended.full == "B Chen"
    await extended.aclose()


async def test_nested_external_revision_change_during_parent_load_is_discarded():
    entered, release = asyncio.Event(), asyncio.Event()
    external = [1]

    class Chain(Context):
        @part(revision=lambda reader: str(external[0]))
        async def child(self, reader):
            return external[0]

        @part(revision=lambda reader: "fixed")
        async def parent(self, reader):
            value = await reader.resolve("child")
            entered.set()
            await release.wait()
            return value * 2

    ctx = Chain()
    task = asyncio.create_task(ctx.resolve("parent"))
    await entered.wait()
    external[0] = 2
    release.set()
    assert await task == 4


async def test_strict_refuses_provider_that_has_revision_method_but_returns_none():
    class NoVersion:
        async def revision(self, reader, resource, **options):
            return None

        async def resolve(self, reader, resource, **options):
            return "unknown version"

    class Project(Context):
        coding = skill("coding")

    with pytest.raises(ConsistencyUnavailable):
        await Project(providers={"skill": NoVersion()}).compile(consistency="strict_fresh")


async def test_protocol_acl_and_declaration_schema_remain_enforced():
    from reactive_context.errors import AccessRevoked

    class Forbidden:
        spec = PartSpec("private", schema=str)

        def dependencies(self):
            return ()

        async def acl_check(self, reader):
            return False

        async def resolve(self, reader):
            pytest.fail("ACL must run before source resolution")

        async def materialize(self, reader, value):
            return PartMaterialization(value)

    with pytest.raises(AccessRevoked):
        await Context([Forbidden()]).compile()

    class Invalid(Context):
        @part(schema=int)
        async def source(self, reader):
            return {"bad": "shape"}

    with pytest.raises(SchemaInvalid):
        await Invalid().compile()


async def test_close_cancels_boundary_while_provider_revision_is_pending():
    entered = asyncio.Event()

    async def pending_revision(reader):
        entered.set()
        await asyncio.Event().wait()

    class Pending(Context):
        @part(revision=pending_revision)
        async def value(self, reader):
            pytest.fail("revision is still pending")

    ctx = Pending()
    task = asyncio.create_task(ctx.compile())
    await entered.wait()
    await asyncio.wait_for(ctx.aclose(), timeout=1)
    assert task.cancelled()


async def test_options_and_legacy_dependencies_share_core_graph_and_versions():
    legacy = ConfigSource("legacy", "first")

    class Mixed(Context):
        @part(deps=("legacy",), revision=lambda reader: "derived-v1")
        async def composed(self, reader):
            return "combined:" + reader.read("legacy")

    ctx = Mixed([legacy.part])
    assert await ctx.resolve("composed") == "combined:first"
    legacy.set("second")
    compiled = await ctx.compile()
    assert "combined:second" in repr(compiled.messages)
    assert "legacy" in compiled.manifest["declarations"]["parts"]["composed"]["dependencies"]


def test_opaque_mutable_defaults_require_factory_and_nested_defaults_are_checked():
    class Mutable:
        def __init__(self):
            self.items = []

    with pytest.raises(InvalidStateDefault):
        state(Mutable())
    with pytest.raises(InvalidStateDefault):
        state(("immutable outer", []))

    class Isolated(Context):
        value = state(default_factory=Mutable)

    a, b = Isolated(), Isolated()
    a.value.items.append("A")
    assert b.value.items == []


def test_scope_resource_identity_cannot_collide_through_separator_characters():
    class A(Context):
        value = state(1, key="c", scope="project")

    class B(Context):
        value = state(1, key="b:c", scope="project")

    a = A(scope_bindings={"project": "a:b"})
    b = B(scope_bindings={"project": "a"})
    assert a.explain("value")["resource_id"] != b.explain("value")["resource_id"]


async def test_strict_request_policy_does_not_leak_to_an_independent_task():
    entered, release = asyncio.Event(), asyncio.Event()

    async def version(reader):
        entered.set()
        await release.wait()
        return "1"

    class Concurrent(Context):
        @part(revision=version)
        async def versioned(self, reader):
            return "fixed"

        @part
        async def best_effort(self, reader):
            return "live"

    ctx = Concurrent()
    strict = asyncio.create_task(ctx.next_context("strict", "strict_fresh", selected=["versioned"]))
    await entered.wait()
    assert await ctx.resolve("best_effort") == "live"
    release.set()
    assert (await strict).part("versioned").value == "fixed"


def test_declarations_cannot_shadow_runtime_services_but_context_history_factory_is_valid():
    from reactive_context import memory

    with pytest.raises(AmbiguousPartDeclaration):

        class Invalid(Context):
            compiler = state("shadow")

    with pytest.raises(AmbiguousPartDeclaration):

        class InvalidManager(ContextManager):
            history = state("would overwrite the archive manager")

    class Valid(Context):
        history = memory("project.history", required=False)

    assert "history" in Valid.describe()
