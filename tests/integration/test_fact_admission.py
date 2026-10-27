"""Generic admission is independent of the travel benchmark and any gold labels."""

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    FactVerdict,
    Principal,
)
from reactive_context.adapters.operations import context_tools
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
from reactive_context.fact_checks import FactAdmissionRejected
from reactive_context.store import VersionConflict

HOST = Principal("host", True)
USER = Principal("agent")


class Preference(BaseModel):
    attribute: str
    value: str
    confirmed: bool


def provision(path, check=None):
    store = ContextStore(path)
    store.create_scope(HOST, "session")
    store.grant_scope(HOST, "session", USER.id)
    store.register_schema("preference", Preference, check=check)
    return store


async def test_source_checks_and_necessity_before_creation_then_changed_only_update(tmp_path):
    observed = {}

    def check(req):
        if not req.value["confirmed"]:
            return FactVerdict("skip", ("unconfirmed_inference",))
        if not any(observed.get(e) == req.value["value"] for e in req.evidence):
            return FactVerdict("reject", ("unsupported_by_source",))
        return FactVerdict("accept", ("durable_preference",))

    with provision(tmp_path, check) as store:
        async with ContextManager(store=store, principal=USER) as manager:
            e = store.observe_evidence(USER, "session", "user-input", "1")
            observed[e] = "zh-CN"
            draft = {"attribute": "language", "value": "zh-CN", "confirmed": False}
            result = await manager.save_fact("session", "preference", "user.language", draft, [e])
            assert result["decision"] == "skip"
            draft["confirmed"] = True
            draft["value"] = "unsupported"
            result = await manager.save_fact("session", "preference", "user.language", draft, [e])
            assert result["decision"] == "reject"
            assert store.db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == 0
            assert store.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0
            draft["value"] = "zh-CN"
            result = await manager.save_fact("session", "preference", "user.language", draft, [e])
            assert result["decision"] == "created" and result["revision"] == 1
            assert result["validation"] == "host_checks_passed"
            assert result["verification"] == "unverified"
            again = await manager.save_fact("session", "preference", "user.language", draft, [e])
            assert again["decision"] == "unchanged" and again["revision"] == 1
            assert not store.active_refs(USER, "session")
            e2 = store.observe_evidence(USER, "session", "user-input", "2")
            observed[e2] = "en-US"
            draft["value"] = "en-US"
            missing = await manager.save_fact("session", "preference", "user.language", draft, [e2])
            assert missing["decision"] == "revision_required"
            with pytest.raises(VersionConflict):
                await manager.save_fact(
                    "session", "preference", "user.language", draft, [e2], expected_revision=2
                )
            updated = await manager.save_fact(
                "session", "preference", "user.language", draft, [e2], expected_revision=1
            )
            assert updated["decision"] == "updated" and updated["revision"] == 2
            assert updated["part_id"] == result["part_id"]
            assert store.db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == 1
            assert store.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 2
            assert store.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0


def test_structural_only_is_explicit_and_commit_rechecks_host_policy(tmp_path):
    with provision(tmp_path) as store:
        e = store.observe_evidence(USER, "session", "source", "1")
        value = {"attribute": "language", "value": "zh-CN", "confirmed": True}
        result = store.check_fact(USER, "session", "preference", "language", value, [e])
        assert result.validation == "structural_only"
        assert store.query(USER, "session") == []
        gate = {"accept": True}
        store.register_schema(
            "preference",
            Preference,
            check=lambda req: (
                FactVerdict() if gate["accept"] else FactVerdict("reject", ("source_changed",))
            ),
        )
        p = store.propose_fact(USER, "session", "preference", "language", value, [e])
        gate["accept"] = False
        with pytest.raises(FactAdmissionRejected):
            store.commit(USER, p)
        assert store.query(USER, "session") == []
        assert store.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0


def test_low_level_identical_commit_does_not_create_revision_and_schema_reopen(tmp_path):
    with provision(tmp_path) as store:
        e = store.observe_evidence(USER, "session", "source", "1")
        value = {"attribute": "language", "value": "zh-CN", "confirmed": True}
        p = store.propose_fact(USER, "session", "preference", "language", value, [e])
        first = store.commit(USER, p)
        p = store.propose_fact(
            USER, "session", "preference", "language", value, [e], expected_revision=1
        )
        again = store.commit(USER, p)
        assert first.id == again.id and again.revision == 1
        p = store.propose_fact(
            USER, "session", "preference", "language", value, [e], expected_revision=0
        )
        with pytest.raises(VersionConflict):
            store.commit(USER, p)
    with ContextStore(tmp_path) as store:
        store.register_schema("preference", Preference)
        assert store.load(USER, first.id).revision == 1
        assert (
            store.check_fact(USER, "session", "preference", "language", value, [e]).decision
            == "unchanged"
        )


async def test_native_save_tool_reports_skip_without_creating_proposal(tmp_path):
    with provision(tmp_path, lambda req: FactVerdict("skip", ("temporary_reasoning",))) as store:
        e = store.observe_evidence(USER, "session", "source", "1")
        step = 0
        async with ContextManager(
            store=store, principal=USER, compiler_policy=CompilerPolicy(max_tokens=128000)
        ) as manager:

            def reply(messages, info):
                nonlocal step
                step += 1
                if step == 1:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "context_save_fact",
                                {
                                    "schema_id": "preference",
                                    "key": "user.language",
                                    "value": {
                                        "attribute": "language",
                                        "value": "en-US",
                                        "confirmed": False,
                                    },
                                    "evidence": [e],
                                },
                                "save",
                            )
                        ]
                    )
                assert "temporary_reasoning" in str(messages)
                return ModelResponse(parts=[TextPart("not saved")])

            await Agent(
                FunctionModel(reply),
                toolsets=[context_tools(manager, "session")],
                capabilities=[ReactiveContextCapability(manager, scope="session", cache=None)],
            ).run("A temporary guess")
        assert store.query(USER, "session") == []
        assert store.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0


def test_invalid_or_async_host_checks_fail_closed(tmp_path):
    async def async_check(req):
        return FactVerdict()

    with provision(tmp_path) as store:
        with pytest.raises(ValueError):
            store.register_schema("async", Preference, check=async_check)
        store.register_schema("bad", Preference, check=lambda req: True)
        e = store.observe_evidence(USER, "session", "source", "1")
        with pytest.raises(TypeError):
            store.propose_fact(
                USER,
                "session",
                "bad",
                "one",
                {"attribute": "x", "value": "y", "confirmed": True},
                [e],
            )
        assert store.query(USER, "session") == []
    with pytest.raises(ValueError):
        FactVerdict("reject", ("private message with spaces",))


async def test_duplicate_detection_ignores_key_order_but_preserves_json_types(tmp_path):
    with provision(tmp_path) as store:
        store.register_schema("object", dict)
        e = store.observe_evidence(USER, "session", "source", "1")
        async with ContextManager(store=store, principal=USER) as manager:
            a = await manager.save_fact("session", "object", "settings", {"a": 1, "b": True}, [e])
            b = await manager.save_fact("session", "object", "settings", {"b": True, "a": 1}, [e])
            assert a["part_id"] == b["part_id"] and b["decision"] == "unchanged"
            c = await manager.save_fact(
                "session", "object", "settings", {"b": 1, "a": 1}, [e], expected_revision=1
            )
            assert c["decision"] == "updated" and c["revision"] == 2


async def test_concurrent_updates_still_require_matching_revision(tmp_path):
    import asyncio

    with provision(tmp_path) as store:
        e = store.observe_evidence(USER, "session", "source", "1")
        async with ContextManager(store=store, principal=USER) as manager:
            base = {"attribute": "language", "value": "zh-CN", "confirmed": True}
            await manager.save_fact("session", "preference", "language", base, [e])
            a = manager.save_fact(
                "session",
                "preference",
                "language",
                {**base, "value": "en-US"},
                [e],
                expected_revision=1,
            )
            b = manager.save_fact(
                "session",
                "preference",
                "language",
                {**base, "value": "fr-FR"},
                [e],
                expected_revision=1,
            )
            results = await asyncio.gather(a, b, return_exceptions=True)
            assert sum(isinstance(r, VersionConflict) for r in results) == 1
            assert store.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 2


def test_restart_does_not_silently_disable_required_host_check(tmp_path):
    value = {"attribute": "language", "value": "zh-CN", "confirmed": True}
    with provision(tmp_path, lambda req: FactVerdict()) as store:
        e = store.observe_evidence(USER, "session", "source", "1")
        p = store.propose_fact(USER, "session", "preference", "language", value, [e])
        record = store.commit(USER, p)
    with ContextStore(tmp_path) as store:
        from reactive_context.errors import SchemaInvalid

        store.register_schema("preference", Preference)  # reads remain supported
        assert store.load(USER, record.id).revision == 1
        with pytest.raises(SchemaInvalid, match="required host Fact checker"):
            store.propose_fact(USER, "session", "preference", "other", value, [e])
        store.register_schema(
            "preference", Preference, check=lambda req: FactVerdict("skip", ("not_durable",))
        )
        assert (
            store.check_fact(USER, "session", "preference", "other", value, [e]).decision == "skip"
        )


async def test_explicit_schema_context_is_not_a_stored_fact_and_omits_defaults(tmp_path):
    from pydantic import Field

    from reactive_context import fact_schema_part
    from reactive_context.core.snapshot import thaw

    class InternalModel(BaseModel):
        value: str
        internal_default: str = Field(
            default="HOST_DEFAULT_DO_NOT_PUBLISH", examples=["HOST_EXAMPLE"]
        )

    with provision(tmp_path, lambda req: FactVerdict()) as store:
        store.register_schema("internal", InternalModel)
        part = fact_schema_part(store, {"preference": 1})
        async with ContextManager(
            [part], mandatory=[part.key], store=store, principal=USER
        ) as manager:
            snapshot = await manager.next_context("session")
            compiled = manager.compile(snapshot)
            text = str(compiled.provider_messages)
            assert "confirmed" in text and "attribute" in text and "host_check_available" in text
            assert "internal_default" not in text
            assert store.query(USER, "session", kind="fact") == []
            assert store.db.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0
        selected = fact_schema_part(store, {"internal": 1})
        async with ContextManager(
            [selected], mandatory=[selected.key], store=store, principal=USER
        ) as manager:
            snapshot = await manager.next_context("session")
            payload = thaw(snapshot.part(selected.key).value)
            schema = payload["schemas"][0]["json_schema"]
            assert "internal_default" in schema["properties"]
            assert "HOST_DEFAULT_DO_NOT_PUBLISH" not in str(schema) and "HOST_EXAMPLE" not in str(
                schema
            )


async def test_schema_annotations_do_not_erase_properties_named_default_or_examples(tmp_path):
    from pydantic import create_model

    from reactive_context import fact_schema_part
    from reactive_context.core.snapshot import thaw

    Child = create_model("examples", default=(str, ...))
    Schema = create_model("config", default=(Child, ...), examples=(list[str], ...))
    with provision(tmp_path) as store:
        store.register_schema("config", Schema)
        part = fact_schema_part(store, {"config": 1})
        async with ContextManager(
            [part], mandatory=[part.key], store=store, principal=USER
        ) as manager:
            schema = thaw((await manager.next_context("session")).part(part.key).value)["schemas"][
                0
            ]["json_schema"]
            assert set(schema["properties"]) == {"default", "examples"}
            assert "examples" in schema["$defs"]
            assert "default" in schema["$defs"]["examples"]["properties"]


async def test_native_save_supports_explicit_schema_versions(tmp_path):
    with provision(tmp_path) as store:
        store.register_schema("preference", Preference, version=2)
        e = store.observe_evidence(USER, "session", "source", "1")
        step = 0
        async with ContextManager(
            store=store, principal=USER, compiler_policy=CompilerPolicy(max_tokens=128000)
        ) as manager:

            def reply(messages, info):
                nonlocal step
                step += 1
                if step == 1:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "context_save_fact",
                                {
                                    "schema_id": "preference",
                                    "schema_version": 2,
                                    "key": "user.language",
                                    "value": {
                                        "attribute": "language",
                                        "value": "zh-CN",
                                        "confirmed": True,
                                    },
                                    "evidence": [e],
                                },
                                "save-v2",
                            )
                        ]
                    )
                return ModelResponse(parts=[TextPart("done")])

            await Agent(
                FunctionModel(reply),
                toolsets=[context_tools(manager, "session")],
                capabilities=[ReactiveContextCapability(manager, scope="session", cache=None)],
            ).run("save")
        assert store.query(USER, "session")[0]["schema_version"] == 2
