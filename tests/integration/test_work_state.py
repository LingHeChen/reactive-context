"""Retention is enforced independently of natural-language inference and gold labels."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import (
    CompilerPolicy,
    ContextManager,
    ContextStore,
    Principal,
    StateUpdateRejected,
    WorkState,
    apply_state_operations,
    state_diff,
)
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
from reactive_context.errors import AccessRevoked, SchemaInvalid
from reactive_context.store import StoreCorruption, VersionConflict

HOST, ACTOR = Principal("host", True), Principal("agent")
SCOPE = "task"
BASE = {
    "goal": "release",
    "settings": {"language": "zh", "tests": "required"},
    "milestones": ["review", "ship"],
    "notes": "keep",
}


def provision(path, **options):
    store = ContextStore(path)
    store.create_scope(HOST, SCOPE)
    store.grant_scope(HOST, SCOPE, ACTOR.id)
    evidence = store.observe_evidence(ACTOR, SCOPE, "initial-input", "1")
    state = WorkState(store, ACTOR, **options)
    state.initialize(SCOPE, BASE, [evidence], host=HOST)
    return store, state, evidence


def permit(state, evidence, *grants, revision=1):
    return state.authorize(
        SCOPE,
        [{"path": p, "action": a} for p, a in grants],
        [evidence],
        expected_revision=revision,
        host=HOST,
    )


def test_patch_preserves_unmentioned_fields_and_full_replacement_cannot_drop_them(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/settings/language", "replace"))
        preview = state.replace(
            SCOPE,
            {"settings": {"language": "en"}},
            expected_revision=1,
            permit_id=token,
            preview=True,
        )
        assert preview["decision"] == "reject"
        assert "unpermitted_remove" in preview["reasons"]
        assert {c["path"] for c in preview["changes"] if c["action"] == "remove"} == {
            "/goal",
            "/settings/tests",
            "/milestones",
            "/notes",
        }
        with pytest.raises(StateUpdateRejected):
            state.replace(
                SCOPE, {"settings": {"language": "en"}}, expected_revision=1, permit_id=token
            )
        assert state.read(SCOPE)["value"] == BASE
        receipt = state.patch(
            SCOPE,
            [{"op": "set", "path": "/settings/language", "value": "en"}],
            expected_revision=1,
            permit_id=token,
        )
        assert receipt["decision"] == "updated" and receipt["revision"] == 2
        assert state.read(SCOPE)["value"] == {
            **BASE,
            "settings": {"language": "en", "tests": "required"},
        }
        assert store.db.execute("SELECT COUNT(*) FROM work_state_revisions").fetchone()[0] == 2


@pytest.mark.parametrize("empty", [None, "", [], {}])
def test_replacement_grant_cannot_clear_and_explicit_clear_can(tmp_path, empty):
    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/notes", "replace"))
        op = [{"op": "set", "path": "/notes", "value": empty}]
        with pytest.raises(StateUpdateRejected, match="state_update_rejected"):
            state.patch(SCOPE, op, expected_revision=1, permit_id=token)
        clear = permit(state, e, ("/notes", "clear"))
        state.patch(SCOPE, op, expected_revision=1, permit_id=clear)
        assert state.read(SCOPE)["value"]["notes"] == empty


def test_parent_set_and_parent_remove_require_child_removal_grants(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/settings", "replace"))
        for op in [
            {"op": "remove", "path": "/settings"},
            {"op": "set", "path": "/settings", "value": {"language": "zh"}},
        ]:
            result = state.patch(SCOPE, [op], expected_revision=1, permit_id=token, preview=True)
            assert result["decision"] == "reject"
        remove = permit(state, e, ("/settings/tests", "remove"))
        state.patch(
            SCOPE,
            [{"op": "set", "path": "/settings", "value": {"language": "zh"}}],
            expected_revision=1,
            permit_id=remove,
        )
        assert state.read(SCOPE)["value"]["settings"] == {"language": "zh"}


def test_empty_sentinel_is_domain_defined_not_hardcoded(tmp_path):
    store, state, e = provision(tmp_path, empty_values=(None, "", [], {}, "-"))
    with store:
        token = permit(state, e, ("/notes", "replace"))
        result = state.patch(
            SCOPE,
            [{"op": "set", "path": "/notes", "value": "-"}],
            expected_revision=1,
            permit_id=token,
            preview=True,
        )
        assert result["reasons"] == ["unpermitted_clear"]


def test_agent_cannot_issue_grants_and_arbitrary_observed_evidence_is_insufficient(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        with pytest.raises(AccessRevoked):
            state.authorize(
                SCOPE,
                [{"path": "/goal", "action": "replace"}],
                [e],
                expected_revision=1,
                host=ACTOR,
            )
        with pytest.raises(StateUpdateRejected):
            state.patch(
                SCOPE,
                [{"op": "set", "path": "/goal", "value": "other"}],
                expected_revision=1,
                permit_id=e,
            )
        store.create_scope(HOST, "other")
        other_e = store.observe_evidence(HOST, "other", "source", "1")
        with pytest.raises(AccessRevoked):
            permit(state, other_e, ("/goal", "replace"))


def test_permits_bind_scope_principal_revision_and_revocation(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/notes", "replace"))
        state.revoke_permit(SCOPE, token, host=HOST)
        op = [{"op": "set", "path": "/notes", "value": "changed"}]
        assert (
            state.patch(SCOPE, op, expected_revision=1, permit_id=token, preview=True)["decision"]
            == "reject"
        )
        token = permit(state, e, ("/notes", "replace"))
        other = Principal("other-agent")
        store.grant_scope(HOST, SCOPE, other.id)
        other_state = WorkState(store, other)
        assert (
            other_state.patch(SCOPE, op, expected_revision=1, permit_id=token, preview=True)[
                "decision"
            ]
            == "reject"
        )
        state.patch(SCOPE, op, expected_revision=1, permit_id=token)
        with pytest.raises(VersionConflict):
            state.patch(SCOPE, op, expected_revision=1, permit_id=token)
        assert not state.read(SCOPE)["permits"]
        assert (
            state.patch(
                SCOPE,
                [{"op": "set", "path": "/notes", "value": "newer"}],
                expected_revision=2,
                permit_id=token,
                preview=True,
            )["decision"]
            == "reject"
        )
        store.revoke_scope(HOST, SCOPE, ACTOR.id)
        with pytest.raises(AccessRevoked):
            state.read(SCOPE)
        with pytest.raises(AccessRevoked):
            state.patch(SCOPE, op, expected_revision=2, permit_id=token)


def test_noop_does_not_create_revision_and_restart_keeps_protection(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        assert state.replace(SCOPE, BASE, expected_revision=1)["decision"] == "unchanged"
        token = permit(state, e, ("/notes", "replace"))
    with ContextStore(tmp_path) as store:
        state = WorkState(store, ACTOR)
        assert state.read(SCOPE)["value"] == BASE
        with pytest.raises(StateUpdateRejected):
            state.replace(SCOPE, {}, expected_revision=1, permit_id=token)
        state.patch(
            SCOPE,
            [{"op": "set", "path": "/notes", "value": "done"}],
            expected_revision=1,
            permit_id=token,
        )
        with pytest.raises(ValueError, match="migration"):
            WorkState(store, ACTOR, empty_values=(None, "-"))


def test_atomic_lists_escaped_pointers_and_json_type_changes():
    base = {"a/b": {"~key": [1, 2]}, "flag": True}
    candidate = apply_state_operations(
        base, [{"op": "set", "path": "/a~1b/~0key", "value": [2, 1]}]
    )
    changes = state_diff(base, candidate)
    assert len(changes) == 1 and changes[0].path == "/a~1b/~0key"
    assert state_diff({"x": True}, {"x": 1})[0].action == "replace"
    assert base["a/b"]["~key"] == [1, 2]
    with pytest.raises(ValueError):
        apply_state_operations(base, [{"op": "set", "path": "/a~1b/~0key/0", "value": 3}])


def test_list_replacement_cannot_hide_removed_items_or_repeated_items(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/milestones", "replace"))
        op = [{"op": "set", "path": "/milestones", "value": ["ship"]}]
        preview = state.patch(SCOPE, op, expected_revision=1, permit_id=token, preview=True)
        assert preview["reasons"] == ["unpermitted_remove"]
        assert [c["action"] for c in preview["changes"]] == ["remove", "replace"]
        explicit = permit(state, e, ("/milestones", "replace"), ("/milestones", "remove"))
        state.patch(SCOPE, op, expected_revision=1, permit_id=explicit)
        assert state.read(SCOPE)["value"]["milestones"] == ["ship"]
    assert {c.action for c in state_diff({"a": [1, 1]}, {"a": [1]})} == {"replace", "remove"}


@pytest.mark.parametrize(
    "ops",
    [
        [
            {"op": "set", "path": "/settings", "value": {}},
            {"op": "remove", "path": "/settings/tests"},
        ],
        [{"op": "set", "path": "/settings/language"}],
        [{"op": "remove", "path": "/missing"}],
        [{"op": "set", "path": "", "value": {}}],
        [{"op": "set", "path": "/bad~2", "value": 1}],
    ],
)
def test_invalid_operations_never_change_state(tmp_path, ops):
    store, state, _ = provision(tmp_path)
    with store:
        with pytest.raises(ValueError):
            state.patch(SCOPE, ops, expected_revision=1)
        assert state.read(SCOPE)["value"] == BASE


def test_schema_normalization_cannot_silently_drop_or_inject_fields(tmp_path):
    class StrictState(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, SCOPE)
        store.grant_scope(HOST, SCOPE, ACTOR.id)
        e = store.observe_evidence(ACTOR, SCOPE, "input", "1")
        state = WorkState(store, ACTOR, schema=StrictState)
        state.initialize(SCOPE, {"value": "x"}, [e], host=HOST)
        with pytest.raises(SchemaInvalid):
            state.replace(SCOPE, {}, expected_revision=1)
        with pytest.raises(SchemaInvalid):
            state.replace(SCOPE, {"value": "x", "hidden": "drop"}, expected_revision=1)


def test_concurrent_writers_one_cas_winner_and_transaction_failure_rolls_back(
    tmp_path, monkeypatch
):
    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/notes", "replace"))

        def update(value):
            try:
                return state.patch(
                    SCOPE,
                    [{"op": "set", "path": "/notes", "value": value}],
                    expected_revision=1,
                    permit_id=token,
                )
            except VersionConflict as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(update, ["a", "b"]))
        assert sum(isinstance(result, VersionConflict) for result in results) == 1
        token = permit(state, e, ("/notes", "remove"), revision=2)
        old_audit = state._audit

        def fail(scope, event, revision, **metadata):
            if event == "committed":
                raise OSError("simulated write failure")
            return old_audit(scope, event, revision, **metadata)

        monkeypatch.setattr(state, "_audit", fail)
        with pytest.raises(OSError):
            state.patch(
                SCOPE, [{"op": "remove", "path": "/notes"}], expected_revision=2, permit_id=token
            )
        assert state.read(SCOPE)["revision"] == 2
        assert "notes" in state.read(SCOPE)["value"]


def test_state_audit_is_content_free_and_corruption_fails_closed(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        with pytest.raises(StateUpdateRejected):
            state.patch(SCOPE, [{"op": "remove", "path": "/notes"}], expected_revision=1)
        audit = [dict(row) for row in store.db.execute("SELECT * FROM work_state_audit")]
        assert any(row["event"] == "rejected" for row in audit)
        assert "required" not in json.dumps(audit) and "release" not in json.dumps(audit)
        store.db.execute("UPDATE work_state_revisions SET payload=?", (b'{"tampered":true}',))
        with pytest.raises(StoreCorruption):
            state.read(SCOPE)


async def test_native_tools_auto_install_and_refresh_current_part_after_patch(tmp_path):
    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/settings/language", "replace"))
        step = 0

        def reply(messages, info):
            nonlocal step
            step += 1
            names = {tool.name for tool in info.function_tools}
            assert {
                "context_get_work_state",
                "context_preview_work_state",
                "context_update_work_state",
            } <= names
            assert not any("authorize" in name for name in names)
            if step == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "context_update_work_state",
                            {
                                "operations": [
                                    {"op": "set", "path": "/settings/language", "value": "en"}
                                ],
                                "expected_revision": 1,
                                "permit_id": token,
                            },
                            "patch",
                        )
                    ]
                )
            assert '"language":"en"' in str(messages).replace(" ", "").replace("\\", "")
            assert "required" in str(messages)
            return ModelResponse(parts=[TextPart("updated")])

        async with ContextManager(
            store=store,
            principal=ACTOR,
            work_state=state,
            compiler_policy=CompilerPolicy(max_tokens=128000),
        ) as manager:
            result = await Agent(
                FunctionModel(reply),
                capabilities=[ReactiveContextCapability(manager, scope=SCOPE, cache=None)],
            ).run("Change language only")
            assert result.output == "updated"
        assert state.read(SCOPE)["value"]["settings"] == {"language": "en", "tests": "required"}


async def test_live_frame_sees_another_instance_update_and_permission_revocation(tmp_path):
    from reactive_context.core.snapshot import thaw

    store, state, e = provision(tmp_path)
    with store:
        async with ContextManager(store=store, principal=ACTOR, work_state=state) as manager:
            first = await manager.next_context(SCOPE)
            token = permit(state, e, ("/notes", "replace"))
            permitted = await manager.next_context(SCOPE)
            assert first.part(state.key).revision != permitted.part(state.key).revision
            other = WorkState(store, ACTOR)
            other.patch(
                SCOPE,
                [{"op": "set", "path": "/notes", "value": "from another writer"}],
                expected_revision=1,
                permit_id=token,
            )
            updated = await manager.next_context(SCOPE)
            value = thaw(updated.part(state.key).value)
            assert value["revision"] == 2 and value["value"]["notes"] == "from another writer"
            assert value["permits"] == []
            token = permit(other, e, ("/notes", "remove"), revision=2)
            await manager.next_context(SCOPE)
            other.revoke_permit(SCOPE, token, host=HOST)
            revoked = await manager.next_context(SCOPE)
            assert thaw(revoked.part(state.key).value)["permits"] == []


async def test_objective_and_state_tools_coexist_and_denied_native_patch_can_be_repaired(tmp_path):
    from reactive_context import Objective, ObjectivePolicy

    store, state, e = provision(tmp_path)
    with store:
        token = permit(state, e, ("/settings/language", "replace"))
        objective = Objective(store, ACTOR, policy=ObjectivePolicy(update_mode="agent"))
        step = 0

        def reply(messages, info):
            nonlocal step
            step += 1
            names = {tool.name for tool in info.function_tools}
            assert {"context_update_objective", "context_update_work_state"} <= names
            if step == 1:
                op = {"op": "remove", "path": "/settings/tests"}
            elif step == 2:
                assert "unpermitted_remove" in str(messages)
                assert state.read(SCOPE)["value"] == BASE
                op = {"op": "set", "path": "/settings/language", "value": "en"}
            else:
                assert state.read(SCOPE)["value"]["settings"]["tests"] == "required"
                return ModelResponse(parts=[TextPart("done")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "context_update_work_state",
                        {"operations": [op], "expected_revision": 1, "permit_id": token},
                        str(step),
                    )
                ]
            )

        async with ContextManager(
            store=store,
            principal=ACTOR,
            work_state=state,
            objective=objective,
            compiler_policy=CompilerPolicy(max_tokens=128000),
        ) as manager:
            await Agent(
                FunctionModel(reply),
                capabilities=[ReactiveContextCapability(manager, scope=SCOPE, cache=None)],
            ).run("Change language only")
        assert step == 3
