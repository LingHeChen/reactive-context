"""Run offline: .venv/bin/python examples/structured_work_state.py"""

import asyncio
import json
from tempfile import TemporaryDirectory

from reactive_context import ContextManager, ContextStore, Principal, WorkState


async def main():
    host, agent = Principal("host", True), Principal("agent")
    with TemporaryDirectory(prefix="work-state-") as path, ContextStore(path) as store:
        store.create_scope(host, "release")
        store.grant_scope(host, "release", agent.id)
        evidence = store.observe_evidence(agent, "release", "user-request", "1")
        state = WorkState(store, agent)
        initial = {
            "target": "ship",
            "requirements": {"language": "zh", "tests": "pass"},
            "tasks": {"review": {"status": "pending"}, "deploy": {"status": "pending"}},
        }
        state.initialize("release", initial, [evidence], host=host)
        # Application code decides the current request permits marking review done.
        # This is not an extra user confirmation, and is not an agent-callable grant.
        permit = state.authorize(
            "release",
            [{"path": "/tasks/review/status", "action": "replace"}],
            [evidence],
            expected_revision=1,
            host=host,
        )
        async with ContextManager(store=store, principal=agent, work_state=state) as manager:
            results = []
            missing = state.replace(
                "release",
                {"tasks": {"review": {"status": "done"}}},
                expected_revision=1,
                permit_id=permit,
                preview=True,
            )
            assert missing["decision"] == "reject"
            results.append({"case": "incomplete_full_replacement", **missing})
            operations = [{"op": "set", "path": "/tasks/review/status", "value": "done"}]
            result = state.patch("release", operations, expected_revision=1, permit_id=permit)
            results.append({"case": "incremental_update", **result})
            assert state.read("release")["value"]["tasks"]["deploy"] == {"status": "pending"}
            same = state.patch("release", operations, expected_revision=2)
            results.append({"case": "identical_replay", **same})
            clear_permit = state.authorize(
                "release",
                [{"path": "/requirements/tests", "action": "replace"}],
                [evidence],
                expected_revision=2,
                host=host,
            )
            blocked = state.patch(
                "release",
                [{"op": "set", "path": "/requirements/tests", "value": ""}],
                expected_revision=2,
                permit_id=clear_permit,
                preview=True,
            )
            assert blocked["reasons"] == ["unpermitted_clear"]
            results.append({"case": "implicit_clear", **blocked})
            cancel = store.observe_evidence(agent, "release", "user-cancels-deployment", "2")
            remove_permit = state.authorize(
                "release",
                [{"path": "/tasks/deploy/status", "action": "remove"}],
                [cancel],
                expected_revision=2,
                host=host,
            )
            deleted = state.patch(
                "release",
                [{"op": "remove", "path": "/tasks/deploy"}],
                expected_revision=2,
                permit_id=remove_permit,
            )
            results.append({"case": "explicit_cancellation", **deleted})
            snapshot = await manager.next_context("release")
            assert snapshot.part(state.key).value["revision"] == 3
            print(
                json.dumps(
                    {"results": results, "final": state.read("release")},
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
