import asyncio
import json
import stat
import tempfile
from pathlib import Path

import pytest

from reactive_context.bridge import JsonRpcBridge
from reactive_context.manager import ContextManager
from reactive_context.sources.config import ConfigSource
from reactive_context.store import ContextStore, Principal

HOST = Principal("host", admin=True)
USER = Principal("agent")


def request(method="query", key="one", arguments=None, scope="chat"):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {
            "scope": scope,
            "invocation": "test",
            "idempotency_key": key,
            "arguments": arguments or {},
        },
    }


@pytest.fixture
def setup(tmp_path):
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        store.grant_scope(HOST, "chat", USER.id)
        store.register_schema("note", dict[str, str])

        def factory(principal):
            return ContextManager(store=store, principal=principal)

        yield store, JsonRpcBridge(factory, store)


async def test_bridge_fact_creation_is_idempotent_and_identity_is_host_bound(setup):
    store, bridge = setup
    evidence = store.observe_evidence(USER, "chat", "source", "r1")
    payload = request(
        "create_fact",
        arguments={
            "schema_id": "note",
            "key": "fact",
            "value": {"text": "x"},
            "evidence": [evidence],
        },
    )
    first = await bridge.dispatch(payload, USER)
    second = await bridge.dispatch({**payload, "id": 2}, USER)
    assert first["result"] == second["result"] and second["id"] == 2
    assert store.db.execute("SELECT COUNT(*) FROM parts WHERE kind='fact'").fetchone()[0] == 1
    assert first["result"]["verification"] == "unverified"
    outsider = await bridge.dispatch(payload, Principal("outsider"))
    assert outsider["error"]["data"]["kind"] == "access_revoked"
    spoof = request(key="spoof")
    spoof["params"]["principal"] = {"id": "host", "admin": True}
    assert (await bridge.dispatch(spoof, USER))["error"]["data"]["kind"] == "invalid_request"


async def test_bridge_cached_response_cannot_bypass_part_acl(setup):
    store, bridge = setup
    evidence = store.observe_evidence(USER, "chat", "source", "r1")
    part = store.commit(
        USER, store.propose_fact(USER, "chat", "note", "key", {"secret": "value"}, [evidence])
    )
    payload = request("load", arguments={"part_id": part.id})
    assert "result" in await bridge.dispatch(payload, USER)
    store.revoke_part_access(HOST, part.id, USER.id)
    response = await bridge.dispatch(payload, USER)
    assert "result" not in response and "secret" not in json.dumps(response)


async def test_in_doubt_request_does_not_blindly_repeat_mutation(setup):
    store, bridge = setup
    await bridge.dispatch(request(), USER)
    store.db.execute("UPDATE bridge_requests SET state='running'")
    result = await bridge.dispatch(request(), USER)
    assert "error" in result
    assert store.db.execute("SELECT COUNT(*) FROM bridge_requests").fetchone()[0] == 1


async def test_unix_ipc_roundtrip_and_dispose(setup, tmp_path):
    store, bridge = setup
    directory = tempfile.TemporaryDirectory(prefix="rc-", dir="/tmp")
    path = Path(directory.name) / "context.sock"
    async with bridge.unix_server(path, principal=USER):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        reader, writer = await asyncio.open_unix_connection(path)
        writer.write(json.dumps(request()).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
        assert response["result"] == []
        writer.close()
        await writer.wait_closed()
    assert not path.exists()
    directory.cleanup()


async def test_bridge_external_source_acl_is_rechecked_for_cached_payload(tmp_path):
    allowed = True
    source = ConfigSource("rules", "private", acl_check=lambda ctx: allowed)
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        store.grant_scope(HOST, "chat", USER.id)
        async with ContextManager([source.part], store=store, principal=USER) as manager:
            manager.publish_catalog(HOST, "chat")
            part_id = (await manager.query("chat"))[0]["id"]
        bridge = JsonRpcBridge(
            lambda principal: ContextManager([source.part], store=store, principal=principal), store
        )
        payload = request("load", arguments={"part_id": part_id})
        assert "result" in await bridge.dispatch(payload, USER)
        allowed = False
        assert "error" in await bridge.dispatch(payload, USER)


async def test_forget_purges_payload_bearing_bridge_responses(setup):
    store, bridge = setup
    evidence = store.observe_evidence(USER, "chat", "source", "r1")
    part = store.commit(
        USER, store.propose_fact(USER, "chat", "note", "key", {"text": "FORGET_THIS"}, [evidence])
    )
    assert "result" in await bridge.dispatch(request("load", arguments={"part_id": part.id}), USER)
    store.retire(USER, part.id, status="forgotten")
    responses = store.db.execute("SELECT response FROM bridge_requests").fetchall()
    assert all(r[0] is None or b"FORGET_THIS" not in bytes(r[0]) for r in responses)


async def test_bridge_compile_tracks_current_resource_and_request_budget(tmp_path):
    source = ConfigSource("schema", "version one")
    with ContextStore(tmp_path) as store:
        store.create_scope(HOST, "chat")
        store.grant_scope(HOST, "chat", USER.id)
        async with ContextManager([source.part], store=store, principal=USER) as manager:
            manager.publish_catalog(HOST, "chat")
            part_id = (await manager.query("chat"))[0]["id"]
        bridge = JsonRpcBridge(
            lambda principal: ContextManager([source.part], store=store, principal=principal), store
        )
        assert "result" in await bridge.dispatch(
            request("activate", arguments={"part_id": part_id}), USER
        )
        first = await bridge.dispatch(request("compile", key="compile-one"), USER)
        source.set("version two")
        second = await bridge.dispatch(request("compile", key="compile-two"), USER)
        assert first["result"]["revision_vector"] != second["result"]["revision_vector"]
        assert first["result"]["compile_id"] != second["result"]["compile_id"]
        assert "version two" in str(second["result"]["messages"])
        source.set("large payload " * 2000)
        response = await bridge.dispatch(
            request("compile", key="small-budget", arguments={"budget": 2049}), USER
        )
        assert "error" in response
        assert response["error"]["data"]["kind"] == "context_budget_exceeded"
