"""Versioned JSON-RPC boundary for host-authenticated adapters; no agent loop or HTTP exposure.

Unix transport binds a host principal through filesystem permissions. Other transports must pass
an authenticated Principal separately; model-controlled JSON cannot claim an identity or ACL.
Interrupted mutations remain in-doubt instead of being silently executed twice after a crash.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .core.snapshot import thaw
from .errors import ContextBudgetExceeded, ContextError
from .store import Principal


class BridgeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["0.2"] = "0.2"
    scope: str = Field(min_length=1, max_length=200)
    invocation: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=200)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    arguments: dict[str, Any] = Field(default_factory=dict)


class BridgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    method: Literal["query", "load", "activate", "deactivate", "create_fact", "compile", "observe"]
    params: BridgeParams


class JsonRpcBridge:
    def __init__(self, manager_factory, store, *, max_response_bytes=1024 * 1024):
        self.manager_factory, self.store = manager_factory, store
        self.max_response_bytes = max_response_bytes
        self._lock = asyncio.Lock()
        self.store.db.execute(
            "CREATE TABLE IF NOT EXISTS bridge_authorizations "
            "(principal TEXT, scope TEXT, key TEXT, part_id TEXT, "
            "PRIMARY KEY(principal,scope,key,part_id))"
        )
        self.store.db.execute(
            "CREATE TABLE IF NOT EXISTS bridge_requests (principal TEXT, scope TEXT, "
            "key TEXT, digest TEXT, state TEXT, response BLOB, "
            "PRIMARY KEY(principal,scope,key))"
        )

    async def dispatch(self, payload, principal: Principal):
        request_id = payload.get("id") if isinstance(payload, dict) else None
        try:
            request = BridgeRequest.model_validate(payload)
            params = request.params
            # Re-authorize even when returning an idempotently cached response.
            self.store._scope_access(principal, params.scope)
            digest = sha256(
                json.dumps(
                    {"method": request.method, "params": params.model_dump()}, sort_keys=True
                ).encode()
            ).hexdigest()
            async with self._lock:
                with self.store.transaction():
                    saved = self.store.db.execute(
                        "SELECT * FROM bridge_requests WHERE principal=? AND scope=? AND key=?",
                        (principal.id, params.scope, params.idempotency_key),
                    ).fetchone()
                    if not saved:
                        self.store.db.execute(
                            "INSERT INTO bridge_requests VALUES(?,?,?,?,?,NULL)",
                            (principal.id, params.scope, params.idempotency_key, digest, "running"),
                        )
                if saved:
                    if saved["digest"] != digest:
                        raise ValueError("idempotency_key_conflict")
                    if saved["state"] != "complete":
                        raise ValueError("request_in_doubt_reconcile_before_retry")
                    cached = json.loads(saved["response"])
                    checker = self.manager_factory(principal)
                    try:
                        for part_id in cached.get("authorized_parts", []):
                            meta = self.store.metadata(principal, part_id)
                            if meta["kind"] != "fact":
                                await checker._authorize_resource(params.scope, meta["key"])
                    finally:
                        await checker.close()
                    cached.pop("authorized_parts", None)
                    return {**cached, "id": request.id}
                manager = self.manager_factory(principal)
                try:
                    async with asyncio.timeout(params.timeout_seconds):
                        result, authorized = await self._execute(manager, request)
                    envelope = {
                        "jsonrpc": "2.0",
                        "id": request.id,
                        "result": result,
                        "authorized_parts": authorized,
                    }
                    body = json.dumps(envelope).encode()
                    if len(body) > self.max_response_bytes:
                        raise ValueError("response_limit_exceeded")
                    with self.store.transaction():
                        self.store.db.execute(
                            "UPDATE bridge_requests SET state='complete',response=? "
                            "WHERE principal=? AND scope=? AND key=?",
                            (body, principal.id, params.scope, params.idempotency_key),
                        )
                        self.store.db.executemany(
                            "INSERT OR IGNORE INTO bridge_authorizations VALUES(?,?,?,?)",
                            [
                                (principal.id, params.scope, params.idempotency_key, part_id)
                                for part_id in authorized
                            ],
                        )
                    envelope.pop("authorized_parts")
                    return envelope
                finally:
                    await manager.close()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = (
                error.code
                if isinstance(error, ContextError)
                else (
                    "invalid_request"
                    if isinstance(error, ValidationError)
                    else type(error).__name__
                )
            )
            details: dict[str, Any] = {"kind": code}
            if isinstance(error, ValueError) and str(error) in {
                "idempotency_key_conflict",
                "request_in_doubt_reconcile_before_retry",
            }:
                details["reason"] = str(error)
            if isinstance(error, ContextBudgetExceeded):
                details.update(
                    parts=error.parts,
                    estimated_tokens=error.estimated_tokens,
                    suggested_scopes=error.suggested_scopes,
                )
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": "Context operation failed", "data": details},
            }

    async def _execute(self, manager, request):
        scope, args = request.params.scope, request.params.arguments
        method = request.method
        if method == "query":
            result = await manager.query(scope, **args)
            return result, [r["id"] for r in result]
        if method == "load":
            loaded = await manager.load(scope, **args)
            return {
                "part_id": loaded.id,
                "revision": loaded.snapshot.revision,
                "value": thaw(loaded.snapshot.value),
                "evidence_id": loaded.evidence_id,
            }, [loaded.id]
        if method == "activate":
            return asdict(await manager.activate(scope, **args)), [args["part_id"]]
        if method == "deactivate":
            await manager.deactivate(scope, **args)
            return {"deactivated": True}, [args["part_id"]]
        if method == "create_fact":
            proposal = self.store.propose_fact(manager.principal, scope, **args)
            fact = self.store.commit(manager.principal, proposal)
            return {
                "part_id": fact.id,
                "revision": fact.revision,
                "verification": fact.verification,
            }, [fact.id]
        if method == "observe":
            if not manager.principal.admin:
                from .errors import AccessRevoked

                raise AccessRevoked(detail="source observation is runtime-only")
            if set(args) != {"part_key"}:
                raise ValueError("invalid observe arguments")
            manager.notify(args["part_key"], scope=scope)
            return {"observed": True}, []
        if set(args) - {"budget"}:
            raise ValueError("unknown compile argument")
        budget = args.get("budget", manager.compiler.policy.max_tokens)
        if not isinstance(budget, int) or budget <= manager.compiler.policy.output_reserve:
            raise ValueError("invalid request budget")
        snapshot = await manager.next_context(scope)
        reserve = max(0, manager.compiler.policy.max_tokens - budget)
        compiled = manager.compile(snapshot, history_tokens=reserve)
        ids = []
        for part in snapshot.parts:
            row = self.store.db.execute(
                "SELECT id FROM parts WHERE scope_id=? AND (id=? OR key=?)",
                (scope, part.resource_id, part.part_id),
            ).fetchone()
            if row:
                ids.append(row[0])
        return {
            "protocol_version": "0.2",
            "compile_id": compiled.compile_id,
            "snapshot_id": snapshot.id,
            "revision_vector": dict(snapshot.revision_vector),
            "messages": [asdict(m) for m in compiled.provider_messages],
            "tools": thaw(compiled.provider_tools),
            "trace": [asdict(t) for t in compiled.trace],
            "estimated_tokens": compiled.token_estimate - reserve,
            "capabilities": {"history_replacement": False, "host_compaction": True},
        }, ids

    @asynccontextmanager
    async def unix_server(self, path, *, principal: Principal, max_request_bytes=1024 * 1024):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.parent.stat().st_mode & 0o022:
            raise PermissionError("IPC directory must not be group/world writable")
        if path.exists():
            raise FileExistsError(path)

        clients = set()

        async def client(reader, writer):
            task = asyncio.current_task()
            clients.add(task)
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=30)
                if not line or len(line) > max_request_bytes:
                    return
                try:
                    payload = json.loads(line)
                except ValueError:
                    payload = None
                response = await self.dispatch(payload, principal)
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
            finally:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=5)
                except (ConnectionError, TimeoutError):
                    pass
                clients.discard(task)

        server = await asyncio.start_unix_server(client, path=path, limit=max_request_bytes + 1)
        os.chmod(path, 0o600)
        try:
            yield server
        finally:
            server.close()
            await server.wait_closed()
            for task in list(clients):
                task.cancel()
            await asyncio.gather(*clients, return_exceptions=True)
            path.unlink(missing_ok=True)
