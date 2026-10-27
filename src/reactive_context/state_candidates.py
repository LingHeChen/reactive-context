"""Scoped candidate inbox: checking never activates data, committing always rechecks."""

import json
from time import time
from uuid import uuid4

from .core.snapshot import digest
from .errors import AccessRevoked
from .store import StoreCorruption, VersionConflict
from .work_state import StateOperation, StateUpdateRejected


class StateCandidates:
    """Companion owned by WorkState. Records hold patches, not duplicate full plans."""

    def __init__(self, state):
        self.state, self.store = state, state.store
        with self.store.transaction():
            self.store.db.execute(
                "CREATE TABLE IF NOT EXISTS work_state_candidates("
                "id TEXT PRIMARY KEY,scope TEXT,key TEXT,principal TEXT,payload BLOB,digest TEXT,"
                "status TEXT,last_check TEXT,committed_revision INTEGER,created_at REAL)"
            )
            self.store.db.execute(
                "CREATE INDEX IF NOT EXISTS work_state_candidate_identity "
                "ON work_state_candidates(scope,key,principal,digest)"
            )

    def _row(self, scope, identity, *, write=False):
        self.store.authorize_scope(self.state.principal, scope)
        if write:
            self.store.authorize_scope(self.state.principal, scope, "write")
        row = self.store.db.execute(
            "SELECT * FROM work_state_candidates WHERE id=? AND scope=? AND key=? AND principal=?",
            (identity, scope, self.state.key, self.state.principal.id),
        ).fetchone()
        if row is None:
            raise AccessRevoked(detail="candidate unavailable")
        payload = json.loads(row["payload"])
        if digest(payload) != row["digest"]:
            raise StoreCorruption()
        return row, payload

    def read(self, scope, candidate_id):
        with self.store.transaction():
            row, payload = self._row(scope, candidate_id)
            return {
                "candidate_id": candidate_id,
                "status": row["status"],
                **payload,
                "last_check": json.loads(row["last_check"]),
                "committed_revision": row["committed_revision"],
            }

    @staticmethod
    def _status(receipt):
        validation = receipt["validation"]
        if validation["status"] == "structural_only":
            return "pending"
        if receipt["decision"] == "reject":
            return "pending" if validation["status"] == "pending" else "rejected"
        return "ready"

    def propose(self, scope, operations, evidence, *, expected_revision, permit_id=None):
        operations = [
            StateOperation.model_validate(op).model_dump(mode="json", exclude_unset=True)
            for op in operations
        ]
        if not evidence:
            raise ValueError("a candidate needs observed evidence")
        with self.store.transaction():
            # Preview runs the same permission and value checks as direct updates.
            # Failed/pending business checks are kept for inspection, not activated.
            receipt = self.state.patch(
                scope,
                operations,
                expected_revision=expected_revision,
                permit_id=permit_id,
                evidence=evidence,
                preview=True,
            )
            if any(not reason.startswith("value_validation_") for reason in receipt["reasons"]):
                raise StateUpdateRejected(receipt["reasons"])
            payload = {
                "operations": operations,
                "evidence": list(evidence),
                "expected_revision": expected_revision,
                "permit_id": permit_id,
                "requirements_revision": self.state.requirements(scope)["revision"],
            }
            checksum = digest(payload)
            previous = self.store.db.execute(
                "SELECT id FROM work_state_candidates WHERE scope=? AND key=? AND principal=? "
                "AND digest=? AND status!='discarded' ORDER BY created_at DESC LIMIT 1",
                (scope, self.state.key, self.state.principal.id, checksum),
            ).fetchone()
            identity = previous["id"] if previous else uuid4().hex
            status = self._status(receipt)
            if receipt["validation"]["status"] == "structural_only":
                receipt["reasons"] = [*receipt["reasons"], "value_checker_required"]
            if previous:
                self.store.db.execute(
                    "UPDATE work_state_candidates SET status=?,last_check=? WHERE id=?",
                    (status, json.dumps(receipt), identity),
                )
            else:
                self.store.db.execute(
                    "INSERT INTO work_state_candidates VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        identity,
                        scope,
                        self.state.key,
                        self.state.principal.id,
                        json.dumps(payload).encode(),
                        checksum,
                        status,
                        json.dumps(receipt),
                        None,
                        time(),
                    ),
                )
            self.state._audit(scope, "candidate_proposed", expected_revision, status=status)
            return self.read(scope, identity)

    def _recheck(self, scope, identity):
        row, payload = self._row(scope, identity, write=True)
        if row["status"] in {"committed", "discarded"}:
            return self.read(scope, identity), None
        current = self.state.read(scope)
        if current["revision"] != payload["expected_revision"] or (
            current["requirements"]["revision"] != payload["requirements_revision"]
        ):
            receipt = {
                "decision": "reject",
                "reasons": ["candidate_base_changed"],
                "validation": {"status": "not_run", "issues": []},
            }
            status = "stale"
        else:
            receipt = self.state.patch(
                scope,
                payload["operations"],
                expected_revision=payload["expected_revision"],
                permit_id=payload["permit_id"],
                evidence=payload["evidence"],
                preview=True,
            )
            status = self._status(receipt)
            if receipt["validation"]["status"] == "structural_only":
                receipt["reasons"] = [*receipt["reasons"], "value_checker_required"]
        self.store.db.execute(
            "UPDATE work_state_candidates SET status=?,last_check=? WHERE id=?",
            (status, json.dumps(receipt), identity),
        )
        return self.read(scope, identity), receipt

    def check(self, scope, candidate_id):
        with self.store.transaction():
            result, _ = self._recheck(scope, candidate_id)
            return result

    def commit(self, scope, candidate_id):
        with self.store.transaction():
            result, receipt = self._recheck(scope, candidate_id)
            if result["status"] != "ready":
                return result
            # No externally cached verdict is trusted. state.patch checks the
            # current source/requirements once more inside this same transaction.
            committed = self.state.patch(
                scope,
                result["operations"],
                expected_revision=result["expected_revision"],
                permit_id=result["permit_id"],
                evidence=result["evidence"],
            )
            self.store.db.execute(
                "UPDATE work_state_candidates SET status='committed',last_check=?,"
                "committed_revision=? "
                "WHERE id=?",
                (json.dumps(committed), committed["revision"], candidate_id),
            )
            self.state._audit(scope, "candidate_committed", committed["revision"])
            return self.read(scope, candidate_id)

    def discard(self, scope, candidate_id):
        with self.store.transaction():
            row, _ = self._row(scope, candidate_id, write=True)
            if row["status"] == "committed":
                raise VersionConflict(detail="a committed candidate cannot be discarded")
            self.store.db.execute(
                "UPDATE work_state_candidates SET status='discarded' WHERE id=?", (candidate_id,)
            )
            return self.read(scope, candidate_id)
