import concurrent.futures
import json

import pytest
from pydantic import BaseModel

from reactive_context.errors import AccessRevoked, SchemaInvalid
from reactive_context.store import ContextStore, Principal, StoreCorruption, VersionConflict

HOST = Principal("host", admin=True)
ALICE = Principal("alice")
BOB = Principal("bob")


class Fact(BaseModel):
    text: str


@pytest.fixture
def store(tmp_path):
    with ContextStore(tmp_path, cache_bytes=128, inline_bytes=20) as store:
        store.register_schema("note", Fact)
        store.create_scope(HOST, "a")
        store.create_scope(HOST, "b")
        store.grant_scope(HOST, "a", "alice")
        store.grant_scope(HOST, "a", "bob")
        store.grant_scope(HOST, "b", "bob")
        yield store


def create(store, key="one", text="a" * 30, actor=ALICE, scope="a", expected=None):
    evidence = store.observe_evidence(actor, scope, "file://source", "release-1")
    proposal = store.propose_fact(
        actor, scope, "note", key, {"text": text}, [evidence], expected_revision=expected
    )
    return store.commit(actor, proposal)


def test_fact_revisions_are_immutable_and_unverified(store):
    first = create(store)
    second = create(store, text="new", expected=1)
    assert first.id == second.id
    assert first.revision == 1 and second.revision == 2
    assert second.verification == "unverified"
    assert store.load(ALICE, first.id, revision=1).value.text == "a" * 30
    with pytest.raises(TypeError):
        first.value.text = "changed"
    with pytest.raises(VersionConflict):
        create(store, text="overwrite", expected=1)


def test_schema_and_evidence_validation(store):
    with pytest.raises(SchemaInvalid):
        store.propose_fact(ALICE, "a", "missing", "key", {}, ["made-up"])
    with pytest.raises(AccessRevoked):
        store.propose_fact(ALICE, "a", "note", "key", {"text": "x"}, ["made-up"])
    evidence = store.observe_evidence(BOB, "b", "external", "r")
    with pytest.raises(AccessRevoked):
        store.propose_fact(ALICE, "a", "note", "key", {"text": "x"}, [evidence])


def test_acl_applies_to_query_load_activate_and_cached_reads(store):
    fact = create(store, text="confidential")
    assert store.query(BOB, "a", "confidential") == []
    with pytest.raises(AccessRevoked):
        store.load(BOB, fact.id)
    store.grant_part(HOST, fact.id, "bob")
    assert len(store.query(BOB, "a", "confidential")) == 1
    store.load(BOB, fact.id)
    store.activate(BOB, "a", fact.id)
    store.revoke_part_access(HOST, fact.id, "bob")
    with pytest.raises(AccessRevoked):
        store.load(BOB, fact.id)
    with pytest.raises(AccessRevoked):
        store.active_refs(BOB, "a")
    with pytest.raises(AccessRevoked):
        store.activate(BOB, "a", fact.id)
    with pytest.raises(AccessRevoked):
        store.activate(ALICE, "b", fact.id)


def test_hot_cache_hard_limit_and_active_ref_does_not_load(store):
    facts = [create(store, key=str(i)) for i in range(20)]
    store.cache.clear()
    for fact in facts:
        store.activate(ALICE, "a", fact.id)
    assert store.cache.stats.bytes == 0
    for fact in facts:
        store.load(ALICE, fact.id)
        assert store.cache.stats.bytes <= 128
    assert store.cache.stats.evictions > 0
    assert len(store.active_refs(ALICE, "a")) == 20


def test_restart_and_second_connection_reconcile(store):
    fact = create(store)
    store.activate(ALICE, "a", fact.id, policy="pinned")
    with ContextStore(store.directory) as other:
        other.register_schema("note", Fact)
        assert other.active_refs(ALICE, "a")[0].revision == 1
        assert other.load(ALICE, fact.id).value.text == "a" * 30
        create(store, text="new revision", expected=1)
        assert other.load(ALICE, fact.id).revision == 2
        store.revoke_scope(HOST, "a", "alice")
        with pytest.raises(AccessRevoked):
            other.load(ALICE, fact.id)


def test_cross_connection_optimistic_writes(store):
    fact = create(store)
    evidence = store.observe_evidence(ALICE, "a", "file://source", "r2")
    proposals = [
        store.propose_fact(
            ALICE, "a", "note", "one", {"text": str(i)}, [evidence], expected_revision=1
        )
        for i in range(2)
    ]

    def commit(proposal):
        with ContextStore(store.directory) as other:
            other.register_schema("note", Fact)
            try:
                return other.commit(ALICE, proposal).revision
            except VersionConflict:
                return "conflict"

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        result = list(pool.map(commit, proposals))
    assert sorted(map(str, result)) == ["2", "conflict"]
    assert store.load(ALICE, fact.id).revision == 2


def test_gc_respects_revision_leases_and_pinned_refs(store):
    fact = create(store)
    lease = store.lease(ALICE, fact.id, 1)
    create(store, text="new" * 20, expected=1)
    store.gc(HOST, prune_revisions=True)
    assert store.load(ALICE, fact.id, revision=1).revision == 1
    store.activate(ALICE, "a", fact.id, policy="pinned", revision=1)
    store.release(lease)
    store.gc(HOST, prune_revisions=True)
    assert store.load(ALICE, fact.id, revision=1).revision == 1
    store.deactivate(ALICE, "a", fact.id)
    assert store.gc(HOST, prune_revisions=True)["orphan_objects_removed"] == 1
    with pytest.raises(VersionConflict):
        store.load(ALICE, fact.id, revision=1)


def test_cas_corruption_is_fail_closed(store):
    fact = create(store)
    path = next(p for p in store.objects.rglob("*") if p.is_file())
    path.write_bytes(b"corrupt")
    store.cache.clear()
    with pytest.raises(StoreCorruption):
        store.load(ALICE, fact.id)
    assert store.integrity(HOST)["bad_revisions"]


def test_crash_before_sql_commit_leaves_only_collectable_orphan(store, monkeypatch):
    evidence = store.observe_evidence(ALICE, "a", "file://source", "r")
    proposal = store.propose_fact(ALICE, "a", "note", "key", {"text": "large" * 30}, [evidence])
    original = store._emit

    def fail(kind, *args, **kwargs):
        if kind == "fact_committed":
            raise OSError("injected write failure")
        return original(kind, *args, **kwargs)

    monkeypatch.setattr(store, "_emit", fail)
    with pytest.raises(OSError):
        store.commit(ALICE, proposal)
    monkeypatch.setattr(store, "_emit", original)
    assert store.query(ALICE, "a") == []
    assert store.gc(HOST)["orphan_objects_removed"] == 1
    assert store.commit(ALICE, proposal).revision == 1


def test_forget_and_revoke_remove_fts_cache_and_active_refs(store):
    first = create(store, text="secret")
    second = create(store, key="replacement")
    store.activate(ALICE, "a", first.id)
    store.retire(ALICE, first.id, status="superseded", superseded_by=second.id)
    assert store.query(ALICE, "a", "secret") == []
    assert store.active_refs(ALICE, "a") == []
    store.retire(ALICE, second.id, status="forgotten")
    with pytest.raises(AccessRevoked):
        store.load(ALICE, second.id)
    store.gc(HOST)
    # Retained superseded revision is historical, not active; forgotten payload is removed.
    assert (
        store.db.execute("SELECT COUNT(*) FROM revisions WHERE part_id=?", (second.id,)).fetchone()[
            0
        ]
        == 0
    )
    assert "secret" not in json.dumps(store.events_since(ALICE, "a"))


def test_schema_contract_cannot_change_after_restart(store):
    with ContextStore(store.directory) as other:
        with pytest.raises(ValueError):
            other.register_schema("note", dict[str, int])
        other.register_schema("note", dict[str, int], version=2)


def test_process_crash_rolls_back_wal_and_orphan_cas_is_collectable(tmp_path):
    import subprocess
    import sys

    script = """
import os,sys
from reactive_context.store import ContextStore,Principal
s=ContextStore(sys.argv[1],inline_bytes=1)
a=Principal('host',admin=True)
s.create_scope(a,'task');s.register_schema('note',dict[str,str])
e=s.observe_evidence(a,'task','source','r1')
p=s.propose_fact(a,'task','note','one',{'text':'payload'},[e])
original=s._emit
def crash(kind,*args,**kwargs):
    if kind=="fact_committed":os._exit(17)
    return original(kind,*args,**kwargs)
s._emit=crash
s.commit(a,p)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert result.returncode == 17
    with ContextStore(tmp_path) as recovered:
        recovered.register_schema("note", dict[str, str])
        assert recovered.query(HOST, "task") == []
        assert recovered.integrity(HOST)["sqlite"] == "ok"
        assert recovered.gc(HOST)["orphan_objects_removed"] == 1


def test_batch_rollback_and_keyset_pagination(store):
    with pytest.raises(RuntimeError):
        with store.batch():
            create(store, key="rolled-back")
            raise RuntimeError("abort batch")
    assert store.query(ALICE, "a") == []
    for i in range(5):
        create(store, key=f"note-{i}")
    first = store.query(ALICE, "a", limit=2)
    second = store.query(ALICE, "a", limit=3, after_id=first[-1]["id"])
    assert len({r["id"] for r in first + second}) == 5
