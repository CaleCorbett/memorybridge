"""Regression tests for issue #181 (P2-1): orphaned embeddings, archived
filtering, and BLOB vector storage.

Covers:
  - _pack_vector / _unpack_vector round-trip (float32 BLOB, not JSON text).
  - _migrate_embeddings_v2 purges embedding rows whose parent memory no
    longer exists (the pre-existing FK-cascade cleanup only ran on a fresh
    DB, never on one already initialized before this fix existed).
  - _migrate_embeddings_v2 converts legacy JSON-text vector rows to BLOB.
  - Both migrations are idempotent (meta-flag guarded, don't re-run/re-log
    on a second construction).
  - search_semantic excludes embeddings belonging to archived memories.
"""
import json
import os

os.environ.setdefault("MEMORYBRIDGE_NO_EMBED", "1")

from db.store import MemoryStore, _pack_vector, _unpack_vector  # noqa: E402


def _reset_migration_flags(st):
    """MemoryStore.__init__ already runs _migrate_embeddings_v2() once via
    _ensure_maintenance(), so a fresh store's meta flags are already set.
    Tests that want to observe the migration actually running (against
    rows inserted after construction, simulating an already-existing
    pre-fix DB) must clear the flags first."""
    st._conn.execute("DELETE FROM meta WHERE key='embeddings_orphan_purge_v1'")
    st._conn.execute("DELETE FROM meta WHERE key='embeddings_blob_migration_v1'")
    st._conn.commit()


def _insert_orphan_embedding(st, mem_id, vector_blob):
    """Insert an embedding row with no matching memories row, bypassing the
    FK constraint -- simulates a row orphaned before PRAGMA foreign_keys=ON
    was set on the deleting connection, or written out-of-band (exactly the
    real-world case issue #181 found: 440/1144 rows orphaned this way)."""
    st._conn.execute("PRAGMA foreign_keys=OFF")
    st._conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (id, profile, vector) VALUES (?,?,?)",
        (mem_id, "default", vector_blob),
    )
    st._conn.commit()
    st._conn.execute("PRAGMA foreign_keys=ON")


# --------------------------------------------------------------------- pack/unpack
def test_pack_unpack_vector_roundtrip():
    vec = [0.1, -0.5, 3.25, 0.0]
    blob = _pack_vector(vec)
    assert isinstance(blob, (bytes, bytearray))
    restored = _unpack_vector(blob)
    assert restored is not None
    for a, b in zip(vec, restored):
        assert abs(a - b) < 1e-6


def test_unpack_vector_backward_compat_with_legacy_json_string():
    vec = [1.0, 2.0, 3.0]
    legacy = json.dumps(vec)
    restored = _unpack_vector(legacy)
    assert restored == vec


def test_unpack_vector_returns_none_on_garbage():
    assert _unpack_vector("not json and not a blob {{{") is None
    assert _unpack_vector(None) is None


# --------------------------------------------------------------------- orphan purge
def test_migrate_embeddings_purges_orphaned_rows(tmp_path):
    st = MemoryStore(tmp_path / "m.db")
    st.ensure_profile("default")
    mid = st.add_memory("default", content="alpha", category="general", importance="medium")
    _reset_migration_flags(st)

    # A legit embedding for the real memory, plus an orphan whose parent
    # memory was deleted out from under it (simulating a pre-fix DB where
    # FK cascade never ran because PRAGMA foreign_keys wasn't ON at delete
    # time, or the row predates this fix entirely).
    st._conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (id, profile, vector) VALUES (?,?,?)",
        (mid, "default", _pack_vector([1.0, 0.0])),
    )
    st._conn.commit()
    _insert_orphan_embedding(st, "mem_orphan_ghost", _pack_vector([0.0, 1.0]))

    before = st._conn.execute("SELECT COUNT(*) c FROM memory_embeddings").fetchone()["c"]
    assert before == 2

    st._migrate_embeddings_v2()

    rows = st._conn.execute("SELECT id FROM memory_embeddings").fetchall()
    ids = {r["id"] for r in rows}
    assert ids == {mid}
    assert "mem_orphan_ghost" not in ids


def test_migrate_embeddings_orphan_purge_is_idempotent(tmp_path):
    st = MemoryStore(tmp_path / "m.db")
    st.ensure_profile("default")
    st.add_memory("default", content="alpha", category="general", importance="medium")
    _reset_migration_flags(st)
    _insert_orphan_embedding(st, "mem_orphan_ghost", _pack_vector([0.0, 1.0]))

    st._migrate_embeddings_v2()
    row_count_after_first = st._conn.execute(
        "SELECT COUNT(*) c FROM memory_embeddings").fetchone()["c"]

    # Re-run the migration -- the guard flag must prevent a second orphan
    # sweep (this is a one-time hygiene pass, not a per-run scan), so
    # nothing extra should change here.
    st._migrate_embeddings_v2()
    row_count_after_second = st._conn.execute(
        "SELECT COUNT(*) c FROM memory_embeddings").fetchone()["c"]

    assert row_count_after_first == row_count_after_second == 0


# --------------------------------------------------------------------- BLOB migration
def test_migrate_embeddings_converts_legacy_json_to_blob(tmp_path):
    st = MemoryStore(tmp_path / "m.db")
    st.ensure_profile("default")
    mid = st.add_memory("default", content="alpha", category="general", importance="medium")
    _reset_migration_flags(st)

    vec = [0.25, 0.75, -0.5]
    st._conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (id, profile, vector) VALUES (?,?,?)",
        (mid, "default", json.dumps(vec)),
    )
    st._conn.commit()

    raw_before = st._conn.execute(
        "SELECT vector FROM memory_embeddings WHERE id=?", (mid,)).fetchone()["vector"]
    assert isinstance(raw_before, str)

    st._migrate_embeddings_v2()

    raw_after = st._conn.execute(
        "SELECT vector FROM memory_embeddings WHERE id=?", (mid,)).fetchone()["vector"]
    assert isinstance(raw_after, (bytes, bytearray))
    restored = _unpack_vector(raw_after)
    for a, b in zip(vec, restored):
        assert abs(a - b) < 1e-6


def test_migrate_embeddings_blob_migration_is_idempotent(tmp_path):
    st = MemoryStore(tmp_path / "m.db")
    st.ensure_profile("default")
    mid = st.add_memory("default", content="alpha", category="general", importance="medium")
    _reset_migration_flags(st)
    st._conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (id, profile, vector) VALUES (?,?,?)",
        (mid, "default", json.dumps([1.0, 2.0])),
    )
    st._conn.commit()

    st._migrate_embeddings_v2()
    first = st._conn.execute(
        "SELECT vector FROM memory_embeddings WHERE id=?", (mid,)).fetchone()["vector"]

    # Second run should be a no-op (meta flag set) and not error or re-touch rows.
    st._migrate_embeddings_v2()
    second = st._conn.execute(
        "SELECT vector FROM memory_embeddings WHERE id=?", (mid,)).fetchone()["vector"]

    assert bytes(first) == bytes(second)


# --------------------------------------------------------------------- archived filter
def test_search_semantic_excludes_archived_memories(tmp_path):
    st = MemoryStore(tmp_path / "m.db")
    st.ensure_profile("default")

    live_id = st.add_memory("default", content="live memory", category="general", importance="medium")
    archived_id = st.add_memory("default", content="archived memory", category="general", importance="medium")

    st._embed_texts = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    for mid in (live_id, archived_id):
        st._conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (id, profile, vector) VALUES (?,?,?)",
            (mid, "default", _pack_vector([1.0, 0.0, 0.0])),
        )
    st._conn.execute(
        "UPDATE memories SET archived=1, archived_at=?, archive_reason='test' WHERE id=?",
        ("2026-01-01T00:00:00", archived_id),
    )
    st._conn.commit()

    results = st.search_semantic("default", "irrelevant query", limit=10, max_tokens=10**6)
    result_contents = {r["content"] for r in results}
    assert "live memory" in result_contents
    assert "archived memory" not in result_contents
