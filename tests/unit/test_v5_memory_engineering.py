import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from db.constants import guardrail_check, VALID_CATEGORIES
from db.store import MemoryStore, GuardrailRejection
import server


def test_v5_categories():
    assert "procedural" in VALID_CATEGORIES
    assert "episodic" in VALID_CATEGORIES


def test_secrets_guardrail():
    # OpenAI key
    ok, reason = guardrail_check("Remember my key sk-abcdef12345678901234567890")
    assert not ok
    assert "security check failed" in reason

    # GitHub token
    ok, reason = guardrail_check("Token ghp_123456789012345678901234567890123456")
    assert not ok
    assert "security check failed" in reason

    # DB connection string
    ok, reason = guardrail_check("Database url postgres://user:password@localhost:5432/dbname")
    assert not ok
    assert "security check failed" in reason

    # Normal text
    ok, reason = guardrail_check("This repository requires uv for package installation.")
    assert ok
    assert reason == ""


def test_v5_store_confidence_and_ttl(tmp_path):
    db_file = tmp_path / "memory.db"
    store = MemoryStore(db_file)
    store.ensure_profile("default")

    # Add active memory with confidence 0.95
    m1 = store.add_memory("default", "Run database migrations before API start", category="procedural", confidence=0.95)
    assert m1 is not None

    # Add low-confidence memory
    m2 = store.add_memory("default", "Staging image might be using python 3.9", category="fact", confidence=0.4)
    assert m2 is not None

    # Add expired memory (TTL)
    past_iso = (datetime.now() - timedelta(days=1)).isoformat()
    m3 = store.add_memory("default", "Temporary debug flag active", category="fact", expires_at=past_iso)
    assert m3 is not None

    # Test get_memories with min_confidence
    mems = store.get_memories("default", min_confidence=0.8)
    mem_ids = [m["id"] for m in mems]
    assert m1 in mem_ids
    assert m2 not in mem_ids  # filtered out by min_confidence
    assert m3 not in mem_ids  # filtered out by TTL expired

    # Test search_hybrid with min_confidence
    search_res = store.search_hybrid("default", "database migrations", min_confidence=0.8)
    search_ids = [m["id"] for m in search_res]
    assert m1 in search_ids


def test_v5_knowledge_graph_edges(tmp_path):
    db_file = tmp_path / "memory.db"
    store = MemoryStore(db_file)
    store.ensure_profile("default")

    m1 = store.add_memory("default", "Billing feature flag enabled", category="fact")
    m2 = store.add_memory("default", "Sarah approves billing refunds", category="procedural")

    edge_id = store.add_edge(m1, m2, relation="depends_on")
    assert edge_id.startswith("edge_")

    edges = store.get_edges(m1)
    assert len(edges) == 1
    assert edges[0]["relation"] == "depends_on"
    assert edges[0]["source_id"] == m1
    assert edges[0]["target_id"] == m2

    deleted = store.delete_edge(edge_id)
    assert deleted
    assert len(store.get_edges(m1)) == 0


def test_server_consolidate_session(tmp_path, monkeypatch):
    db_file = tmp_path / "memory.db"
    store = MemoryStore(db_file)
    monkeypatch.setattr(server, "_store", store)

    session_log = """
    - Fixed staging build by switching to uv sync command.
    - Verified that database migrations must run prior to API initialization.
    """
    fn = getattr(server.consolidate_session, "fn", server.consolidate_session)
    res_raw = fn(session_log, profile="default", project_id="billing-app")
    res = json.loads(res_raw)

    assert res["status"] == "consolidated"
    assert res["added_count"] >= 1
    assert len(res["promoted_memories"]) >= 1


def test_server_graph_edge_tools(tmp_path, monkeypatch):
    db_file = tmp_path / "memory.db"
    store = MemoryStore(db_file)
    monkeypatch.setattr(server, "_store", store)

    m1 = store.add_memory("default", "Fact A", category="fact")
    m2 = store.add_memory("default", "Fact B", category="fact")

    add_fn = getattr(server.add_memory_edge, "fn", server.add_memory_edge)
    get_fn = getattr(server.get_memory_edges, "fn", server.get_memory_edges)

    add_res = json.loads(add_fn(m1, m2, relation="relates_to"))
    assert add_res["status"] == "edge_created"

    get_res = json.loads(get_fn(m1))
    assert get_res["count"] == 1
    assert get_res["edges"][0]["relation"] == "relates_to"
