"""
Integration Simulation: Claude (stdio) and Hermes (remote multi-agent) workflow.
Mocks real-world multi-session memory recall, procedural promotion, graph edge links,
and safety guardrails.
"""
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from db.store import MemoryStore, GuardrailRejection
import server


@pytest.fixture
def memory_env(tmp_path, monkeypatch):
    """Setup a isolated MemoryBridge environment for Claude & Hermes simulation."""
    db_file = tmp_path / "sim_memory.db"
    store = MemoryStore(db_file)
    monkeypatch.setattr(server, "_store", store)
    store.ensure_profile("default")
    return store


def _call_tool(tool_func, **kwargs):
    """Helper to invoke a FastMCP tool or naked function cleanly."""
    fn = getattr(tool_func, "fn", tool_func)
    return fn(**kwargs)


def test_claude_session_lifecycle_simulation(memory_env):
    """Simulates Claude in Session 1 making a mistake, learning a procedural rule,
    saving it to MemoryBridge, and in Session 2 recalling and following the rule."""

    # -------------------------------------------------------------------------
    # SESSION 1: Claude encounters build issue and consolidates procedural memory
    # -------------------------------------------------------------------------
    
    # 1. Claude starts by checking memory for existing deployment rules
    search_json = _call_tool(server.search_memory, query="billing app deployment", category="procedural")
    search_res = json.loads(search_json)
    assert len(search_res["results"]) == 0  # No prior knowledge

    # 2. Claude works on task, encounters `pip install` failure, fixes with `uv sync`
    raw_session_notes = """
    - Fix build issue by using command uv sync for dependencies instead of direct pip install.
    - Step 2: Database migrations must run prior to starting the API process.
    """
    
    # 3. Claude promotes the lesson to MemoryBridge via consolidate_session
    consolidate_json = _call_tool(server.consolidate_session, session_notes=raw_session_notes, profile="default", project_id="billing-app")
    consolidate_res = json.loads(consolidate_json)
    
    assert consolidate_res["status"] == "consolidated"
    assert consolidate_res["added_count"] >= 1
    
    promoted_memories = consolidate_res["promoted_memories"]
    proc_mem = next(m for m in promoted_memories if m["category"] == "procedural")
    proc_id = proc_mem["memory_id"]
    
    # 4. Claude links this procedural rule to the main project entity memory
    proj_mem_id = memory_env.add_memory("default", "Billing Application Core Service", category="fact", project_id="billing-app")
    edge_json = _call_tool(server.add_memory_edge, source_id=proc_id, target_id=proj_mem_id, relation="procedural_rule_for")
    edge_res = json.loads(edge_json)
    assert edge_res["status"] == "edge_created"

    # -------------------------------------------------------------------------
    # SESSION 2: New session (empty context window). Claude recalls procedural rule
    # -------------------------------------------------------------------------

    # 1. Claude starts new task: "Deploy the billing app"
    recall_json = _call_tool(server.get_memory, profile="default", context_hint="billing app deployment", min_confidence=0.8)
    recall_res = json.loads(recall_json)
    
    recalled_contents = [m["content"] for m in recall_res["memories"]]
    assert any("uv sync" in c for c in recalled_contents)

    # 2. Claude inspects connected knowledge graph edges
    edges_json = _call_tool(server.get_memory_edges, memory_id=proc_id)
    edges_res = json.loads(edges_json)
    assert edges_res["count"] == 1
    assert edges_res["edges"][0]["relation"] == "procedural_rule_for"


def test_hermes_cross_agent_collaboration_simulation(memory_env, monkeypatch):
    """Simulates Hermes (remote client) storing operational constraints,
    which Claude (stdio agent) reads and respects in a later session."""

    # -------------------------------------------------------------------------
    # HERMES AGENT SESSION (Remote Client)
    # -------------------------------------------------------------------------
    monkeypatch.setattr(server, "_REMOTE_MODE", True)  # Simulate HTTP bridge connection

    # 1. Hermes adds an operational DB constraint memory
    hermes_add_json = _call_tool(
        server.add_memory,
        content="Production DB connection requires SSL certificate validation",
        category="constraint",
        importance="critical",
        client_name="hermes",
        confidence=0.98
    )
    hermes_res = json.loads(hermes_add_json)
    assert hermes_res["status"] == "added"
    hermes_mem_id = hermes_res["memory_id"]

    # -------------------------------------------------------------------------
    # CLAUDE AGENT SESSION (Local Stdio Agent)
    # -------------------------------------------------------------------------
    monkeypatch.setattr(server, "_REMOTE_MODE", False)  # Back to local stdio

    # 1. Claude searches memory for DB connection constraints
    claude_search_json = _call_tool(server.search_memory, query="Production DB connection", category="constraint")
    claude_search_res = json.loads(claude_search_json)
    
    assert len(claude_search_res["results"]) == 1
    found_mem = claude_search_res["results"][0]
    assert found_mem["id"] == hermes_mem_id
    assert found_mem["client_name"] == "hermes"
    assert found_mem["source"] == "remote"
    assert "SSL certificate validation" in found_mem["content"]


def test_safety_and_ttl_expiration_simulation(memory_env):
    """Simulates secret redaction blocking and automatic TTL expiration."""

    # 1. Secret leakage attempt: Agent tries to store DB password
    secret_json = _call_tool(
        server.add_memory,
        content="Database credentials: postgres://admin:SuperSecretPass123@db.internal:5432/prod",
        category="fact"
    )
    secret_res = json.loads(secret_json)
    assert secret_res["status"] == "rejected"
    assert "security check failed" in secret_res["reason"]

    # 2. Temporary fact with expired TTL
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    ttl_json = _call_tool(
        server.add_memory,
        content="Temporary staging feature flag enabled for sprint test",
        category="fact",
        expires_at=yesterday
    )
    ttl_res = json.loads(ttl_json)
    assert ttl_res["status"] == "added"
    expired_id = ttl_res["memory_id"]

    # 3. Verify get_memory excludes expired TTL memory
    recall_json = _call_tool(server.get_memory, profile="default")
    recall_res = json.loads(recall_json)
    recalled_ids = [m["id"] for m in recall_res["memories"]]
    assert expired_id not in recalled_ids
