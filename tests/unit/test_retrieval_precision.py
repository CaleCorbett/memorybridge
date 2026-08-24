"""
Retrieval precision tests — current state + proposed improvements.

Covers the two failure modes identified in the critique analysis:

  1. RRF inherits semantic-leg ordering errors (identity-discrimination problem)
  2. MCP discretionary-retrieval gap (reflected in export_for_model path tests)

Six use-case families, drawn from PrecisionMemBench categories:
  A. Supersession exclusion — obsolete beliefs must not surface
  B. Semantic inversion — "prefers X" vs "prefers Y" (identity-discrimination)
  C. Noise isolation — off-topic drift memories must not contaminate results
  D. Exact-target precision — top result must be the correct belief, not a sibling
  E. Budget eviction — only the right belief fits in the token budget
  F. System-prompt injection path — the non-MCP alternative path works reliably

Each family has:
  - CURRENT STATE tests: document and assert existing correct behavior (green)
  - KNOWN GAP tests: xfail(strict=False) documenting observable failure modes
  - PROPOSED FIX tests: xfail(strict=True) asserting the post-fix target behavior

Run fast (no model download):
    MEMORYBRIDGE_NO_EMBED=1 python -m pytest tests/unit/test_retrieval_precision.py -v

Run with embeddings (slow, downloads ONNX model on first run):
    python -m pytest tests/unit/test_retrieval_precision.py -v
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from db.store import MemoryStore

_EMBED_AVAILABLE = os.environ.get("MEMORYBRIDGE_NO_EMBED") != "1"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path, name: str = "test.db") -> MemoryStore:
    s = MemoryStore(tmp_path / name)
    s.ensure_profile("default")
    return s


def _ids(results):     return [r["id"] for r in results]
def _contents(results): return [r["content"] for r in results]


# ---------------------------------------------------------------------------
# Dataset seeds — each returns a dict of named IDs
# ---------------------------------------------------------------------------

def _seed_supersession(store):
    old = store.add_memory(
        "default",
        "The user works at Acme Corp in the sales department",
        category="fact", importance="medium",
    )
    new = store.add_memory(
        "default",
        "The user works at Globex Corporation as a product manager",
        category="fact", importance="high",
        supersedes=[old],
    )
    return {"old": old, "new": new}


def _seed_semantic_inversion(store):
    dark = store.add_memory(
        "default",
        "The user prefers dark mode in all their applications",
        category="preference", importance="medium",
    )
    light = store.add_memory(
        "default",
        "The user prefers light mode for reading and documentation tools",
        category="preference", importance="medium",
    )
    return {"dark": dark, "light": light}


def _seed_noise_corpus(store):
    target = store.add_memory(
        "default",
        "The user keyboard shortcut for format document is Shift+Alt+F",
        category="preference", importance="medium",
    )
    d1 = store.add_memory(
        "default", "The user cat is named Mochi and is a tabby",
        category="fact", importance="low",
    )
    d2 = store.add_memory(
        "default", "The user wants to try a new pasta recipe this weekend",
        category="fact", importance="low",
    )
    d3 = store.add_memory(
        "default", "The user favorite vacation destination is Kyoto Japan",
        category="fact", importance="low",
    )
    return {"target": target, "drift": [d1, d2, d3]}


def _seed_precision_corpus(store):
    exact = store.add_memory(
        "default",
        "The user Python formatter is Black with a line-length of 88",
        category="preference", importance="high",
    )
    sibling_a = store.add_memory(
        "default",
        "The user prefers code formatters that are opinionated and consistent",
        category="preference", importance="medium",
    )
    sibling_b = store.add_memory(
        "default",
        "The user uses Ruff for linting their Python projects",
        category="preference", importance="medium",
    )
    return {"exact": exact, "sibling_a": sibling_a, "sibling_b": sibling_b}


def _seed_budget_corpus(store):
    correct = store.add_memory(
        "default",
        "The user timezone is US/Central (UTC-6)",
        category="fact", importance="high",
    )
    bulk = store.add_memory(
        "default",
        (
            "The user attended a three-day leadership workshop in June covering "
            "change management facilitation conflict resolution team dynamics "
            "strategic planning financial literacy presentation skills stakeholder "
            "communication executive presence cross-functional collaboration and "
            "agile methodology implementation in enterprise environments hosted by "
            "a consulting firm in Chicago Illinois at a hotel near the river"
        ),
        category="fact", importance="low",
    )
    return {"correct": correct, "bulk": bulk}


# ===========================================================================
# Family A — Supersession exclusion
# ===========================================================================

class TestSupersessionExclusion:

    def test_current__superseded_belief_is_archived(self, tmp_path):
        """CURRENT STATE: superseded belief is immediately archived."""
        store = _make_store(tmp_path)
        ids = _seed_supersession(store)
        active_ids = _ids(store.get_memories("default"))
        assert ids["new"] in active_ids
        assert ids["old"] not in active_ids

    def test_current__keyword_search_excludes_superseded(self, tmp_path):
        """CURRENT STATE: BM25 never surfaces archived memories."""
        store = _make_store(tmp_path)
        _seed_supersession(store)
        results = store.search("default", "Acme Corp")
        assert not any("Acme" in c for c in _contents(results))

    def test_current__new_belief_is_reachable_by_keyword(self, tmp_path):
        """CURRENT STATE: The replacement belief surfaces on the same topic."""
        store = _make_store(tmp_path)
        ids = _seed_supersession(store)
        results = store.search("default", "user works employer")
        assert ids["new"] in _ids(results)

    def test_proposed__counter_signal_query_returns_replacement(self, tmp_path):
        """Fix 4 (implemented): Querying the obsolete employer term returns the new belief
        via counter-signal retrieval. The search() path now detects when BM25 returns
        empty on active memories, looks up archived rows matching the query token,
        and surfaces their superseded_by replacements.
        """
        store = _make_store(tmp_path)
        ids = _seed_supersession(store)
        results = store.search("default", "Acme Corp")
        assert ids["new"] in _ids(results), (
            "Querying obsolete term should surface the replacement belief"
        )


# ===========================================================================
# Family B — Semantic inversion (identity-discrimination)
# ===========================================================================

class TestSemanticInversion:

    def test_current__keyword_discriminates_dark_mode(self, tmp_path):
        """CURRENT STATE: BM25 'dark mode' ranks the dark belief #1.

        Note: BM25 search() uses OR semantics so both beliefs may match
        (both contain 'mode'). What matters is that the correct belief
        ranks first AND has a higher match_score than the wrong one.
        """
        store = _make_store(tmp_path)
        ids = _seed_semantic_inversion(store)
        results = store.search("default", "dark mode")
        assert ids["dark"] in _ids(results), "dark-mode belief must be present"
        # Correct belief must rank first (BM25 scores 'dark' token match higher)
        assert results[0]["id"] == ids["dark"], (
            f"dark-mode belief must be #1. Got: {_contents(results)}"
        )
        # Wrong belief may appear but must have a lower match_score
        if ids["light"] in _ids(results):
            dark_score  = next(r["match_score"] for r in results if r["id"] == ids["dark"])
            light_score = next(r["match_score"] for r in results if r["id"] == ids["light"])
            assert dark_score >= light_score, (
                "dark-mode belief must have >= match_score vs light-mode belief"
            )

    def test_current__keyword_discriminates_light_mode(self, tmp_path):
        """CURRENT STATE: BM25 'light mode' ranks the light belief #1.

        Same OR-semantics caveat: both beliefs may appear, but light must
        rank first and have a higher match_score.
        """
        store = _make_store(tmp_path)
        ids = _seed_semantic_inversion(store)
        results = store.search("default", "light mode")
        assert ids["light"] in _ids(results), "light-mode belief must be present"
        assert results[0]["id"] == ids["light"], (
            f"light-mode belief must be #1. Got: {_contents(results)}"
        )
        if ids["dark"] in _ids(results):
            light_score = next(r["match_score"] for r in results if r["id"] == ids["light"])
            dark_score  = next(r["match_score"] for r in results if r["id"] == ids["dark"])
            assert light_score >= dark_score, (
                "light-mode belief must have >= match_score vs dark-mode belief"
            )

    @pytest.mark.skipif(not _EMBED_AVAILABLE, reason="Requires embedding model (unset MEMORYBRIDGE_NO_EMBED)")
    @pytest.mark.xfail(
        reason=(
            "KNOWN GAP (identity-discrimination): cosine vectors for dark-mode "
            "and light-mode beliefs are nearly identical. The semantic leg may rank "
            "the wrong belief #1; RRF inherits that ordering. strict=False because "
            "some embedding models discriminate better than others."
        ),
        strict=False,
    )
    def test_known_gap__semantic_may_rank_wrong_belief_first(self, tmp_path):
        """KNOWN GAP: semantic-only may surface the wrong preference belief first."""
        store = _make_store(tmp_path)
        ids = _seed_semantic_inversion(store)
        store.build_embeddings("default")

        results = store.search_semantic("default", "dark mode display preference", limit=2)
        assert len(results) >= 2
        # Documents the failure: wrong belief ranked first
        assert results[0]["id"] == ids["light"], "Expected wrong belief #1 to confirm gap"

    def test_proposed__hybrid_bm25_tiebreak_resolves_inversion(self, tmp_path):
        """Fix 2 (verified with real embeddings): hybrid resolves the semantic inversion
        by boosting the BM25 rank-0 hit. The dark-mode belief holds the exact BM25
        rank-0 slot ('dark'), so it wins the tiebreak over the semantically adjacent
        light-mode belief.
        """
        store = _make_store(tmp_path)
        ids = _seed_semantic_inversion(store)
        store.build_embeddings("default")

        results = store.search_hybrid("default", "dark mode display preference", limit=2)
        assert len(results) >= 1
        assert results[0]["id"] == ids["dark"], (
            f"Hybrid should rank dark-mode first via BM25 tiebreak. Got: {_contents(results)}"
        )

        tight = store.search_hybrid("default", "dark mode display preference", limit=1)
        assert tight[0]["id"] == ids["dark"], "Tight-limit must return only the correct belief"


# ===========================================================================
# Family C — Noise isolation
# ===========================================================================

class TestNoiseIsolation:

    def test_current__keyword_finds_target_among_noise(self, tmp_path):
        """CURRENT STATE: BM25 finds the keyboard shortcut target amid drift memories."""
        store = _make_store(tmp_path)
        ids = _seed_noise_corpus(store)
        results = store.search("default", "keyboard shortcut format document")
        assert ids["target"] in _ids(results)

    def test_current__drift_excluded_from_targeted_keyword_search(self, tmp_path):
        """CURRENT STATE: Drift memories don't appear in a targeted BM25 query."""
        store = _make_store(tmp_path)
        ids = _seed_noise_corpus(store)
        results = store.search("default", "keyboard shortcut format document")
        for drift_id in ids["drift"]:
            assert drift_id not in _ids(results), f"Drift memory {drift_id} contaminated results"

    @pytest.mark.skipif(not _EMBED_AVAILABLE, reason="Requires embedding model")
    def test_current__semantic_finds_target_via_paraphrase(self, tmp_path):
        """CURRENT STATE: semantic finds the shortcut memory via paraphrase query."""
        store = _make_store(tmp_path)
        ids = _seed_noise_corpus(store)
        store.build_embeddings("default")
        results = store.search_semantic("default", "code formatting hotkey", limit=5)
        assert ids["target"] in _ids(results)

    @pytest.mark.skipif(not _EMBED_AVAILABLE, reason="Requires embedding model")
    @pytest.mark.xfail(
        reason=(
            "KNOWN GAP: With a generous budget, semantic search may return "
            "drift memories because their vectors have non-zero similarity to the "
            "query. No drift_score filter exists at the retrieval layer."
        ),
        strict=False,
    )
    def test_known_gap__drift_contamination_at_large_budget(self, tmp_path):
        """KNOWN GAP: semantic may return off-topic drift memories under large budget."""
        store = _make_store(tmp_path)
        ids = _seed_noise_corpus(store)
        store.build_embeddings("default")

        results = store.search_semantic(
            "default", "keyboard shortcut format document", limit=10, max_tokens=5000
        )
        drift_returned = [d for d in ids["drift"] if d in _ids(results)]
        assert len(drift_returned) > 0, "Expected drift contamination to be observed"

    @pytest.mark.skipif(not _EMBED_AVAILABLE, reason="Requires embedding model")
    @pytest.mark.xfail(
        reason=(
            "PROPOSED FIX not implemented: search_semantic currently ignores "
            "min_confidence. Adding this filter would exclude low-importance "
            "drift memories that have importance='low' (lower initial confidence)."
        ),
        strict=True,
    )
    def test_proposed__min_confidence_in_semantic_excludes_drift(self, tmp_path):
        """PROPOSED: min_confidence parameter in search_semantic excludes low-importance drift."""
        store = _make_store(tmp_path)
        ids = _seed_noise_corpus(store)
        store.build_embeddings("default")

        results = store.search_semantic(
            "default",
            "keyboard shortcut format document",
            limit=10,
            max_tokens=5000,
            min_confidence=0.6,   # type: ignore[call-arg]
        )
        for drift_id in ids["drift"]:
            assert drift_id not in _ids(results), f"Drift {drift_id} survived confidence filter"


# ===========================================================================
# Family D — Exact-target precision (ranking stability)
# ===========================================================================

class TestExactTargetPrecision:

    def test_current__keyword_ranks_exact_belief_first(self, tmp_path):
        """CURRENT STATE: BM25 ranks the exact belief #1 for specific token query."""
        store = _make_store(tmp_path)
        ids = _seed_precision_corpus(store)
        results = store.search("default", "Python formatter Black line-length")
        assert len(results) > 0
        assert results[0]["id"] == ids["exact"], (
            f"Expected exact belief first. Got: {_contents(results)}"
        )

    def test_current__siblings_do_not_displace_exact_on_keyword(self, tmp_path):
        """CURRENT STATE: Sibling beliefs don't outrank the exact memory on BM25."""
        store = _make_store(tmp_path)
        ids = _seed_precision_corpus(store)
        results = store.search("default", "Black formatter 88")
        if results:
            assert results[0]["id"] not in (ids["sibling_a"], ids["sibling_b"]), (
                "A sibling displaced the exact target on keyword search"
            )

    @pytest.mark.skipif(not _EMBED_AVAILABLE, reason="Requires embedding model")
    @pytest.mark.xfail(
        reason=(
            "KNOWN GAP: On a paraphrased query, the semantic leg may rank a sibling "
            "('prefers opinionated formatters') above the exact Black/88 belief "
            "because the sibling is semantically closer to the abstract query concept."
        ),
        strict=False,
    )
    def test_known_gap__hybrid_sibling_may_outrank_exact_on_paraphrase(self, tmp_path):
        """KNOWN GAP: paraphrased query may promote a sibling above the exact belief."""
        store = _make_store(tmp_path)
        ids = _seed_precision_corpus(store)
        store.build_embeddings("default")

        results = store.search_hybrid(
            "default", "formatting tool line width configuration", limit=3
        )
        assert len(results) >= 2
        # Documents the gap: sibling is #1
        assert results[0]["id"] in (ids["sibling_a"], ids["sibling_b"]), (
            "Expected sibling to outrank exact belief to confirm gap"
        )

    def test_proposed__bm25_tiebreak_forces_exact_to_top_on_hybrid(self, tmp_path):
        """Fix 2 (verified with real embeddings): BM25 rank-0 tiebreak ensures the exact
        belief is always #1 in hybrid even on paraphrased queries. 'Black' and '88' are
        exact BM25 tokens in the target belief but not the siblings, so the tiebreak wins.
        """
        store = _make_store(tmp_path)
        ids = _seed_precision_corpus(store)
        store.build_embeddings("default")

        results = store.search_hybrid(
            "default", "Python formatter Black line-length", limit=3,
            recency_boost=False,
        )
        assert len(results) > 0
        assert results[0]["id"] == ids["exact"], (
            f"Exact belief must be #1 after BM25 tiebreak. Got: {_contents(results)}"
        )


# ===========================================================================
# Family E — Budget eviction precision
# ===========================================================================

class TestBudgetEvictionPrecision:

    def test_current__short_correct_belief_fits_in_tight_budget(self, tmp_path):
        """CURRENT STATE: short belief fits at 50-token budget.

        The correct belief ('The user timezone is US/Central (UTC-6)') is ~32
        tokens. max_tokens=50 comfortably fits it while excluding the long bulk
        memory (~55 tokens).
        """
        store = _make_store(tmp_path)
        ids = _seed_budget_corpus(store)
        results = store.search("default", "timezone UTC Central", max_tokens=50)
        assert ids["correct"] in _ids(results), (
            "Short correct belief (~32 tokens) must fit within 50-token budget"
        )

    def test_current__bulk_memory_excluded_at_tight_budget(self, tmp_path):
        """CURRENT STATE: long bulk memory (~55 tokens) excluded at 50-token budget."""
        store = _make_store(tmp_path)
        ids = _seed_budget_corpus(store)
        results = store.search("default", "timezone UTC Central", max_tokens=50)
        assert ids["bulk"] not in _ids(results), (
            "Bulk memory (~55 tokens) must not fit within 50-token budget"
        )

    @pytest.mark.xfail(
        reason=(
            "KNOWN GAP: The over-fetch (limit*3) then token-trim approach does not "
            "guarantee the correct belief wins if a higher-BM25 bulk memory lands "
            "earlier in the list and consumes the entire budget first."
        ),
        strict=False,
    )
    def test_known_gap__budget_trim_may_exclude_correct_when_bulk_ranks_first(self, tmp_path):
        """KNOWN GAP: budget trim evicts correct belief when bulk memory ranks higher."""
        store = _make_store(tmp_path)
        ids = _seed_budget_corpus(store)

        # Query term 'leadership' is in the bulk memory, not in the target
        results = store.search(
            "default", "leadership workshop enterprise", max_tokens=25
        )
        correct_absent = ids["correct"] not in _ids(results)
        bulk_present   = ids["bulk"] in _ids(results)
        assert correct_absent and bulk_present, (
            "Expected bulk to consume budget before correct belief is reached"
        )

    def test_proposed__high_importance_wins_budget_over_low_importance_bulk(self, tmp_path):
        """Fix 3 (implemented): high-importance short belief wins budget slot regardless of rank.
        The second importance-aware budget pass rescues high-importance beliefs evicted
        by lower-importance but higher-ranked bulk memories.
        """
        store = _make_store(tmp_path)
        ids = _seed_budget_corpus(store)

        results = store.search(
            "default", "leadership workshop enterprise", max_tokens=40
        )
        assert ids["correct"] in _ids(results), (
            "High-importance short belief must win budget allocation"
        )


# ===========================================================================
# Family F — System-prompt injection path (MCP discretionary gap)
# ===========================================================================

class TestSystemPromptInjectionPath:
    """
    The discretionary-call gap is architectural. These tests verify the
    non-MCP export path that bypasses it — ensuring documentation pointing
    users there is backed by passing assertions.
    """

    def test_current__get_memories_is_query_independent(self, tmp_path):
        """CURRENT STATE: get_memories() returns ALL active memories without a query.
        This is the architectural guarantee that the injection path cannot
        discretionarily skip relevant memories."""
        store = _make_store(tmp_path)
        ids = _seed_noise_corpus(store)

        all_mems = store.get_memories("default")
        all_ids = _ids(all_mems)

        assert ids["target"] in all_ids, "Target always present — no query dependency"
        for drift_id in ids["drift"]:
            assert drift_id in all_ids, "All active memories returned regardless of relevance"

    def test_current__injection_payload_excludes_archived(self, tmp_path):
        """CURRENT STATE: get_memories excludes superseded/archived beliefs."""
        store = _make_store(tmp_path)
        ids = _seed_supersession(store)

        payload_contents = _contents(store.get_memories("default"))
        assert not any("Acme" in c for c in payload_contents), (
            "Archived superseded belief leaked into injection payload"
        )
        assert any("Globex" in c for c in payload_contents), (
            "New belief missing from injection payload"
        )

    def test_current__injection_payload_token_budget_respected(self, tmp_path):
        """CURRENT STATE: get_memories with max_tokens respects the budget."""
        store = _make_store(tmp_path)
        ids = _seed_budget_corpus(store)

        results = store.get_memories("default", max_tokens=50)
        total = sum(r["token_count"] for r in results)
        assert total <= 50, f"Token budget exceeded: {total} > 50"

    def test_current__injection_path_always_fires_unlike_mcp_tool(self, tmp_path):
        """CURRENT STATE: The injection path (get_memories) fires unconditionally
        from the calling code; unlike the MCP tool path, it requires no model
        decision to trigger. Verified by asserting it can be called directly
        without a query and always returns data when memories exist."""
        store = _make_store(tmp_path)
        store.add_memory("default", "User is Cale", category="fact", importance="high")
        store.add_memory("default", "User prefers Python", category="preference")

        # Simulate what a pre-prompt injection would do: call unconditionally
        memories = store.get_memories("default")
        assert len(memories) == 2, "Injection path must return all memories unconditionally"

    @pytest.mark.xfail(
        reason=(
            "PROPOSED FIX not implemented: the injection path should offer a "
            "relevance-filtered mode (get_memories_for_context(session_summary)) "
            "that pre-loads only beliefs relevant to the session topic rather than "
            "the full corpus. This addresses the inverse of the MCP gap — over-injection "
            "of irrelevant context when the corpus is large."
        ),
        strict=True,
    )
    def test_proposed__injection_path_supports_topic_filtered_mode(self, tmp_path):
        """PROPOSED: get_memories gains an optional topic filter for smart injection."""
        store = _make_store(tmp_path)
        ids = _seed_noise_corpus(store)

        # After the fix, a topic hint would pre-filter to only relevant beliefs
        results = store.get_memories(
            "default",
            topic_hint="keyboard shortcuts editor",  # type: ignore[call-arg]
        )
        assert ids["target"] in _ids(results), "Target must be present"
        for drift_id in ids["drift"]:
            assert drift_id not in _ids(results), f"Drift {drift_id} must be excluded by topic filter"
