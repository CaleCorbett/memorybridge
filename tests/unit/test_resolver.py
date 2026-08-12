"""Regression tests for issue #181 (P2 cost-efficiency bundle) resolver fixes.

Covers:
  P2-2  dead prompt-caching cache_control block removed from _resolve_one's
        system param (SYSTEM_PROMPT is ~200 tokens, well under Anthropic's
        1024-token minimum cacheable prefix, so cache_control was a no-op).
  P2-3  RESOLVER_MODEL defaults to Haiku (not Sonnet), and the 404 fallback
        tier tracks whichever tier is configured instead of hardcoding
        "sonnet" (which would silently re-introduce the 3x-more-expensive
        tier the fix is trying to get off of).
"""
import importlib
import os
import sys
import types

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")


def _reload_resolver(monkeypatch=None, resolver_model=None):
    """Reload ingestion.resolver fresh so module-level constants
    (RESOLVER_MODEL, _FALLBACK_TIER, _RESOLVED_MODEL) reflect current env."""
    if resolver_model is not None:
        os.environ["RESOLVER_MODEL"] = resolver_model
    elif "RESOLVER_MODEL" in os.environ:
        del os.environ["RESOLVER_MODEL"]
    sys.modules.pop("ingestion.resolver", None)
    sys.modules.pop("resolver", None)
    import ingestion.resolver as resolver  # noqa: E402
    importlib.reload(resolver)
    return resolver


# --------------------------------------------------------------------- P2-3
def test_resolver_model_defaults_to_haiku():
    resolver = _reload_resolver(resolver_model=None)
    assert "haiku" in resolver.RESOLVER_MODEL.lower()
    assert "sonnet" not in resolver.RESOLVER_MODEL.lower()


def test_fallback_tier_tracks_configured_haiku_model():
    resolver = _reload_resolver(resolver_model="claude-haiku-4-5")
    assert resolver._FALLBACK_TIER == "haiku"


def test_fallback_tier_tracks_configured_sonnet_model():
    # If an operator explicitly overrides back to a Sonnet model, the 404
    # fallback must search for Sonnet, not silently hardcode Haiku either —
    # the fallback tier always matches whatever was configured.
    resolver = _reload_resolver(resolver_model="claude-sonnet-4-5")
    assert resolver._FALLBACK_TIER == "sonnet"


def test_pick_model_404_fallback_searches_configured_tier_not_hardcoded_sonnet():
    resolver = _reload_resolver(resolver_model="claude-haiku-4-5")

    class FakeNotFound(Exception):
        pass

    # Patch the real anthropic.NotFoundError reference used in except clause
    resolver.anthropic.NotFoundError = FakeNotFound

    class FakeModel:
        def __init__(self, id_, created_at):
            self.id = id_
            self.created_at = created_at

    class FakeModelsList:
        data = [
            FakeModel("claude-sonnet-4-5", "2026-01-01"),
            FakeModel("claude-haiku-4-6", "2026-02-01"),
            FakeModel("claude-haiku-4-5", "2026-01-01"),
        ]

    class FakeModels:
        def list(self, limit=50):
            return FakeModelsList()

    class FakeClient:
        models = FakeModels()

        def _ping(self, *a, **k):
            raise FakeNotFound("model not found")

    fake_client = FakeClient()
    fake_client.messages = types.SimpleNamespace(create=fake_client._ping)

    chosen = resolver._pick_model(fake_client)
    # Must fall back to the newest Haiku model, never Sonnet, since the
    # configured tier was Haiku.
    assert "haiku" in chosen.lower()
    assert chosen == "claude-haiku-4-6"


# --------------------------------------------------------------------- P2-2
def test_resolve_one_system_param_is_plain_string_no_cache_control():
    """The dead cache_control block must be gone: system= should be a plain
    string, not a content-block list with cache_control (which was a no-op
    since SYSTEM_PROMPT is far under the 1024-token cacheable minimum)."""
    resolver = _reload_resolver(resolver_model="claude-haiku-4-5")

    captured = {}

    class FakeMessage:
        content = [types.SimpleNamespace(text='{"verdict": "reject", "merged_fact": null}')]

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage()

    class FakeClient:
        messages = FakeMessages()

    resolver._RESOLVED_MODEL = "claude-haiku-4-5"  # skip _pick_model's network ping
    result = resolver._resolve_one(FakeClient(), {"fact": "test fact", "confidence": 0.9})

    assert isinstance(captured["system"], str)
    assert captured["system"] == resolver.SYSTEM_PROMPT
    assert result["verdict"] == "reject"


def test_no_functional_cache_control_block_in_resolve_one_source():
    """Belt-and-suspenders: the source may still explain in a comment why
    cache_control was removed, but must not contain a live cache_control=
    dict key/kwarg (the actual no-op code path) anywhere in the function."""
    import inspect
    resolver = _reload_resolver(resolver_model="claude-haiku-4-5")
    src = inspect.getsource(resolver._resolve_one)
    assert '"cache_control":' not in src
    assert "'cache_control':" not in src
