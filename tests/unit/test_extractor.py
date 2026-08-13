"""Regression tests for issue #181 (P2 cost-efficiency bundle) extractor fixes.

Covers:
  P2-4a  _call_deepseek uses the cheaper deepseek-chat model (not
         deepseek-reasoner), overridable via DEEPSEEK_MODEL.
  P2-4b  Noise screening runs on the INPUT (whole conversations that are
         100% ops telemetry) before the paid extraction call, not only on
         the model's output — while conversations mixing noise with a real
         fact still go through extraction untouched.
"""
import os

os.environ.setdefault("MEMORYBRIDGE_NO_EMBED", "1")

import ingestion.extractor as extractor  # noqa: E402


# --------------------------------------------------------------------- P2-4a
def test_call_deepseek_uses_deepseek_chat_by_default(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    captured = {}

    class FakeChoice:
        class message:
            content = "[]"

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChatNS:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChatNS()

    extractor._call_deepseek(FakeClient(), [{"id": "c1", "messages": []}])
    assert captured["model"] == "deepseek-chat"
    assert captured["model"] != "deepseek-reasoner"


def test_call_deepseek_respects_deepseek_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    captured = {}

    class FakeChoice:
        class message:
            content = "[]"

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChatNS:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChatNS()

    extractor._call_deepseek(FakeClient(), [{"id": "c1", "messages": []}])
    assert captured["model"] == "deepseek-reasoner"


# --------------------------------------------------------------------- P2-4b
def test_all_noise_conversation_detected():
    conv = {
        "id": "c1",
        "messages": [
            {"role": "user", "content": "what's the disk usage on the box?"},
            {"role": "assistant", "content": "disk usage was 47% this morning, no action needed"},
        ],
    }
    assert extractor._is_all_noise_conversation(conv) is True


def test_mixed_noise_and_real_fact_conversation_not_skipped():
    conv = {
        "id": "c2",
        "messages": [
            {"role": "assistant", "content": "cron job ran successfully, no new commits"},
            {"role": "user", "content": "I just took a new job as VP of Engineering at Acme Corp"},
        ],
    }
    assert extractor._is_all_noise_conversation(conv) is False


def test_empty_conversation_not_treated_as_noise():
    conv = {"id": "c3", "messages": []}
    assert extractor._is_all_noise_conversation(conv) is False


def test_extract_skips_noise_only_conversation_without_api_call(monkeypatch):
    calls = []

    def fake_call_deepseek(client, batch):
        calls.append(batch)
        return []

    monkeypatch.setattr(extractor, "_call_deepseek", fake_call_deepseek)
    monkeypatch.setattr(extractor, "_get_client", object)

    normalized = {
        "conversations": [
            {
                "id": "noise-1",
                "messages": [{"role": "assistant", "content": "load average is 2.1, uptime 14 days"}],
            },
        ]
    }
    facts, processed = extractor.extract(normalized)
    assert facts == []
    # The noise-only conversation must never reach _call_deepseek.
    assert calls == []
    # But it's still recorded as processed, so the idempotency ledger
    # doesn't re-scan it on every future run.
    assert len(processed) == 1
    assert processed[0]["id"] == "noise-1"


def test_extract_still_calls_api_for_real_conversation(monkeypatch):
    calls = []

    def fake_call_deepseek(client, batch):
        calls.append(batch)
        return [{"fact": "User's manager is Sarah Chen", "category": "fact",
                  "importance": "medium", "confidence": 0.9}]

    monkeypatch.setattr(extractor, "_call_deepseek", fake_call_deepseek)
    monkeypatch.setattr(extractor, "_get_client", object)

    normalized = {
        "conversations": [
            {
                "id": "real-1",
                "messages": [{"role": "user", "content": "My manager is Sarah Chen"}],
            },
        ]
    }
    facts, processed = extractor.extract(normalized)
    assert len(calls) == 1
    assert len(facts) == 1
    assert facts[0]["fact"] == "User's manager is Sarah Chen"
    assert len(processed) == 1
