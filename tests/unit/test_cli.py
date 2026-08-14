"""Tests for the `mb` CLI (init path + argument parsing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import cli
import config


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYBRIDGE_DATA", str(tmp_path))
    monkeypatch.setenv("MEMORYBRIDGE_NO_EMBED", "1")
    config.reset_cache()
    yield
    config.reset_cache()


def test_init_creates_scaffold(tmp_path, capsys):
    rc = cli.main(["init"])
    assert rc == 0
    assert (tmp_path / ".env").exists()
    assert (tmp_path / "memorybridge.yaml").exists()
    assert (tmp_path / "memory.db").exists()
    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "logs").is_dir()

    # Store has a usable default profile.
    from db.store import MemoryStore
    store = MemoryStore(tmp_path / "memory.db")
    assert store.get_profile("default") is not None

    # Prints a Claude Desktop config snippet with the `mb serve` command.
    out = capsys.readouterr().out
    assert '"command": "mb"' in out
    assert '"serve"' in out


def test_init_is_idempotent(tmp_path):
    assert cli.main(["init"]) == 0
    # Second run must not clobber an existing .env / config.
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=secret\n")
    assert cli.main(["init"]) == 0
    assert "secret" in (tmp_path / ".env").read_text()


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_ingest_args_parse():
    args = cli.build_parser().parse_args(["ingest", "--source", "claude", "--file", "x.json"])
    assert args.source == "claude" and args.file == "x.json"


def test_status_no_db(tmp_path, capsys):
    rc = cli.main(["status"])
    assert rc == 1
    assert "Database not found" in capsys.readouterr().out


def test_status_empty_profile(tmp_path, capsys):
    assert cli.main(["init"]) == 0
    capsys.readouterr()  # discard init output
    rc = cli.main(["status", "--profile", "empty"])
    assert rc == 0
    assert "No active memories" in capsys.readouterr().out


def test_status_groups_and_trust_labels(tmp_path, capsys):
    assert cli.main(["init"]) == 0
    from db.store import MemoryStore
    store = MemoryStore(tmp_path / "memory.db")
    store.add_memory("default", "stdio-written fact", source="claude",
                     skip_enrichment=True)
    store.add_memory("default", "bridge-written fact", source="remote",
                     skip_enrichment=True)
    store.add_memory("default", "hermes-claimed fact", source="remote",
                     client_name="hermes", skip_enrichment=True)
    store.add_memory("default", "pre-provenance fact", skip_enrichment=True)
    capsys.readouterr()  # discard init output

    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out

    # One line per (source, client_name) identity with its honest trust label.
    assert "verified" in out          # source="claude"
    assert "unattributed" in out      # source="remote", no client_name
    assert "self-reported" in out     # client_name set — never shown as fact
    assert "hermes" in out
    assert "untracked" in out         # NULL source (pre-#180 rows)
    # Totals line reflects all four memories and the budget.
    assert "4 total" in out
    assert "of token budget" in out
    # Everything was written just now, so the 7-day delta shows on each row.
    assert "▲1" in out
