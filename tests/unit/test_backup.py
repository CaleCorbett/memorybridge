"""Tests for db.backup — VACUUM INTO backup, verify, rotate (issue #182)."""

import sqlite3
from pathlib import Path

import pytest

from db.backup import (
    BACKUP_PREFIX,
    BACKUP_SUFFIX,
    BackupResult,
    create_backup,
    list_backups,
    verify_backup,
)


def _seed_db(db_path: Path, n_memories: int = 5) -> Path:
    """Create a minimal MemoryBridge-shaped DB with *n_memories* rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            name TEXT PRIMARY KEY
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id         TEXT PRIMARY KEY,
            profile    TEXT NOT NULL,
            content    TEXT NOT NULL,
            category   TEXT NOT NULL DEFAULT 'general',
            importance TEXT NOT NULL DEFAULT 'medium',
            archived   INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("INSERT OR IGNORE INTO profiles VALUES ('default')")
    for i in range(n_memories):
        conn.execute(
            "INSERT INTO memories (id, profile, content, category, importance, archived, created_at, updated_at) "
            "VALUES (?, 'default', ?, 'general', 'medium', 0, datetime('now'), datetime('now'))",
            (f"mem_{i:04d}", f"Test memory #{i}"),
        )
    conn.commit()
    conn.close()
    return db_path


class TestCreateBackup:
    def test_produces_valid_db(self, tmp_path):
        db_path = _seed_db(tmp_path / "memory.db")
        backup_dir = tmp_path / "backups"

        result = create_backup(db_path, backup_dir)

        assert result.integrity_ok
        assert result.path.exists()
        assert result.size_bytes > 0

    def test_backup_contains_memories(self, tmp_path):
        n = 7
        db_path = _seed_db(tmp_path / "memory.db", n_memories=n)
        backup_dir = tmp_path / "backups"

        result = create_backup(db_path, backup_dir)

        assert result.memory_count == n

    def test_backup_has_no_sidecars(self, tmp_path):
        db_path = _seed_db(tmp_path / "memory.db")
        backup_dir = tmp_path / "backups"

        result = create_backup(db_path, backup_dir)

        # VACUUM INTO should produce exactly one file — no sidecars.
        parent = result.path.parent
        siblings = list(parent.iterdir())
        assert len(siblings) == 1, f"Expected only backup file, got: {[s.name for s in siblings]}"
        assert siblings[0] == result.path

    def test_backup_is_independently_openable(self, tmp_path):
        db_path = _seed_db(tmp_path / "memory.db", n_memories=3)
        backup_dir = tmp_path / "backups"

        result = create_backup(db_path, backup_dir)

        # Open with a completely fresh connection — no WAL dependency.
        conn = sqlite3.connect(str(result.path))
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        assert count == 3


class TestRotation:
    def test_deletes_oldest_when_over_limit(self, tmp_path):
        import time

        db_path = _seed_db(tmp_path / "memory.db")
        backup_dir = tmp_path / "backups"

        names = []
        for _ in range(4):
            result = create_backup(db_path, backup_dir, max_backups=3)
            names.append(result.path.name)
            time.sleep(1.1)  # ensure distinct timestamps

        remaining = sorted(p.name for p in backup_dir.iterdir())
        assert len(remaining) == 3
        # Oldest (first created) should have been rotated away.
        assert names[0] not in remaining

    def test_keeps_all_when_under_limit(self, tmp_path):
        import time

        db_path = _seed_db(tmp_path / "memory.db")
        backup_dir = tmp_path / "backups"

        for _ in range(2):
            create_backup(db_path, backup_dir, max_backups=5)
            time.sleep(1.1)

        remaining = list(backup_dir.iterdir())
        assert len(remaining) == 2


class TestVerifyBackup:
    def test_valid_backup(self, tmp_path):
        db_path = _seed_db(tmp_path / "memory.db")
        backup_dir = tmp_path / "backups"
        result = create_backup(db_path, backup_dir)

        verified = verify_backup(result.path)

        assert verified.integrity_ok
        assert verified.memory_count >= 0

    def test_detects_corruption(self, tmp_path):
        corrupt_path = tmp_path / f"{BACKUP_PREFIX}20260101_000000{BACKUP_SUFFIX}"
        corrupt_path.write_bytes(b"this is not a sqlite database at all")

        result = verify_backup(corrupt_path)

        assert not result.integrity_ok

    def test_missing_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.db"

        result = verify_backup(missing)

        assert not result.integrity_ok
        assert result.size_bytes == 0


class TestListBackups:
    def test_empty_dir(self, tmp_path):
        assert list_backups(tmp_path / "nonexistent") == []

    def test_sorted_oldest_first(self, tmp_path):
        import time

        db_path = _seed_db(tmp_path / "memory.db")
        backup_dir = tmp_path / "backups"

        for _ in range(3):
            create_backup(db_path, backup_dir, max_backups=10)
            time.sleep(1.1)

        results = list_backups(backup_dir)
        assert len(results) == 3
        # Sorted oldest first.
        names = [r.path.name for r in results]
        assert names == sorted(names)
