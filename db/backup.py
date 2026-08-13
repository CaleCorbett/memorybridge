"""VACUUM INTO-based backup for MemoryBridge (issue #182).

Produces a single, self-contained, WAL-consistent backup file with no
sidecars (-wal, -shm, -journal).  This solves every failure mode of the
old shutil.copy2 approach:

- No misnamed WAL sidecars (the .wal-vs--wal bug)
- No mid-transaction copies with hot rollback journals
- No dependency on FUSE-safe file locking during the copy

Usage:
    from db.backup import create_backup, verify_backup, list_backups

    result = create_backup(db_path)          # creates + verifies
    results = list_backups(backup_dir)       # sorted oldest-first
    result = verify_backup(backup_path)      # re-check an existing backup
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default: keep 3 rolling backups (~39 MB at current DB size).
DEFAULT_MAX_BACKUPS = 3
DEFAULT_BACKUP_DIR_NAME = "backups"
BACKUP_PREFIX = "memory.backup-"
BACKUP_SUFFIX = ".db"


@dataclass
class BackupResult:
    """Outcome of a backup creation or verification."""
    path: Path
    size_bytes: int
    memory_count: int
    integrity_ok: bool
    created_at: str  # ISO-8601


def create_backup(
    db_path: Path,
    backup_dir: Path | None = None,
    max_backups: int = DEFAULT_MAX_BACKUPS,
) -> BackupResult:
    """Create a VACUUM INTO backup of *db_path* and verify it.

    The backup is a single file — no -wal, -shm, or -journal sidecars.
    Old backups beyond *max_backups* are rotated (oldest deleted first).

    Returns a ``BackupResult`` with verification details.
    Raises ``RuntimeError`` if the backup fails integrity check.
    """
    if backup_dir is None:
        backup_dir = db_path.parent / DEFAULT_BACKUP_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{BACKUP_PREFIX}{ts}{BACKUP_SUFFIX}"

    # VACUUM INTO writes a complete, self-contained copy of the database.
    # It works even while other connections hold the DB open in WAL mode.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute(f"VACUUM INTO '{backup_path}'")
    finally:
        conn.close()

    logger.info("Backup created: %s", backup_path)

    # Verify the backup immediately.
    result = verify_backup(backup_path)
    if not result.integrity_ok:
        raise RuntimeError(
            f"Backup failed integrity check: {backup_path}"
        )

    # Rotate: delete oldest backups beyond the retention limit.
    _rotate(backup_dir, max_backups)

    return result


def verify_backup(backup_path: Path) -> BackupResult:
    """Open a backup file and verify its integrity.

    Returns a ``BackupResult``.  Does NOT raise on failure — check
    ``result.integrity_ok`` instead.
    """
    integrity_ok = False
    memory_count = -1
    created_at = ""

    if not backup_path.exists():
        return BackupResult(
            path=backup_path,
            size_bytes=0,
            memory_count=-1,
            integrity_ok=False,
            created_at="",
        )

    try:
        conn = sqlite3.connect(str(backup_path))
        try:
            # Must pass — single 'ok' row.
            check = conn.execute("PRAGMA integrity_check").fetchone()
            integrity_ok = check is not None and check[0] == "ok"

            if integrity_ok:
                row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
                memory_count = row[0] if row else -1
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Backup verification failed for %s: %s", backup_path, exc)
        integrity_ok = False

    # Parse timestamp from filename (memory.backup-YYYYMMDD_HHMMSS.db).
    stem = backup_path.stem  # memory.backup-YYYYMMDD_HHMMSS
    ts_part = stem.replace(BACKUP_PREFIX.rstrip("."), "")  # YYYYMMDD_HHMMSS
    try:
        dt = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
        created_at = dt.isoformat()
    except ValueError:
        created_at = ""

    return BackupResult(
        path=backup_path,
        size_bytes=backup_path.stat().st_size,
        memory_count=memory_count,
        integrity_ok=integrity_ok,
        created_at=created_at,
    )


def list_backups(backup_dir: Path) -> list[BackupResult]:
    """Return all backups in *backup_dir*, sorted oldest-first."""
    if not backup_dir.exists():
        return []

    paths = sorted(
        backup_dir.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"),
        key=lambda p: p.name,
    )
    return [verify_backup(p) for p in paths]


def _rotate(backup_dir: Path, max_backups: int) -> None:
    """Delete oldest backups so at most *max_backups* remain."""
    if max_backups <= 0:
        return

    backups = sorted(
        backup_dir.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"),
        key=lambda p: p.name,
    )
    while len(backups) > max_backups:
        oldest = backups.pop(0)
        logger.info("Rotating old backup: %s", oldest)
        oldest.unlink()
