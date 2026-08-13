#!/usr/bin/env python3
"""One-time cleanup of stale/broken backup files from the data directory.

The files removed here are legacy backups created by shutil.copy2 (or
external tools) that have known issues:

- Misnamed WAL sidecars (.wal instead of -wal)
- Mid-transaction copies with hot rollback journals
- Pre-data-loss timestamps (no recovery value)
- Vestigial lock files with no code references

Dry-run by default.  Pass --execute to actually delete.

Usage:
    python scripts/cleanup_stale_backups.py              # dry-run
    python scripts/cleanup_stale_backups.py --execute     # delete files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Stale files to remove (relative to data dir).
STALE_FILES = [
    "memory.db.bak-20260623_045441",
    "memory.db.bak-auto-20260706_103726",
    "memory.db.bak-auto-20260706_103726-journal",
    "memory.db.bak-auto-20260706_103740",
    "memory.db.bak-auto-20260706_103740.wal",          # misnamed sidecar
    "memory.db.bak-before-infra-prune-162157",
    "memory.db.bak-before-wipe-20260621-105735",
    "memory.lock",                                      # vestigial, 0 bytes
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / "memorybridge",
                        help="path to the MemoryBridge data directory")
    parser.add_argument("--execute", action="store_true",
                        help="actually delete the files (default: dry-run)")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    found = []
    for name in STALE_FILES:
        p = data_dir / name
        if p.exists():
            size = p.stat().st_size
            found.append((p, size))

    if not found:
        print("No stale backup files found.  Nothing to do.")
        return 0

    total_bytes = sum(s for _, s in found)
    total_mb = total_bytes / 1024 / 1024

    print(f"{'DRY RUN — ' if not args.execute else ''}Stale backup files ({total_mb:.1f} MB total):\n")
    for p, size in found:
        size_str = f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size} B"
        print(f"  {p.name:<50}  {size_str:>10}")

    if args.execute:
        print()
        for p, _ in found:
            p.unlink()
            print(f"  Deleted: {p.name}")
        print(f"\nCleaned up {len(found)} files ({total_mb:.1f} MB freed).")
    else:
        print(f"\nRe-run with --execute to delete {len(found)} files ({total_mb:.1f} MB).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
