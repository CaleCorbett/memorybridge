#!/usr/bin/env bash
# memorybridge log groomer
# Maintenance script: checks log size, scans for crash signatures, reports process health.
# Safe to run at any frequency — observe-and-report only, never kills processes.
#
# Wired up as a nightly Cowork scheduled task (memorybridge-log-groomer).

set -euo pipefail

DATA_DIR="${MEMORYBRIDGE_DATA:-$HOME/memorybridge}"
# Both server.py's own error stream AND the HTTP bridge's — the bridge log
# grew to 112MB unrotated (issue #177) because only server.error.log was
# groomed here, so a remote-triggerable crash loop had no size ceiling at all.
LOGS=("$DATA_DIR/logs/server.error.log" "$DATA_DIR/logs/http-bridge.error.log")
LOG_LIMIT_MB=50
BRIDGE_PORT=8484
# Issue #181 (P2-6): one-off scripts (backfill-tags.py, backfill-entities.py)
# each write a timestamped memory.db.bak-<label>-<ts> before mutating the
# DB, and nothing ever deletes them — 53MB of stale .bak DBs accumulated.
# Age-gated so a backup made moments ago (e.g. mid-migration) is never
# touched; only .bak files older than this are considered stale.
BAK_MAX_AGE_DAYS=14

# --- 1-3. Per-log size check, crash scan (before truncation), truncate ---
CRASH_HITS=""
TRUNCATED=false
TRUNCATED_SUMMARY=""

for LOG in "${LOGS[@]}"; do
  [ -f "$LOG" ] || continue

  LOG_SIZE_HUMAN=$(du -sh "$LOG" | cut -f1)
  LOG_SIZE_MB=$(du -sm "$LOG" | cut -f1)

  HITS=$(tail -200 "$LOG" 2>/dev/null \
    | grep -E 'Fatal Python error|Traceback|SIGTERM|MCP loop ended|Parent process gone|backfilled|embed failed' \
    | tail -20 || true)
  if [ -n "$HITS" ]; then
    CRASH_HITS="${CRASH_HITS}${CRASH_HITS:+$'\n'}--- $LOG ---
$HITS"
  fi

  if [ "$LOG_SIZE_MB" -gt "$LOG_LIMIT_MB" ]; then
    > "$LOG"
    TRUNCATED=true
    TRUNCATED_SUMMARY="${TRUNCATED_SUMMARY}${TRUNCATED_SUMMARY:+, }$LOG was $LOG_SIZE_HUMAN"
  fi
done

# --- 3b. Stale .bak DB cleanup (issue #181) ---
BAK_DELETED=""
BAK_DELETED_COUNT=0
if [ -d "$DATA_DIR" ]; then
  while IFS= read -r -d '' BAK; do
    BAK_SIZE_HUMAN=$(du -sh "$BAK" | cut -f1)
    BAK_DELETED="${BAK_DELETED}${BAK_DELETED:+, }$(basename "$BAK") ($BAK_SIZE_HUMAN)"
    BAK_DELETED_COUNT=$((BAK_DELETED_COUNT + 1))
    rm -f "$BAK"
  done < <(find "$DATA_DIR" -maxdepth 1 -name '*.bak*' -type f -mtime "+${BAK_MAX_AGE_DAYS}" -print0 2>/dev/null)
fi

# --- 4. Process health check ---
# Identify the HTTP bridge PID (legitimately runs with PROC_PPID 1)
BRIDGE_PID=$(lsof -ti:$BRIDGE_PORT 2>/dev/null || true)

# All server.py processes (excluding disclaimer wrappers)
PROCS=$(ps -axo pid,ppid,command | grep '[s]erver.py' | grep -v disclaimer || true)

ORPHANS=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  PID=$(echo "$line" | awk '{print $1}')
  PROC_PPID=$(echo "$line" | awk '{print $2}')
  # PROC_PPID 1 + not the bridge = orphan that survived the watchdog
  if [ "$PROC_PPID" = "1" ] && [ "$PID" != "$BRIDGE_PID" ]; then
    ORPHANS="${ORPHANS}\nORPHAN DETECTED: PID $PID (PROC_PPID 1, not bridge) — watchdog failed to clean up"
  fi
done <<< "$PROCS"

PROC_COUNT=$(echo "$PROCS" | grep -c '.' || true)

# --- 5. Report ---
ISSUES=false

if [ -n "$CRASH_HITS" ]; then
  ISSUES=true
  echo "=== CRASH SIGNATURES FOUND ==="
  echo "$CRASH_HITS"
fi

if [ -z "$BRIDGE_PID" ]; then
  ISSUES=true
  echo "WARNING: HTTP bridge is DOWN (nothing listening on port $BRIDGE_PORT)"
fi

if [ -n "$ORPHANS" ]; then
  ISSUES=true
  printf "%b\n" "$ORPHANS"
fi

if $TRUNCATED; then
  ISSUES=true
  echo "Log(s) truncated (over ${LOG_LIMIT_MB}MB limit): $TRUNCATED_SUMMARY"
fi

if [ "$BAK_DELETED_COUNT" -gt 0 ]; then
  ISSUES=true
  echo "Deleted $BAK_DELETED_COUNT stale .bak DB file(s) (older than ${BAK_MAX_AGE_DAYS}d): $BAK_DELETED"
fi

if $ISSUES; then
  echo ""
  BRIDGE_STATUS="${BRIDGE_PID:-NONE}"
  echo "Summary: $PROC_COUNT server.py process(es), bridge PID=$BRIDGE_STATUS"
fi

# Silence = all clear
