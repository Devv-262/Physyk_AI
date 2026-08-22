#!/usr/bin/env bash
# ==============================================================================
# Physyk AI — Nemotron Watchdog
# ==============================================================================
# Polls Nemotron's health endpoint and auto-restarts it (via start_nemotron.sh) if
# it goes down unexpectedly — e.g. a stray `pkill -f vllm`, an OOM, someone running
# the OLD version of run.sh's blanket stop_services(). Does NOT fight an explicit,
# intentional stop: `./run.sh --stop` sets logs/.nemotron_intentionally_stopped
# before killing Nemotron, and this watchdog checks that flag before acting.
#
# Usage:
#   nohup bash nemotron_watchdog.sh > /opt/dlami/nvme/physyk/logs/nemotron_watchdog.log 2>&1 &
#   disown
# Stop it the same way you'd stop any background process:
#   pkill -f nemotron_watchdog.sh
# ==============================================================================

set -uo pipefail

WS="/opt/dlami/nvme/physyk/workspace"
FLAG="/opt/dlami/nvme/physyk/logs/.nemotron_intentionally_stopped"
PIDFILE="/opt/dlami/nvme/physyk/logs/.nemotron_watchdog.pid"

# Refuse to start a second instance — two watchdogs would double up restart attempts
# against the same fail-count logic. Stale pidfile (process no longer alive) is fine.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "Nemotron watchdog already running (pid $(cat "$PIDFILE")) — exiting." >&2
    exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

CHECK_INTERVAL_S=20
# Require this many consecutive failed checks before restarting — avoids restarting
# on one flaky health check.
FAILS_BEFORE_RESTART=3
# Nemotron-30B-FP8 cold boot measured at ~142s engine-init-to-ready (logs/nemotron_restart3.log,
# 15:47:23 -> 15:49:45), plus load variance — give it real headroom before treating a
# still-booting instance as "down" again.
BOOT_GRACE_TIMEOUT_S=300
BOOT_POLL_INTERVAL_S=5

fail_count=0

echo "[$(date -u +%FT%TZ)] Nemotron watchdog started — polling every ${CHECK_INTERVAL_S}s."

while true; do
    sleep "$CHECK_INTERVAL_S"

    if [ -f "$FLAG" ]; then
        fail_count=0
        continue   # human explicitly stopped it — do nothing until they restart it
    fi

    if curl -sf -m 5 http://localhost:8000/health > /dev/null 2>&1; then
        fail_count=0
        continue
    fi

    fail_count=$((fail_count + 1))
    echo "[$(date -u +%FT%TZ)] Nemotron health check failed (${fail_count}/${FAILS_BEFORE_RESTART})."

    if [ "$fail_count" -ge "$FAILS_BEFORE_RESTART" ]; then
        echo "[$(date -u +%FT%TZ)] Nemotron appears down and not intentionally stopped — restarting."
        bash "$WS/start_nemotron.sh" >> /opt/dlami/nvme/physyk/logs/nemotron_watchdog_restarts.log 2>&1 &
        fail_count=0

        # Actively wait for real readiness (not a blind sleep) before resuming normal
        # health-checking — a fixed short sleep here previously risked firing a SECOND
        # restart (killing the still-booting first one) if boot took longer than the
        # sleep, which could loop indefinitely under load. Bail out of the wait early,
        # without treating it as a new failure, if someone sets the stop flag mid-boot.
        waited=0
        while [ "$waited" -lt "$BOOT_GRACE_TIMEOUT_S" ]; do
            if [ -f "$FLAG" ]; then
                break
            fi
            if curl -sf -m 5 http://localhost:8000/health > /dev/null 2>&1; then
                echo "[$(date -u +%FT%TZ)] Nemotron back up after restart (${waited}s)."
                break
            fi
            sleep "$BOOT_POLL_INTERVAL_S"
            waited=$((waited + BOOT_POLL_INTERVAL_S))
        done
        if [ "$waited" -ge "$BOOT_GRACE_TIMEOUT_S" ]; then
            echo "[$(date -u +%FT%TZ)] Nemotron still not healthy after ${BOOT_GRACE_TIMEOUT_S}s post-restart — resuming normal checks (will retry)."
        fi
    fi
done
