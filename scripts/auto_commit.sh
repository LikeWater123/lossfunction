#!/usr/bin/env bash
# Periodic git commit script: commits and pushes every 30 minutes.
# Excludes large files (runs/, *.pt, data/) via .gitignore.
# Usage: nohup bash scripts/auto_commit.sh > /tmp/auto_commit.log 2>&1 &

set -e
cd /home/proj/lossfunction
INTERVAL=${1:-1800}  # 30 minutes in seconds

echo "[$(date)] Auto-commit started. Interval: ${INTERVAL}s"

while true; do
    sleep "$INTERVAL"
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if there are any changes to commit (staged or unstaged)
    if git diff --quiet HEAD 2>/dev/null && [ -z "$(git ls-files --others --exclude-standard)" ]; then
        echo "[$TIMESTAMP] No changes to commit."
        continue
    fi

    # Stage all changes (respecting .gitignore)
    git add -A

    # Commit with timestamp
    git commit -m "auto: periodic commit at ${TIMESTAMP}" 2>/dev/null || {
        echo "[$TIMESTAMP] Commit failed (possibly empty)."
        continue
    }

    # Push to remote
    if git push origin main 2>&1; then
        echo "[$TIMESTAMP] Pushed to origin/main."
    else
        echo "[$TIMESTAMP] Push failed. Will retry next cycle."
    fi
done
