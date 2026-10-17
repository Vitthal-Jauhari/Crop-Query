#!/usr/bin/env bash
# Continue the pre-registered one-seed screen after its common base completes.
set -euo pipefail
cd "$(dirname "$0")/.."
base_log=logs/vision_base_2026-10-15.log
for attempt in $(seq 1 480); do
  if grep -q '^base epoch=5 ' "$base_log"; then
    break
  fi
  if ! pgrep -f '^python -u scripts/train_clean_resnet.py base --epochs 6 --batch-size 32 --workers 8$' >/dev/null; then
    echo 'Base process exited before completing epoch 5; see base log.' >&2
    exit 1
  fi
  sleep 30
done
if ! grep -q '^base epoch=5 ' "$base_log"; then
  echo 'Timed out waiting for common base completion.' >&2
  exit 1
fi
printf 'Common base completed; starting matched CE continuation.\n'
python -u scripts/train_clean_resnet.py ce --epochs 4 --batch-size 32 --workers 8 > logs/vision_ce_2026-10-15.log 2>&1
printf 'CE continuation completed; starting matched F2 continuation.\n'
python -u scripts/train_clean_resnet.py f2 --epochs 4 --batch-size 32 --workers 8 > logs/vision_f2_2026-10-15.log 2>&1
printf 'Both arms completed; opening the untouched test partition once.\n'
python -u scripts/train_clean_resnet.py evaluate --batch-size 64 --workers 8 > logs/vision_evaluate_2026-10-15.log 2>&1
printf 'Paired test evaluation completed.\n'
