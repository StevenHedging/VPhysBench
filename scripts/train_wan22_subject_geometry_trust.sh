#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}"
TEMPLATE="$BENCHMARK_ROOT/scripts/train_wan22_subject_motion.sh"
if [[ "$(grep -c 'wan22_subject_motion_train.py' "$TEMPLATE")" != "1" ]]; then
  echo "Subject-motion launcher template changed unexpectedly: $TEMPLATE" >&2
  exit 1
fi

sed \
  's#scripts/wan22_subject_motion_train.py#scripts/wan22_subject_geometry_trust_train.py#' \
  "$TEMPLATE" | bash
