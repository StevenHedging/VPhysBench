#!/usr/bin/env bash
# Self-locating launcher for the dependency-free VPhysBench bootstrap planner.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

profile="metadata"
for ((index = 1; index <= $#; index++)); do
    argument="${!index}"
    case "$argument" in
        --profile)
            next=$((index + 1))
            if ((next <= $#)); then
                profile="${!next}"
            fi
            ;;
        --profile=*)
            profile="${argument#--profile=}"
            ;;
    esac
done

if [[ -n "${VPHYSBENCH_BOOTSTRAP_PYTHON:-}" ]]; then
    bootstrap_python="$VPHYSBENCH_BOOTSTRAP_PYTHON"
else
    if [[ "$profile" == "evaluation" ]]; then
        candidates=(python3.12 python3)
        minimum="3.12"
    else
        candidates=(python3.12 python3.11 python3)
        minimum="3.11"
    fi
    minimum_major="${minimum%%.*}"
    minimum_minor="${minimum#*.}"
    bootstrap_python=""
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= ($minimum_major, $minimum_minor) else 1)" 2>/dev/null; then
            bootstrap_python="$candidate"
            break
        fi
    done
    if [[ -z "$bootstrap_python" ]]; then
        printf 'error: no Python %s+ interpreter found; set VPHYSBENCH_BOOTSTRAP_PYTHON\n' "$minimum" >&2
        exit 2
    fi
fi

PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    exec "$bootstrap_python" -m physbench.bootstrap --project-root "$PROJECT_ROOT" "$@"
