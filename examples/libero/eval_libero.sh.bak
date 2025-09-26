#!/usr/bin/env bash
set -euo pipefail

# Usage: bash examples/libero/eval_libero.sh <task_suite> <episodes_per_task> <out_markdown>
# Requires: CKPT env var pointing to checkpoint directory containing model.safetensors (pi0.5_libero), optional CONFIG (default pi05_libero)

TASK_SUITE=${1:-libero_10}
EPISODES=${2:-5}
OUT_MD=${3:-duquant_eval.md}

CONFIG=${CONFIG:-pi05_libero}

# Choose runner: prefer uv if available
if command -v uv >/dev/null 2>&1; then
  PYRUN=(uv run)
else
  PYRUN=(python)
fi

if [[ -z "${CKPT:-}" ]]; then
  echo "ERROR: please set CKPT to the checkpoint directory containing model.safetensors" >&2
  exit 1
fi

function run_once() {
  local label="$1"; shift
  local log_dir="duquant_eval_logs"
  mkdir -p "$log_dir"
  local srv_log="$log_dir/server_${label}.log"
  local cli_log="$log_dir/client_${label}.log"

  # Start server
  (
    OPENPI_DUQUANT_PACKDIR=${OPENPI_DUQUANT_PACKDIR:-./duquant_packed} \
    SERVER_ARGS="policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT" \
    "${PYRUN[@]}" scripts/serve_policy.py --env LIBERO $SERVER_ARGS
  ) > "$srv_log" 2>&1 &
  SRV_PID=$!
  sleep 5

  # Client eval
  /usr/bin/time -p "${PYRUN[@]}" examples/libero/main.py --task-suite-name "$TASK_SUITE" --num_trials_per_task "$EPISODES" > "$cli_log" 2>&1 || true

  # Stop server
  kill $SRV_PID >/dev/null 2>&1 || true
  sleep 1

  # Parse results
  local success_line
  success_line=$(grep -E "Total success rate:" -m 1 "$cli_log" | tail -n 1 | awk '{print $4}')
  local total_episodes_line
  total_episodes_line=$(grep -E "Total episodes:" -m 1 "$cli_log" | tail -n 1 | awk '{print $3}')
  local time_secs
  time_secs=$(grep -E "^real" -m 1 "$cli_log" | awk '{print $2}' || echo "NA")
  echo "$success_line" "$total_episodes_line" "$time_secs"
}

function run_variant() {
  local label="$1"; shift
  echo "Running variant: $label" >&2
  # shellcheck disable=SC2068
  ( $@ )
}

declare -A RESULTS

# 1) FP32 baseline
unset OPENPI_DUQUANT_DRYRUN OPENPI_DUQUANT_SCOPE OPENPI_DUQUANT_INCLUDE OPENPI_DUQUANT_EXCLUDE OPENPI_DUQUANT_WBITS OPENPI_DUQUANT_PERMUTE OPENPI_DUQUANT_BLOCK OPENPI_DUQUANT_LS OPENPI_DUQUANT_ABITS OPENPI_DUQUANT_INCLUDE_ACTION_HEAD
read -r s e t < <(run_once FP32)
RESULTS[FP32]="$s $e $t"

# 2) W4Ainf (weight-only)
export OPENPI_DUQUANT_ABITS=0
export OPENPI_DUQUANT_SCOPE=${OPENPI_DUQUANT_SCOPE:-policy.dit.}
read -r s e t < <(run_once W4Ainf)
RESULTS[W4Ainf]="$s $e $t"

# 3) W4A8 (no R/P)
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_PERMUTE=0
read -r s e t < <(run_once W4A8_noRP)
RESULTS[W4A8_noRP]="$s $e $t"

# 4) W4A8 (+R/P)
export OPENPI_DUQUANT_PERMUTE=1
read -r s e t < <(run_once W4A8_RP)
RESULTS[W4A8_RP]="$s $e $t"

# Optional 5) QAT placeholder (not enabling by default)

# Write Markdown report
{
  echo "| Variant | Success | Episodes | Time (s) |"
  echo "|---|---:|---:|---:|"
  for k in FP32 W4Ainf W4A8_noRP W4A8_RP; do
    IFS=' ' read -r s e t <<< "${RESULTS[$k]}"
    echo "| $k | ${s:-NA} | ${e:-NA} | ${t:-NA} |"
  done
} > "$OUT_MD"

echo "Wrote $OUT_MD"
