#!/usr/bin/env bash
# Build a trial/criteria file subset from an ID list and optionally index into Elasticsearch.

set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE=""
IDS_FILE="${IDS_FILE:-$ROOT_DIR/results/qwen_processed_trial_ids.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/data/subset}"
SRC_TRIALS="${SRC_TRIALS:-$ROOT_DIR/data/processed_trials}"
SRC_CRITERIA="${SRC_CRITERIA:-$ROOT_DIR/data/processed_criteria}"
DO_INDEX=1
TRIALS_INDEX="${TRIALS_INDEX:-clinical_trials_sub}"
CRITERIA_INDEX="${CRITERIA_INDEX:-trials_eligibility_sub}"
BATCH_TRIALS="${BATCH_TRIALS:-100}"
BATCH_CRITERIA="${BATCH_CRITERIA:-100}"
MAX_WORKERS_CRITERIA="${MAX_WORKERS_CRITERIA:-100}"
PROCESSED_IDS_FILE="$ROOT_DIR/utils/Indexer/processed_ids.txt"
_PROCESSED_IDS_BACKUP=""

usage() {
  cat <<'EOF'
create_sub_dset.sh — copy processed trial JSONs and/or criteria subfolders, then index a subset ES.

Usage:
  ./scripts/create_sub_dset.sh [MODE] [options]

MODE (default: all):
  all, both     Copy trials + criteria; index both
                  (before trials copy: each ID must have SRC_CRITERIA/<NCT>/; aborts if missing)
  trials        Copy data/processed_trials/*.json subset only; index trials
                  (each ID must have SRC_CRITERIA/<NCT>/; script exits immediately if missing)
  criteria      Copy data/processed_criteria/<NCT>/ subtrees only; index criteria

Options:
  -f, --ids-file PATH      File with one NCT id per line (default: results/qwen_processed_trial_ids.txt)
  -o, --out-root DIR       Output root (default: data/subset). Used only to build
                             default --out-trials / --out-criteria paths.
  --out-trials DIR         Destination dir for subset trial *.json
                             (default: OUT_ROOT/processed_trials; env: OUT_TRIALS)
  --out-criteria DIR       Destination root for subset criteria/<NCT>/ trees
                             (default: OUT_ROOT/processed_criteria; env: OUT_CRITERIA)
  --src-trials DIR         Source trial JSONs (default: data/processed_trials)
  --src-criteria DIR       Source criteria tree (default: data/processed_criteria)
  -n, --no-index           Only copy files; do not run Elasticsearch indexers
  --trials-index NAME      Target ES index for trials (default: clinical_trials_sub)
  --criteria-index NAME    Target ES index for criteria (default: trials_eligibility_sub)
  --batch-trials N         Bulk batch size for index_trials.py (default: 100)
  --batch-criteria N       Bulk batch size for index_criteria.py (default: 100)
  --max-workers-criteria N Parallel trials for index_criteria.py (default: 100)
  -h, --help               Show this help

Environment (optional overrides):
  IDS_FILE, OUT_ROOT, OUT_TRIALS, OUT_CRITERIA, SRC_TRIALS, SRC_CRITERIA,
  TRIALS_INDEX, CRITERIA_INDEX, ...

EOF
}

restore_processed_ids() {
  if [[ -n "$_PROCESSED_IDS_BACKUP" && -f "$_PROCESSED_IDS_BACKUP" ]]; then
    mv -f "$_PROCESSED_IDS_BACKUP" "$PROCESSED_IDS_FILE"
    _PROCESSED_IDS_BACKUP=""
  fi
}

trap restore_processed_ids EXIT

# Before copying trial JSONs, ensure each listed ID has processed criteria (directory per NCT).
require_criteria_for_listed_trials() {
  info "Checking processed criteria exist for every ID in $IDS_FILE (under $SRC_CRITERIA)"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr -d '\r')"
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    local trial_id="$line"
    local crit_dir="$SRC_CRITERIA/$trial_id"
    if [[ ! -d "$crit_dir" ]]; then
      echo "" >&2
      echo "================================================================" >&2
      echo "  FATAL: trial subset aborted — CRITERIA MISSING for this NCT id" >&2
      echo "================================================================" >&2
      echo "  trial_id:     $trial_id" >&2
      echo "  expected_dir: $crit_dir" >&2
      echo "  ids_file:     $IDS_FILE" >&2
      echo "  src_criteria: $SRC_CRITERIA" >&2
      echo "================================================================" >&2
      echo "" >&2
      exit 1
    fi
  done < "$IDS_FILE"
}

copy_trials_subset() {
  local missing=0
  mkdir -p "$OUT_TRIALS"
  info "Copying trial JSONs into $OUT_TRIALS"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr -d '\r')"
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    local trial_id="$line"
    local src="$SRC_TRIALS/${trial_id}.json"
    if [[ -f "$src" ]]; then
      cp -f "$src" "$OUT_TRIALS/"
    else
      warn "Missing trial JSON: $src"
      missing=$((missing + 1))
    fi
  done < "$IDS_FILE"
  if (( missing > 0 )); then
    warn "Trials missing: $missing (see messages above)"
  fi
}

copy_criteria_subset() {
  local missing=0
  mkdir -p "$OUT_CRITERIA"
  info "Copying criteria folders into $OUT_CRITERIA"
  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr -d '\r')"
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    local trial_id="$line"
    local src="$SRC_CRITERIA/$trial_id"
    if [[ -d "$src" ]]; then
      rm -rf "$OUT_CRITERIA/$trial_id"
      cp -R "$src" "$OUT_CRITERIA/"
    else
      warn "Missing criteria directory: $src"
      missing=$((missing + 1))
    fi
  done < "$IDS_FILE"
  if (( missing > 0 )); then
    warn "Criteria folders missing: $missing (see messages above)"
  fi
}

index_trials() {
  info "Indexing trials → index='$TRIALS_INDEX' folder='$OUT_TRIALS'"
  ( cd "$ROOT_DIR/utils/Indexer" && python index_trials.py \
      --config config.json \
      --processed-folder "$OUT_TRIALS" \
      --index-name "$TRIALS_INDEX" \
      --batch-size "$BATCH_TRIALS" )
}

index_criteria() {
  # CriteriaIndexer skips NCTs listed in processed_ids.txt; use a clean slate for subset index.
  if [[ -f "$PROCESSED_IDS_FILE" ]]; then
    _PROCESSED_IDS_BACKUP="${PROCESSED_IDS_FILE}.bak.create_sub_dset.$$"
    cp "$PROCESSED_IDS_FILE" "$_PROCESSED_IDS_BACKUP"
    : >"$PROCESSED_IDS_FILE"
    warn "Temporarily emptied $PROCESSED_IDS_FILE for subset criteria index (restored on exit)."
  else
    mkdir -p "$(dirname "$PROCESSED_IDS_FILE")"
    : >"$PROCESSED_IDS_FILE"
  fi

  info "Indexing criteria → index='$CRITERIA_INDEX' folder='$OUT_CRITERIA'"
  ( cd "$ROOT_DIR/utils/Indexer" && python index_criteria.py \
      --config config.json \
      --processed-folder "$OUT_CRITERIA" \
      --index-name "$CRITERIA_INDEX" \
      --batch-size "$BATCH_CRITERIA" \
      --max-workers "$MAX_WORKERS_CRITERIA" )

  restore_processed_ids
  trap - EXIT
}

# --- parse args ---
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -f|--ids-file) IDS_FILE="$2"; shift 2 ;;
    -o|--out-root) OUT_ROOT="$2"; shift 2 ;;
    --out-trials) OUT_TRIALS="$2"; shift 2 ;;
    --out-criteria) OUT_CRITERIA="$2"; shift 2 ;;
    --src-trials) SRC_TRIALS="$2"; shift 2 ;;
    --src-criteria) SRC_CRITERIA="$2"; shift 2 ;;
    -n|--no-index) DO_INDEX=0; shift ;;
    --trials-index) TRIALS_INDEX="$2"; shift 2 ;;
    --criteria-index) CRITERIA_INDEX="$2"; shift 2 ;;
    --batch-trials) BATCH_TRIALS="$2"; shift 2 ;;
    --batch-criteria) BATCH_CRITERIA="$2"; shift 2 ;;
    --max-workers-criteria) MAX_WORKERS_CRITERIA="$2"; shift 2 ;;
    all|both|trials|criteria) MODE="$1"; shift ;;
    -*)
      error "Unknown option: $1 (use -h)"
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -gt 0 ]]; then
  if [[ -n "$MODE" ]]; then
    error "Unexpected extra arguments: ${ARGS[*]}"
  fi
  MODE="${ARGS[0]}"
  [[ ${#ARGS[@]} -eq 1 ]] || error "Unexpected extra arguments: ${ARGS[*]}"
fi

if [[ -z "$MODE" ]]; then
  MODE="all"
fi

case "$MODE" in
  all|both) DO_TRIALS=1; DO_CRITERIA=1 ;;
  trials)   DO_TRIALS=1; DO_CRITERIA=0 ;;
  criteria) DO_TRIALS=0; DO_CRITERIA=1 ;;
  *) error "Mode must be all|both|trials|criteria, got: $MODE" ;;
esac

[[ -f "$IDS_FILE" ]] || error "ID list not found: $IDS_FILE"
[[ -d "$SRC_TRIALS" ]] || error "Source trials dir not found: $SRC_TRIALS"
[[ -d "$SRC_CRITERIA" ]] || error "Source criteria dir not found: $SRC_CRITERIA"

OUT_TRIALS="${OUT_TRIALS:-$OUT_ROOT/processed_trials}"
OUT_CRITERIA="${OUT_CRITERIA:-$OUT_ROOT/processed_criteria}"

info "Mode=$MODE  out_root=$OUT_ROOT  ids=$(wc -l <"$IDS_FILE" | tr -d ' ') lines in $IDS_FILE"
info "OUT_TRIALS=$OUT_TRIALS  OUT_CRITERIA=$OUT_CRITERIA"

if [[ "$DO_TRIALS" -eq 1 ]]; then
  require_criteria_for_listed_trials
  copy_trials_subset
fi
if [[ "$DO_CRITERIA" -eq 1 ]]; then
  copy_criteria_subset
fi

if [[ "$DO_INDEX" -eq 1 ]]; then
  if [[ "$DO_TRIALS" -eq 1 ]]; then
    index_trials
  fi
  if [[ "$DO_CRITERIA" -eq 1 ]]; then
    index_criteria
  fi
  info "Indexing finished. Use ES indexes: trials=$TRIALS_INDEX criteria=$CRITERIA_INDEX"
else
  info "Skip indexing (--no-index). Trials dir: $OUT_TRIALS | Criteria dir: $OUT_CRITERIA"
  trap - EXIT
fi

info "Done."

