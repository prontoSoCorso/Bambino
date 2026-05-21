#!/usr/bin/env bash
# run_experiments.sh — sequential training of EVERY active model variant
# (PROJECT_STATE §2.1) followed by aggregation into a comparative grid.
#
# Models executed:
#     logreg, minirocket,
#     inception_time, film_cnn, resnet_gasf,
#     moment_mlp, moment_histgb, moment_logreg, moment_tabpfn, moment_pca_tabpfn,
#     anomaly_detector
#
# Usage:
#     bash run_experiments.sh                # all models, full schedule
#     RUN_TAG=v2 bash run_experiments.sh     # tag run-ids with a version suffix
#     SKIP_MOMENT=1 bash run_experiments.sh  # skip MOMENT (no weights / no GPU)
#     SKIP_TABPFN=1 bash run_experiments.sh  # skip TabPFN heads (heavy install)
#     MAX_EPOCHS=5 bash run_experiments.sh   # quick smoke-run
#     ONLY="logreg minirocket" bash run_experiments.sh   # restrict to a subset
#
# Output:
#     results/<run_id>/                      per-run TB + CSV + ckpts + metrics
#     results/aggregated_grid/               comparative plots (palette-locked)

set -u  # treat unset vars as errors; individual runs are allowed to fail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M)}"
DATA_DIR="${DATA_DIR:-$HERE/data/full/clean}"
ONLY="${ONLY:-}"

MAX_EPOCHS_OPT=""
if [[ -n "${MAX_EPOCHS:-}" ]]; then
    MAX_EPOCHS_OPT="--max-epochs ${MAX_EPOCHS}"
fi

echo "================================================================"
echo "  BAMBINO — sequential experiment sweep"
echo "  RUN_TAG  = ${RUN_TAG}"
echo "  DATA_DIR = ${DATA_DIR}"
echo "  PYTHON   = ${PYTHON}"
echo "  ONLY     = ${ONLY:-<all>}"
echo "================================================================"

# Sanity: clean data must exist
if [[ ! -f "${DATA_DIR}/bambino_clean.pt" && ! -f "${DATA_DIR}/training_set.pt" ]]; then
    echo "[!] No clean .pt files found in ${DATA_DIR}."
    echo "    Run the migration bridge first:"
    echo "        ${PYTHON} scripts/migrate_legacy_data.py"
    exit 1
fi

# Should we run a given model id?
should_run() {
    local model="$1"
    [[ -z "$ONLY" ]] && return 0
    for w in $ONLY; do
        [[ "$w" == "$model" ]] && return 0
    done
    return 1
}

# Run wrapper — non-fatal on individual model crashes
run() {
    local model="$1"; shift
    local run_id="$1"; shift
    if ! should_run "$model"; then
        echo "[skip] ${model}/${run_id} (not in ONLY list)"
        return
    fi
    local rest=("$@")
    echo
    echo "----------------------------------------------------------------"
    echo "  ▶  ${model}  →  run-id=${run_id}"
    echo "----------------------------------------------------------------"
    if ! ${PYTHON} main.py \
            --model "${model}" \
            --run-id "${run_id}" \
            --data-dir "${DATA_DIR}" \
            ${MAX_EPOCHS_OPT} \
            "${rest[@]}"; then
        echo "[!] ${model}/${run_id} failed — continuing with next model."
    fi
}

# ─── 1. Sklearn baselines ────────────────────────────────────────────────────
run logreg     "logreg_${RUN_TAG}"
run logreg     "logreg_noaug_${RUN_TAG}"   --no-augmentation

run minirocket "minirocket_${RUN_TAG}"
run minirocket "minirocket_noaug_${RUN_TAG}" --no-augmentation

# ─── 2. Deep TS classifiers ──────────────────────────────────────────────────
run inception_time "incept_${RUN_TAG}"
run inception_time "incept_perTrial_${RUN_TAG}"           --baseline-norm-mode per_trial
run inception_time "incept_perTrial_decay_${RUN_TAG}"     --baseline-norm-mode per_trial --use-habituation-decay
run inception_time "incept_preStimCtx_${RUN_TAG}"         --use-pre-stim-context

run film_cnn "film_${RUN_TAG}"
run film_cnn "film_perTrial_${RUN_TAG}"                   --baseline-norm-mode per_trial

# ─── 3. ResNet-GASF (image domain) ───────────────────────────────────────────
run resnet_gasf "resnet_gasf_${RUN_TAG}"

# ─── 4. MOMENT heads — skippable when momentfm or tabpfn is missing ──────────
have_moment=0
if [[ "${SKIP_MOMENT:-0}" == "1" ]]; then
    echo "[skip] MOMENT (SKIP_MOMENT=1)"
elif ${PYTHON} -c "import momentfm" 2>/dev/null; then
    have_moment=1
else
    echo "[skip] MOMENT — momentfm not installed (pip install momentfm)"
fi

if [[ $have_moment -eq 1 ]]; then
    run moment_mlp     "moment_mlp_${RUN_TAG}"
    run moment_histgb  "moment_histgb_${RUN_TAG}"
    run moment_logreg  "moment_logreg_${RUN_TAG}"

    if [[ "${SKIP_TABPFN:-0}" == "1" ]]; then
        echo "[skip] TabPFN heads (SKIP_TABPFN=1)"
    elif ${PYTHON} -c "import tabpfn" 2>/dev/null; then
        run moment_tabpfn     "moment_tabpfn_${RUN_TAG}"
        run moment_pca_tabpfn "moment_pca_tabpfn_${RUN_TAG}"
    else
        echo "[skip] TabPFN heads — tabpfn not installed (pip install tabpfn)"
    fi
fi

# ─── 5. Anomaly detector — the active research direction ─────────────────────
run anomaly_detector "ad_global_${RUN_TAG}"
run anomaly_detector "ad_perSubject_${RUN_TAG}"   --baseline-norm-mode per_subject
run anomaly_detector "ad_perTrial_${RUN_TAG}"     --baseline-norm-mode per_trial

# ─── 6. Aggregate ────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "  Aggregating results …"
echo "================================================================"
${PYTHON} -m src.utils.aggregate_results \
    --results-dir "${HERE}/results" \
    --output-dir  "${HERE}/results/aggregated_grid"

echo
echo "Done. Comparative plots: ${HERE}/results/aggregated_grid/"
