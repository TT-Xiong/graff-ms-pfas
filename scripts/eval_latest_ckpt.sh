#!/usr/bin/env bash
# Evaluate latest PFAS checkpoint: predict test + cosine (+ optional full diagnose).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot.pkl}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
QUERIES_DIR="${QUERIES_DIR:-data/pfas/nist_标注/queries}"
OUT_DIR="${OUT_DIR:-output/eval_baseline_rerun}"
GPUS="${GPUS:-1}"
MATCHMS_TOL="${MATCHMS_TOL:-0.1}"
FULL_DIAGNOSE="${FULL_DIAGNOSE:-0}"   # 1 = train/val/test diagnose + CE breakdown

mkdir -p "$OUT_DIR"

# Use newest graff checkpoint unless CKPT is set
if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
fi
echo "Checkpoint: $CKPT"
echo "$CKPT" > "$OUT_DIR/checkpoint.txt"

if [[ ! -f "${QUERIES_DIR}/test.tsv" ]]; then
  echo "=== export test queries ==="
  python scripts/diagnose_fit_vocab_cosine.py export-queries \
    --pkl "$PKL" \
    --output-dir "$QUERIES_DIR"
fi

echo "=== predict test ==="
python run-graff-ms.py "$CKPT" "${QUERIES_DIR}/test.tsv" "$OUT_DIR/pred_test.msp" \
  --has_isotopes 1 --gpus "$GPUS" \
  2>&1 | tee "$OUT_DIR/predict.log"

echo "=== cosine (test) ==="
python cosine-similarity.py "$OUT_DIR/pred_test.msp" "$TEST_MSP" --matchms_tol "$MATCHMS_TOL" \
  2>&1 | tee "$OUT_DIR/cosine_test.log"

if [[ "$FULL_DIAGNOSE" == "1" ]]; then
  echo "=== predict train / val ==="
  for s in train val; do
    python run-graff-ms.py "$CKPT" "${QUERIES_DIR}/${s}.tsv" \
      "$OUT_DIR/pred_${s}.msp" --has_isotopes 1 --gpus "$GPUS"
  done

  echo "=== diagnose report ==="
  python scripts/diagnose_fit_vocab_cosine.py report \
    --pkl "$PKL" \
    --checkpoint "$CKPT" \
    --target-msp-dir data/pfas/nist_标注 \
    --pred-train "$OUT_DIR/pred_train.msp" \
    --pred-val "$OUT_DIR/pred_val.msp" \
    --pred-test "$OUT_DIR/pred_test.msp" \
    --matchms-tol "$MATCHMS_TOL" \
    2>&1 | tee "$OUT_DIR/diagnose.log"

  echo "=== cosine by CE (test) ==="
  python scripts/cosine_by_ce.py \
    --diagnose-csv output/diagnose_fit_vocab_cosine.csv \
    --pkl "$PKL" --split test \
    2>&1 | tee "$OUT_DIR/cosine_by_ce.log"
fi

echo "Done: $OUT_DIR/"
