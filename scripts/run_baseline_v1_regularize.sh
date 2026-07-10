#!/usr/bin/env bash
# PFAS baseline v1.1 (fixed): hybrid MSP + union vocab + NIST transfer + anti-overfit.
#
# Data:  data/pfas/nist_标注/nist_pfas_annot.pkl  (rep3, hybrid enum+MAGMa)
# Train: union 10k+2k, transfer_mode union, dropout 0.15, wd 1e-4, early stop 8
# Target: test cosine ~0.50, train-test gap ~0.07 (vs ~0.16 without regularize)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
QUERIES_DIR="${QUERIES_DIR:-data/pfas/nist_标注/queries}"
OUT_DIR="${OUT_DIR:-output/baseline_v1_regularize}"
GPUS="${GPUS:-1}"

mkdir -p "$OUT_DIR"

echo "=== [1/4] train baseline v1.1 (regularize) ==="
python train-graff-ms.py "$PKL" \
  --dataset pfas \
  --checkpoint "$NIST_CKPT" \
  --vocab_mode union \
  --pfas_extension_size 2000 \
  --transfer_mode union \
  --batch_size 64 \
  --max_epochs 80 \
  --gpus "$GPUS" \
  --num_workers 8 \
  --dropout 0.15 \
  --weight_decay 1e-4 \
  --early_stopping_patience 8 \
  --learning_rate 5e-4 \
  2>&1 | tee "$OUT_DIR/train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"
echo "Best checkpoint: $CKPT"

echo "=== [2/4] export queries ==="
python scripts/diagnose_fit_vocab_cosine.py export-queries \
  --pkl "$PKL" --output-dir "$QUERIES_DIR"

echo "=== [3/4] predict test ==="
python run-graff-ms.py "$CKPT" "${QUERIES_DIR}/test.tsv" "$OUT_DIR/pred_test.msp" \
  --has_isotopes 1 --gpus "$GPUS" \
  2>&1 | tee "$OUT_DIR/predict.log"

echo "=== [4/4] cosine + diagnose ==="
python cosine-similarity.py "$OUT_DIR/pred_test.msp" "$TEST_MSP" --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine_test.log"

for s in train val; do
  python run-graff-ms.py "$CKPT" "${QUERIES_DIR}/${s}.tsv" \
    "$OUT_DIR/pred_${s}.msp" --has_isotopes 1 --gpus "$GPUS"
done

python scripts/diagnose_fit_vocab_cosine.py report \
  --pkl "$PKL" --checkpoint "$CKPT" \
  --target-msp-dir data/pfas/nist_标注 \
  --pred-train "$OUT_DIR/pred_train.msp" \
  --pred-val "$OUT_DIR/pred_val.msp" \
  --pred-test "$OUT_DIR/pred_test.msp" \
  2>&1 | tee "$OUT_DIR/diagnose.log"

python scripts/cosine_by_ce.py \
  --diagnose-csv output/diagnose_fit_vocab_cosine.csv \
  --pkl "$PKL" --split test \
  2>&1 | tee "$OUT_DIR/cosine_by_ce.log"

echo "Done: $OUT_DIR/"
