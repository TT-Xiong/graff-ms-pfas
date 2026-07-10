#!/usr/bin/env bash
# Anti-overfit baseline: union vocab + stronger regularization + early stopping.
# Compare test cosine / train-test gap vs default training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
TEST_QUERIES="${TEST_QUERIES:-data/pfas/nist_标注/test_queries.tsv}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
OUT_DIR="${OUT_DIR:-output/exp_anti_overfit}"
MODE="${MODE:-regularize}"   # regularize | freeze_head | both

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
GPUS="${GPUS:-1}"
DROPOUT="${DROPOUT:-0.15}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EARLY_PATIENCE="${EARLY_PATIENCE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"

mkdir -p "$OUT_DIR"

TRAIN_EXTRA=(
  --dataset pfas
  --checkpoint "$NIST_CKPT"
  --vocab_mode union
  --pfas_extension_size 2000
  --transfer_mode union
  --batch_size "$BATCH_SIZE"
  --max_epochs "$MAX_EPOCHS"
  --gpus "$GPUS"
  --num_workers 8
  --dropout "$DROPOUT"
  --weight_decay "$WEIGHT_DECAY"
  --early_stopping_patience "$EARLY_PATIENCE"
  --learning_rate "$LEARNING_RATE"
)

case "$MODE" in
  regularize)
    OUT_DIR="${OUT_DIR}_regularize"
    ;;
  freeze_head)
    OUT_DIR="${OUT_DIR}_freeze_head"
    TRAIN_EXTRA+=(--freeze_backbone --cov_emb_lr 5e-4 --clf_lr 1e-3)
    ;;
  both)
    OUT_DIR="${OUT_DIR}_both"
    TRAIN_EXTRA+=(--freeze_backbone --cov_emb_lr 5e-4 --clf_lr 1e-3)
    ;;
  *)
    echo "Unknown MODE=$MODE (regularize|freeze_head|both)" >&2
    exit 1
    ;;
esac
mkdir -p "$OUT_DIR"

echo "=== train (MODE=$MODE) ==="
python train-graff-ms.py "$PKL" "${TRAIN_EXTRA[@]}" 2>&1 | tee "$OUT_DIR/train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"

echo "=== export queries ==="
python scripts/diagnose_fit_vocab_cosine.py export-queries \
  --pkl "$PKL" \
  --output-dir data/pfas/nist_标注/queries

echo "=== predict train / val / test ==="
for s in train val test; do
  python run-graff-ms.py "$CKPT" "data/pfas/nist_标注/queries/${s}.tsv" \
    "$OUT_DIR/pred_${s}.msp" --has_isotopes 1 --gpus "$GPUS"
done

echo "=== overall cosine ==="
python cosine-similarity.py "$OUT_DIR/pred_test.msp" "$TEST_MSP" --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine_test.log"

echo "=== diagnose report ==="
python scripts/diagnose_fit_vocab_cosine.py report \
  --pkl "$PKL" \
  --checkpoint "$CKPT" \
  --target-msp-dir data/pfas/nist_标注 \
  --pred-train "$OUT_DIR/pred_train.msp" \
  --pred-val "$OUT_DIR/pred_val.msp" \
  --pred-test "$OUT_DIR/pred_test.msp" \
  2>&1 | tee "$OUT_DIR/diagnose.log"

echo "=== cosine by CE (test) ==="
python scripts/cosine_by_ce.py \
  --diagnose-csv output/diagnose_fit_vocab_cosine.csv \
  --pkl "$PKL" --split test \
  2>&1 | tee "$OUT_DIR/cosine_by_ce.log"

echo "Done: $OUT_DIR/"
