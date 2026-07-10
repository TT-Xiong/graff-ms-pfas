#!/usr/bin/env bash
# cov_emb / CE-FiLM experiments on baseline v1.1 (regularize) hyperparameters.
#
# Modes (MODE env):
#   wide_add      - wider cov_emb MLP, additive fusion (no FiLM)
#   film_decoder  - compact cov_emb + FiLM on decoder output
#   both          - wide cov_emb (768) + FiLM (default)
#
# Usage:
#   MODE=both bash scripts/run_exp_cov_emb_v1.sh
#   MODE=film_decoder OUT_DIR=output/exp_cov_film bash scripts/run_exp_cov_emb_v1.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
QUERIES_DIR="${QUERIES_DIR:-data/pfas/nist_标注/queries}"
MODE="${MODE:-both}"
GPUS="${GPUS:-1}"

# v1.1 regularize baseline hyperparameters
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
DROPOUT="${DROPOUT:-0.15}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EARLY_PATIENCE="${EARLY_PATIENCE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
COV_EMB_DIM="${COV_EMB_DIM:-}"

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
  wide_add)
    OUT_DIR="${OUT_DIR:-output/exp_cov_emb_wide_add}"
    TRAIN_EXTRA+=(--cov_conditioning add)
    if [[ -n "$COV_EMB_DIM" ]]; then
      TRAIN_EXTRA+=(--cov_emb_dim "$COV_EMB_DIM")
    else
      TRAIN_EXTRA+=(--cov_emb_dim 768)
    fi
    ;;
  film_decoder)
    OUT_DIR="${OUT_DIR:-output/exp_cov_emb_film_decoder}"
    TRAIN_EXTRA+=(--cov_conditioning film_decoder)
    ;;
  both)
    OUT_DIR="${OUT_DIR:-output/exp_cov_emb_both}"
    TRAIN_EXTRA+=(--cov_conditioning both)
    if [[ -n "$COV_EMB_DIM" ]]; then
      TRAIN_EXTRA+=(--cov_emb_dim "$COV_EMB_DIM")
    fi
    ;;
  *)
    echo "Unknown MODE=$MODE (wide_add|film_decoder|both)" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

echo "=== [1/4] train cov_emb experiment (MODE=$MODE) ==="
python train-graff-ms.py "$PKL" "${TRAIN_EXTRA[@]}" 2>&1 | tee "$OUT_DIR/train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"
echo "Best checkpoint: $CKPT"

echo "=== [2/4] export queries ==="
python scripts/diagnose_fit_vocab_cosine.py export-queries \
  --pkl "$PKL" --output-dir "$QUERIES_DIR"

echo "=== [3/4] predict train / val / test ==="
for s in train val test; do
  python run-graff-ms.py "$CKPT" "${QUERIES_DIR}/${s}.tsv" \
    "$OUT_DIR/pred_${s}.msp" --has_isotopes 1 --gpus "$GPUS"
done

echo "=== [4/4] cosine + diagnose + CE buckets ==="
python cosine-similarity.py "$OUT_DIR/pred_test.msp" "$TEST_MSP" --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine_test.log"

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
