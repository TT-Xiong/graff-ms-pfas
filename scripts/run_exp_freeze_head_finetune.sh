#!/usr/bin/env bash
# Finetune cov_emb + clf with frozen NIST backbone (GNN/decoder/isotope_shift).
# Compare baseline union vs pruned 5k+5k vocab under the same training recipe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
TEST_QUERIES="${TEST_QUERIES:-data/pfas/nist_标注/test_queries.tsv}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
GPUS="${GPUS:-1}"
EXP="${EXP:-baseline}"   # baseline | pruned5k5k

case "$EXP" in
  baseline)
    OUT_DIR="${OUT_DIR:-output/exp_freeze_head_baseline_union}"
    VOCAB_ARGS=(--vocab_mode union --pfas_extension_size 2000 --transfer_mode union)
    ;;
  pruned5k5k)
    OUT_DIR="${OUT_DIR:-output/exp_freeze_head_pruned_5k5k}"
    VOCAB_ARGS=(
      --vocab_mode union
      --nist_vocab_keep 5000
      --pfas_extension_size 5000
      --transfer_mode clf_map
    )
    ;;
  *)
    echo "Unknown EXP=$EXP (use baseline or pruned5k5k)" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

echo "=== train ($EXP): freeze_backbone + cov_emb_lr + clf_lr ==="
python train-graff-ms.py "$PKL" \
  --dataset pfas \
  --checkpoint "$NIST_CKPT" \
  "${VOCAB_ARGS[@]}" \
  --freeze_backbone \
  --cov_emb_lr 5e-4 \
  --clf_lr 1e-3 \
  --batch_size "$BATCH_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --gpus "$GPUS" \
  --num_workers 8 \
  2>&1 | tee "$OUT_DIR/train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"
echo "Best checkpoint: $CKPT"

echo "=== predict ==="
python run-graff-ms.py "$CKPT" "$TEST_QUERIES" "$OUT_DIR/pred.msp" \
  --has_isotopes 1 --gpus "$GPUS" \
  2>&1 | tee "$OUT_DIR/predict.log"

echo "=== cosine ==="
python cosine-similarity.py "$OUT_DIR/pred.msp" "$TEST_MSP" --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine.log"

echo "Done: $OUT_DIR/"
