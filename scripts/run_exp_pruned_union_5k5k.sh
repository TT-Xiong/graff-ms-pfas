#!/usr/bin/env bash
# Exp: prune NIST vocab to 5k (by PFAS train intensity) + 5k PFAS extensions (~10k total).
# Run on GPU server with conda env `graff` from repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
TEST_QUERIES="${TEST_QUERIES:-data/pfas/nist_标注/test_queries.tsv}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
OUT_DIR="${OUT_DIR:-output/exp_pruned_union_5k5k}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
GPUS="${GPUS:-1}"

mkdir -p "$OUT_DIR"

echo "=== [1/4] preprocess (optional; skip if pkl is up to date) ==="
if [[ "${SKIP_PREPROCESS:-0}" != "1" ]]; then
  python preprocess-pfas.py --no-parallel
fi

echo "=== [2/4] train: 5k NIST + 5k PFAS, clf_map transfer ==="
python train-graff-ms.py "$PKL" \
  --dataset pfas \
  --checkpoint "$NIST_CKPT" \
  --vocab_mode union \
  --nist_vocab_keep 5000 \
  --pfas_extension_size 5000 \
  --transfer_mode clf_map \
  --batch_size "$BATCH_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --gpus "$GPUS" \
  --num_workers 8 \
  2>&1 | tee "$OUT_DIR/train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "Best checkpoint: $CKPT"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"

echo "=== [3/4] predict on test ==="
python run-graff-ms.py "$CKPT" "$TEST_QUERIES" "$OUT_DIR/pred.msp" \
  --has_isotopes 1 \
  --gpus "$GPUS" \
  2>&1 | tee "$OUT_DIR/predict.log"

echo "=== [4/4] cosine vs annotated test ==="
python cosine-similarity.py "$OUT_DIR/pred.msp" "$TEST_MSP" --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine.log"

echo "Done. Results in $OUT_DIR/"
