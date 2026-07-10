#!/usr/bin/env bash
# Full train MGF -> hybrid annotate -> quality filter -> preprocess -> train -> cosine
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MGF="${MGF:-data/pfas/nist_pfas_ms2_train.mgf}"
SEED_MSP="${SEED_MSP:-data/pfas/nist_标注/train.msp}"
VAL_MSP="${VAL_MSP:-data/pfas/nist_标注/val.msp}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
TRAIN_FULL="${TRAIN_FULL:-data/pfas/nist_标注/train_full.mgf.msp}"
TRAIN_FILTERED="${TRAIN_FILTERED:-data/pfas/nist_标注/train_full_filtered.msp}"
PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot_full_filtered.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
OUT_DIR="${OUT_DIR:-output/exp_full_train_filtered}"
MIN_COV="${MIN_INTENSITY_COVERAGE:-0.35}"
MAX_PER_GROUP="${MAX_PER_GROUP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
GPUS="${GPUS:-1}"
SKIP_HYBRID="${SKIP_HYBRID:-0}"

mkdir -p "$OUT_DIR"

echo "=== [1/6] export full train MSP from MGF ==="
python scripts/build_filtered_train_from_mgf.py export \
  --mgf "$MGF" \
  --seed-msp "$SEED_MSP" \
  --output "$TRAIN_FULL" \
  2>&1 | tee "$OUT_DIR/01_export.log"

if [[ "$SKIP_HYBRID" != "1" ]]; then
  echo "=== [2/6] hybrid annotation (enum + MAGMa) on new spectra ==="
  python batch_fill_msp_hybrid.py "$TRAIN_FULL" \
    2>&1 | tee "$OUT_DIR/02_hybrid.log"
else
  echo "=== [2/6] skip hybrid (SKIP_HYBRID=1) ==="
fi

echo "=== [3/6] quality filter (min cov=${MIN_COV}, max/group=${MAX_PER_GROUP}) ==="
python scripts/build_filtered_train_from_mgf.py filter \
  --input "$TRAIN_FULL" \
  --output "$TRAIN_FILTERED" \
  --min-intensity-coverage "$MIN_COV" \
  --max-per-group "$MAX_PER_GROUP" \
  2>&1 | tee "$OUT_DIR/03_filter.log"

echo "=== [4/6] preprocess pkl (filtered train + val + test) ==="
python preprocess-pfas.py --no-parallel \
  "$TRAIN_FILTERED" "$VAL_MSP" "$TEST_MSP" \
  --output "$PKL" \
  2>&1 | tee "$OUT_DIR/04_preprocess.log"

echo "=== [5/6] train baseline union ==="
python train-graff-ms.py "$PKL" \
  --dataset pfas \
  --checkpoint "$NIST_CKPT" \
  --vocab_mode union \
  --pfas_extension_size 2000 \
  --transfer_mode union \
  --batch_size "$BATCH_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --gpus "$GPUS" \
  --num_workers 8 \
  2>&1 | tee "$OUT_DIR/05_train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"

echo "=== [6/6] predict + cosine ==="
python run-graff-ms.py "$CKPT" "data/pfas/nist_标注/test_queries.tsv" "$OUT_DIR/pred.msp" \
  --has_isotopes 1 --gpus "$GPUS" \
  2>&1 | tee "$OUT_DIR/06_predict.log"

python cosine-similarity.py "$OUT_DIR/pred.msp" "$TEST_MSP" --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine.log"

echo "Done: $OUT_DIR/"
