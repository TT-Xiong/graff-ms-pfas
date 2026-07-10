#!/usr/bin/env bash
# M2: v1.2 film_decoder + mild CE loss weights + learnable CE embedding.
#
# Replaces CE one-hot (4-d) with nn.Embedding(ce_embed_dim) concatenated to covariates.
# Default: film_decoder, ce_embed_dim=32, CE weights 0:3.0,1:0.8,2:0.5,3:2.5
#
# Targets: test >= 0.52; train 15eV >= 0.45; test 15eV >= 0.38
#
# Usage:
#   bash scripts/run_exp_ce_embed_v1.2.sh
#   CE_EMBED_DIM=64 bash scripts/run_exp_ce_embed_v1.2.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/nist_标注/nist_pfas_annot.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
TEST_MSP="${TEST_MSP:-data/pfas/nist_标注/test.msp}"
QUERIES_DIR="${QUERIES_DIR:-data/pfas/nist_标注/queries}"
OUT_DIR="${OUT_DIR:-output/exp_ce_embed_v1.2}"
GPUS="${GPUS:-1}"
CE_EMBED_DIM="${CE_EMBED_DIM:-32}"
CE_LOSS_WEIGHTS="${CE_LOSS_WEIGHTS:-0:3.0,1:0.8,2:0.5,3:2.5}"

mkdir -p "$OUT_DIR"

echo "=== [1/5] train v1.2 + CE embed (dim=$CE_EMBED_DIM) ==="
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
  --cov_conditioning film_decoder \
  --ce_embed_dim "$CE_EMBED_DIM" \
  --ce_loss_weights "$CE_LOSS_WEIGHTS" \
  2>&1 | tee "$OUT_DIR/train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"
echo "Best checkpoint: $CKPT"

echo "=== [2/5] export queries ==="
python scripts/diagnose_fit_vocab_cosine.py export-queries \
  --pkl "$PKL" --output-dir "$QUERIES_DIR"

echo "=== [3/5] predict train / val / test ==="
for s in train val test; do
  python run-graff-ms.py "$CKPT" "${QUERIES_DIR}/${s}.tsv" \
    "$OUT_DIR/pred_${s}.msp" --has_isotopes 1 --gpus "$GPUS"
done

echo "=== [4/5] cosine + diagnose ==="
python cosine-similarity.py "$OUT_DIR/pred_test.msp" "$TEST_MSP" --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine_test.log"

python scripts/diagnose_fit_vocab_cosine.py report \
  --pkl "$PKL" --checkpoint "$CKPT" \
  --target-msp-dir data/pfas/nist_标注 \
  --pred-train "$OUT_DIR/pred_train.msp" \
  --pred-val "$OUT_DIR/pred_val.msp" \
  --pred-test "$OUT_DIR/pred_test.msp" \
  2>&1 | tee "$OUT_DIR/diagnose.log"

echo "=== [5/5] cosine by CE (train + test) ==="
for s in train test; do
  python scripts/cosine_by_ce.py \
    --diagnose-csv output/diagnose_fit_vocab_cosine.csv \
    --pkl "$PKL" --split "$s" \
    2>&1 | tee "$OUT_DIR/cosine_by_ce_${s}.log"
done

echo "=== low-cosine 15eV train cases ==="
python scripts/extract_low_cosine_test_cases.py \
  --split train --ce-id 0 --top-n 20 \
  --output "$OUT_DIR/train_low_cosine_ce15_top20.csv" \
  2>&1 | tee "$OUT_DIR/train_low_cosine_ce15.log"

echo "Done: $OUT_DIR/"
