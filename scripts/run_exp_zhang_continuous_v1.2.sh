#!/usr/bin/env bash
# Train GrAFF on Zhang OECD Orbitrap PFAS (continuous CE, ~5k train spectra).
#
# Prerequisites:
#   python preprocess-zhang-pfas.py --mgf /path/to/PFAS_unprocessed_OECD_cleaned.mgf
#
# Usage:
#   bash scripts/run_exp_zhang_continuous_v1.2.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKL="${PKL:-data/pfas/zhang/zhang_pfas.pkl}"
NIST_CKPT="${NIST_CKPT:-lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt}"
QUERIES_DIR="${QUERIES_DIR:-data/pfas/zhang/queries}"
EXTERNAL_TSV="${EXTERNAL_TSV:-data/pfas/zhang/external/nist_pfas_external.tsv}"
OUT_DIR="${OUT_DIR:-output/exp_zhang_continuous_v1.2}"
GPUS="${GPUS:-1}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$PKL" ]]; then
  echo "Missing $PKL — run preprocess-zhang-pfas.py first." >&2
  exit 1
fi

echo "=== [1/6] train Zhang OECD + continuous CE ==="
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
  --ce_encoding continuous \
  --ce_max_ev 120 \
  --ce_clip_min 10 \
  --ce_clip_max 120 \
  2>&1 | tee "$OUT_DIR/train.log"

CKPT="$(ls -t lightning_logs/graff/version_*/checkpoints/*.ckpt | head -1)"
echo "$CKPT" > "$OUT_DIR/best_ckpt.txt"
echo "Best checkpoint: $CKPT"

echo "=== [2/6] predict train / val / test ==="
for s in train val test; do
  python run-graff-ms.py "$CKPT" "${QUERIES_DIR}/${s}.tsv" \
    "$OUT_DIR/pred_${s}.msp" --has_isotopes 1 --gpus "$GPUS"
done

echo "=== [3/6] in-domain cosine (zhang test) ==="
python cosine-similarity.py "$OUT_DIR/pred_test.msp" \
  data/pfas/zhang/zhang_pfas_test.msp --matchms_tol 0.1 \
  2>&1 | tee "$OUT_DIR/cosine_test.log"

echo "=== [4/6] external validation (legacy NIST PFAS) ==="
if [[ -f "$EXTERNAL_TSV" ]]; then
  python run-graff-ms.py "$CKPT" "$EXTERNAL_TSV" \
    "$OUT_DIR/pred_external.msp" --has_isotopes 1 --gpus "$GPUS"
  python cosine-similarity.py "$OUT_DIR/pred_external.msp" \
    data/pfas/zhang/external/nist_pfas_external.msp --matchms_tol 0.1 \
    2>&1 | tee "$OUT_DIR/cosine_external.log"
else
  echo "Skip external: $EXTERNAL_TSV not found"
fi

echo "=== [5/6] diagnose ==="
python scripts/diagnose_fit_vocab_cosine.py report \
  --pkl "$PKL" --checkpoint "$CKPT" \
  --target-msp-dir data/pfas/zhang \
  --pred-train "$OUT_DIR/pred_train.msp" \
  --pred-val "$OUT_DIR/pred_val.msp" \
  --pred-test "$OUT_DIR/pred_test.msp" \
  2>&1 | tee "$OUT_DIR/diagnose.log"

echo "=== [6/6] cosine by CE (test) ==="
python scripts/cosine_by_ce.py \
  --diagnose-csv output/diagnose_fit_vocab_cosine.csv \
  --pkl "$PKL" --split test \
  2>&1 | tee "$OUT_DIR/cosine_by_ce_test.log"

echo "Done: $OUT_DIR/"
