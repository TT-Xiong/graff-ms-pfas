#!/usr/bin/env python3
"""
Diagnose overfitting and whether errors are vocab-limited vs intensity prediction.

Usage (on GPU server with best checkpoint):

  # 1) Export query TSVs from pkl (once)
  python scripts/diagnose_fit_vocab_cosine.py export-queries \\
      --pkl data/pfas/nist_标注/nist_pfas_annot.pkl \\
      --output-dir data/pfas/nist_标注/queries

  # 2) Predict train / val / test
  CKPT=lightning_logs/graff/version_*/checkpoints/epoch=*.ckpt
  for s in train val test; do
    python run-graff-ms.py "$CKPT" data/pfas/nist_标注/queries/${s}.tsv \\
      output/diagnose_${s}.msp --has_isotopes 1 --gpus 1
  done

  # 3) Full report
  python scripts/diagnose_fit_vocab_cosine.py report \\
      --pkl data/pfas/nist_标注/nist_pfas_annot.pkl \\
      --checkpoint "$CKPT" \\
      --target-msp-dir data/pfas/nist_标注 \\
      --pred-train output/diagnose_train.msp \\
      --pred-val output/diagnose_val.msp \\
      --pred-test output/diagnose_test.msp \\
      --matchms-tol 0.1
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io import read_msp  # noqa: E402
from src.metrics import ci95, ms_cosine_similarity  # noqa: E402


def _load_train_vocab_module():
    spec = importlib.util.spec_from_file_location("train_graff_ms", REPO_ROOT / "train-graff-ms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def export_queries(pkl_path: Path, output_dir: Path) -> None:
    df = pd.read_pickle(pkl_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["Spectrum", "SMILES", "Precursor_type", "CE_ID", "Dissociation_type"]
    for split in ("train", "val", "test"):
        part = df.query(f'split=="{split}"')
        if len(part) == 0:
            continue
        out = output_dir / f"{split}.tsv"
        part[cols].to_csv(out, sep="\t", header=False, index=False)
        print(f"{split}: {len(part)} queries -> {out}")


def load_union_vocab(checkpoint: Path, train_df: pd.DataFrame, pfas_extension_size: int = 2000):
    tv = _load_train_vocab_module()
    nist_vocab = tv.load_nist_vocab_from_checkpoint(str(checkpoint))
    extensions = tv.learn_extension_vocabulary(
        train_df, nist_vocab, max_extensions=pfas_extension_size,
    )
    vocab = tv.merge_union_vocabulary(nist_vocab, extensions)
    product_lut = {
        row.formula: i + 1 for i, row in vocab.iterrows() if row.kind == "product"
    }
    loss_lut = {
        row.formula: i + 1 for i, row in vocab.iterrows() if row.kind == "loss"
    }
    return vocab, product_lut, loss_lut


def spectrum_vocab_stats(row, product_lut: dict, loss_lut: dict) -> dict:
    """Fraction of spectrum intensity on peaks whose annotations hit the model vocab."""
    from collections import defaultdict

    intens = np.asarray(row.intensities, dtype=np.float64)
    total = intens.sum()
    if total <= 0:
        total = max(len(intens), 1)

    peak_info = defaultdict(lambda: {"prods": set(), "losses": set()})
    for peak_idx, prod, loss in zip(row.peaks, row.products, row.losses):
        peak_info[int(peak_idx)]["prods"].add(prod)
        peak_info[int(peak_idx)]["losses"].add(loss)

    in_vocab = 0.0
    oov = 0.0
    n_peaks = len(peak_info)
    n_in_vocab_peaks = 0

    for peak_idx, info in peak_info.items():
        inten = float(intens[peak_idx])
        pin = any(p in product_lut for p in info["prods"])
        lin = any(l in loss_lut for l in info["losses"])
        if pin or lin:
            in_vocab += inten
            n_in_vocab_peaks += 1
        else:
            oov += inten

    return {
        "intensity_in_vocab": in_vocab / total,
        "intensity_oov": oov / total,
        "peak_frac_in_vocab": n_in_vocab_peaks / n_peaks if n_peaks else 0.0,
        "n_annot_peaks": n_peaks,
    }


def cosine_for_split(pred_msp: Path, target_msp: Path, tol: float) -> pd.DataFrame:
    pred = read_msp(str(pred_msp))
    true = read_msp(str(target_msp))
    df = true.merge(pred, on="Spectrum", how="inner", suffixes=("_true", "_pred"))
    scores = []
    for row in df.itertuples(index=False):
        scores.append(
            ms_cosine_similarity(
                row.mzs_pred,
                row.intensities_pred,
                row.PrecursorMZ_pred,
                row.mzs_true,
                row.intensities_true,
                row.PrecursorMZ_true,
                tol,
            )
        )
    df = df.copy()
    df["cosine"] = scores
    return df[["Spectrum", "cosine"]]


def report(
    pkl_path: Path,
    checkpoint: Path,
    target_msp_dir: Path,
    preds: Dict[str, Path],
    *,
    matchms_tol: float,
    pfas_extension_size: int,
) -> None:
    df = pd.read_pickle(pkl_path)
    train_df = df.query('split=="train"')
    vocab, product_lut, loss_lut = load_union_vocab(
        checkpoint, train_df, pfas_extension_size=pfas_extension_size,
    )
    print(f"Model vocab: {len(vocab)} entries ({pfas_extension_size} PFAS ext assumed)")
    print("=" * 60)

    vocab_rows = []
    for split in ("train", "val", "test"):
        part = df.query(f'split=="{split}"')
        if len(part) == 0:
            continue
        for row in part.itertuples(index=False):
            st = spectrum_vocab_stats(row, product_lut, loss_lut)
            st.update({"Spectrum": str(row.Spectrum), "split": split})
            vocab_rows.append(st)
    vocab_df = pd.DataFrame(vocab_rows)

    print("\n## Vocabulary coverage on annotated peaks (ground truth labels)")
    for split in ("train", "val", "test"):
        sub = vocab_df.query(f'split=="{split}"')
        if len(sub) == 0:
            continue
        print(
            f"  {split:5s}  N={len(sub):4d}  "
            f"intensity_in_vocab={sub['intensity_in_vocab'].mean():.3f}  "
            f"intensity_oov={sub['intensity_oov'].mean():.3f}  "
            f"peak_frac_in_vocab={sub['peak_frac_in_vocab'].mean():.3f}"
        )

    print("\n## Cosine similarity (predicted vs annotated MSP)")
    cosine_frames = []
    for split, pred_path in preds.items():
        if pred_path is None or not pred_path.is_file():
            print(f"  {split:5s}  (no predictions: {pred_path})")
            continue
        target = target_msp_dir / f"{split}.msp"
        cdf = cosine_for_split(pred_path, target, matchms_tol)
        cdf["split"] = split
        cosine_frames.append(cdf)
        m, err = ci95(cdf["cosine"].values)
        print(f"  {split:5s}  mean={m:.3f} +- {err:.3f}  N={len(cdf)}")

    if not cosine_frames:
        print("\n(No prediction MSPs supplied — vocab section above still valid.)")
        return

    all_cos = pd.concat(cosine_frames, ignore_index=True)
    merged = all_cos.merge(vocab_df, on=["Spectrum", "split"], how="left")

    print("\n## Overfitting check (train vs val vs test cosine)")
    train_m = merged.query('split=="train"')["cosine"].mean()
    val_m = merged.query('split=="val"')["cosine"].mean() if (merged.split == "val").any() else float("nan")
    test_m = merged.query('split=="test"')["cosine"].mean()
    print(f"  train={train_m:.3f}  val={val_m:.3f}  test={test_m:.3f}")
    gap = train_m - test_m
    print(f"  train - test gap = {gap:.3f}", end="")
    if gap > 0.08:
        print("  -> possible overfitting")
    elif gap < 0.03:
        print("  -> little overfitting; gap mostly generalization / OOV")
    else:
        print("  -> moderate gap")

    print("\n## Diagnosis: vocab-limited vs intensity error")
    test = merged.query('split=="test"').copy()
    if len(test) == 0:
        return

    # Low OOV but low cosine -> intensity/shape error
    # High OOV -> vocab limitation
    test["oov_high"] = test["intensity_oov"] > 0.15
    test["cos_low"] = test["cosine"] < 0.45
    high_oov_low_cos = test[test["oov_high"] & test["cos_low"]]
    low_oov_low_cos = test[(~test["oov_high"]) & test["cos_low"]]

    print(f"  test spectra with cosine < 0.45: {test['cos_low'].sum()} / {len(test)}")
    print(f"    high OOV (>15% intensity) + low cosine: {len(high_oov_low_cos)}  [vocab likely binding]")
    print(f"    low OOV + low cosine:                  {len(low_oov_low_cos)}  [intensity/shape likely binding]")

    corr = test["cosine"].corr(test["intensity_in_vocab"])
    print(f"  corr(test cosine, in-vocab intensity fraction) = {corr:.3f}")
    if corr > 0.4:
        print("  -> vocabulary coverage explains a fair share of cosine variance")
    else:
        print("  -> cosine mostly driven by intensity prediction, not vocab OOV")

    out_csv = REPO_ROOT / "output" / "diagnose_fit_vocab_cosine.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"\nPer-spectrum details -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export-queries")
    p_export.add_argument("--pkl", type=Path, required=True)
    p_export.add_argument("--output-dir", type=Path, required=True)

    p_report = sub.add_parser("report")
    p_report.add_argument("--pkl", type=Path, required=True)
    p_report.add_argument("--checkpoint", type=Path, required=True)
    p_report.add_argument("--target-msp-dir", type=Path, required=True)
    p_report.add_argument("--pred-train", type=Path, default=None)
    p_report.add_argument("--pred-val", type=Path, default=None)
    p_report.add_argument("--pred-test", type=Path, default=None)
    p_report.add_argument("--matchms-tol", type=float, default=0.1)
    p_report.add_argument("--pfas-extension-size", type=int, default=2000)

    args = parser.parse_args()
    if args.command == "export-queries":
        export_queries(args.pkl, args.output_dir)
    elif args.command == "report":
        preds = {
            "train": args.pred_train,
            "val": args.pred_val,
            "test": args.pred_test,
        }
        report(
            args.pkl,
            args.checkpoint,
            args.target_msp_dir,
            preds,
            matchms_tol=args.matchms_tol,
            pfas_extension_size=args.pfas_extension_size,
        )


if __name__ == "__main__":
    main()
