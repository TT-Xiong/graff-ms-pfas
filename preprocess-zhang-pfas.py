#!/usr/bin/env python3
"""
Preprocess Zhang OECD Orbitrap PFAS MGF for GrAFF-MS training.

Steps:
  1. Copy raw MGF into data/pfas/zhang/raw/
  2. Annotate via compositional enumeration (orbitrap -> HCD, continuous CE)
  3. Structure-disjoint split 80/10/10 into train/val/test
  4. Export MSP/TSV/pkl under data/pfas/zhang/
  5. Export legacy NIST PFAS set as external validation (no overlap in splits)

Usage:
  python preprocess-zhang-pfas.py \\
      --mgf "C:/Users/TT/Desktop/LC-MS data/zhang/PFAS_unprocessed_OECD_cleaned.mgf"

  python preprocess-zhang-pfas.py --mgf path/to/file.mgf --sample 200  # dry run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.annotate_pfas import (  # noqa: E402
    annotate_mgf_paths,
    export_artifacts,
    structure_disjoint_split,
)
from src.io import read_msp, write_msp  # noqa: E402

ZHANG_DIR = REPO_ROOT / "data" / "pfas" / "zhang"
DEFAULT_MGF = Path(r"C:\Users\TT\Desktop\LC-MS data\zhang\PFAS_unprocessed_OECD_cleaned.mgf")
DEFAULT_NIST_PKL = REPO_ROOT / "data" / "pfas" / "nist_标注" / "nist_pfas_annot.pkl"
ZHANG_PRECURSOR_TYPES = ("[M-H]-", "[M+H]+")
CE_CLIP = (10.0, 120.0)


def export_external_nist(pkl_path: Path, output_prefix: Path) -> None:
    """Export all legacy NIST PFAS spectra as external validation."""
    if not pkl_path.is_file():
        print(f"Skip external export: missing {pkl_path}", flush=True)
        return

    df = pd.read_pickle(pkl_path).copy()
    df["split"] = "external"
    df["dataset"] = "nist_pfas_external"

    tsv_cols = ["Spectrum", "SMILES", "Precursor_type", "CE_ID", "Dissociation_type"]
    msp_cols = tsv_cols + ["InChIKey", "Formula", "PrecursorMZ", "eV", "Dissociation_id"]

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = output_prefix.with_suffix(".tsv")
    msp_path = output_prefix.with_suffix(".msp")
    pkl_out = output_prefix.with_suffix(".pkl")

    df[tsv_cols].to_csv(tsv_path, sep="\t", header=False, index=False)
    write_msp(
        str(msp_path),
        df["mzs"].tolist(),
        df["intensities"].tolist(),
        **{c: df[c].tolist() for c in msp_cols},
    )
    df.to_pickle(pkl_out)
    print(
        f"External NIST export: {len(df)} spectra -> {tsv_path.name}, {msp_path.name}, {pkl_out.name}",
        flush=True,
    )


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in df.groupby("split"):
        rows.append(
            {
                "split": split,
                "n_spectra": len(part),
                "n_structures": part["InChIKey"].astype(str).str.split("-").str[0].nunique(),
                "precursor_[M-H]-": int((part["Precursor_type"] == "[M-H]-").sum()),
                "precursor_[M+H]+": int((part["Precursor_type"] == "[M+H]+").sum()),
                "ce_raw_min": float(part["CE_eV_raw"].min()),
                "ce_raw_median": float(part["CE_eV_raw"].median()),
                "ce_raw_max": float(part["CE_eV_raw"].max()),
                "annot_peak_frac": float(
                    part.apply(lambda r: len(r["peaks"]) / max(len(r["mzs"]), 1), axis=1).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", type=Path, default=DEFAULT_MGF)
    parser.add_argument("--output-dir", type=Path, default=ZHANG_DIR)
    parser.add_argument("--nist-pkl", type=Path, default=DEFAULT_NIST_PKL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--ppm", type=float, default=30.0)
    parser.add_argument("--sample", type=int, default=0, help="Process first N spectra only (0=all)")
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    if not args.mgf.is_file():
        raise FileNotFoundError(f"MGF not found: {args.mgf}")

    out_dir = args.output_dir
    raw_dir = out_dir / "raw"
    stats_dir = out_dir / "stats"
    queries_dir = out_dir / "queries"
    external_dir = out_dir / "external"
    for d in (raw_dir, stats_dir, queries_dir, external_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_copy = raw_dir / args.mgf.name
    if not raw_copy.exists() or raw_copy.stat().st_size != args.mgf.stat().st_size:
        shutil.copy2(args.mgf, raw_copy)
        print(f"Copied raw MGF -> {raw_copy}", flush=True)
    else:
        print(f"Raw MGF already present -> {raw_copy}", flush=True)

    print("Annotating Zhang OECD MGF (orbitrap->HCD, continuous CE)...", flush=True)
    df = annotate_mgf_paths(
        [str(args.mgf)],
        ppm_tolerance=args.ppm,
        dissociation_types=("HCD",),
        precursor_types=ZHANG_PRECURSOR_TYPES,
        ce_mode="continuous",
        ce_clip=CE_CLIP,
        sample=args.sample or None,
    )
    print(f"Annotated spectra: {len(df)}", flush=True)

    df = structure_disjoint_split(
        df,
        seed=args.seed,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
    )
    df["dataset"] = "zhang_oecd_orbitrap"
    df["Instrument"] = "Orbitrap"
    df["Instrument_type"] = "orbitrap"

    prefix = out_dir / "zhang_pfas"
    pkl_path = export_artifacts(df, str(prefix), verbose=True)

    for split in ("train", "val", "test"):
        part = df.query(f'split=="{split}"')
        q = queries_dir / f"{split}.tsv"
        cols = ["Spectrum", "SMILES", "Precursor_type", "CE_ID", "Dissociation_type"]
        part[cols].to_csv(q, sep="\t", header=False, index=False)

    summary = summarize_split(df)
    summary_path = stats_dir / "split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n## Split summary")
    print(summary.to_string(index=False))
    print(f"\nSaved {summary_path}", flush=True)

    overlap = set()
    if args.nist_pkl.is_file():
        old = pd.read_pickle(args.nist_pkl)
        new_keys = set(df["InChIKey"].astype(str).str.split("-").str[0])
        old_keys = set(old["InChIKey"].astype(str).str.split("-").str[0])
        overlap = new_keys & old_keys
    meta = {
        "mgf": str(args.mgf),
        "n_annotated": int(len(df)),
        "ce_clip": CE_CLIP,
        "ce_encoding_train": "continuous",
        "ce_max_ev_train": 120.0,
        "precursor_types": list(ZHANG_PRECURSOR_TYPES),
        "split_seed": args.seed,
        "train_frac": args.train_frac,
        "test_frac": args.test_frac,
        "inchikey14_overlap_with_nist": sorted(overlap),
    }
    meta_path = stats_dir / "preprocess_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {meta_path}", flush=True)

    if not args.skip_external:
        export_external_nist(args.nist_pkl, external_dir / "nist_pfas_external")

    print("\nDone.")
    print(f"Training pickle: {pkl_path}")
    print("Next on server:")
    print("  bash scripts/run_exp_zhang_continuous_v1.2.sh")


if __name__ == "__main__":
    main()
