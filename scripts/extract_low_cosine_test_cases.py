#!/usr/bin/env python3
"""List lowest-cosine test spectra for case study (CE / dissociation / OOV)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnose-csv",
        type=Path,
        default=REPO_ROOT / "output" / "diagnose_fit_vocab_cosine.csv",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=REPO_ROOT / "data" / "pfas" / "nist_标注" / "nist_pfas_annot.pkl",
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--ce-id",
        type=int,
        default=None,
        help="If set, only include spectra with this CE_ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.output is None:
        suffix = args.split
        if args.ce_id is not None:
            suffix = f"{suffix}_ce{args.ce_id}"
        args.output = REPO_ROOT / "output" / f"{suffix}_low_cosine_top{args.top_n}.csv"

    if not args.diagnose_csv.is_file():
        raise FileNotFoundError(
            f"Missing {args.diagnose_csv}. Run diagnose_fit_vocab_cosine.py report first."
        )

    diag = pd.read_csv(args.diagnose_csv)
    subset = diag.query(f'split=="{args.split}"').copy()
    if len(subset) == 0:
        raise ValueError(f"No {args.split} rows in diagnose CSV")

    meta_cols = [
        "Spectrum",
        "InChIKey",
        "SMILES",
        "Precursor_type",
        "Dissociation_type",
        "Dissociation_id",
        "eV",
        "CE_ID",
        "CE_bin",
        "Formula",
        "PrecursorMZ",
    ]
    pkl = pd.read_pickle(args.pkl)
    meta = pkl.query(f'split=="{args.split}"')[meta_cols].copy()
    meta["Spectrum"] = meta["Spectrum"].astype(str)
    subset["Spectrum"] = subset["Spectrum"].astype(str)

    merged = subset.merge(meta, on="Spectrum", how="left", suffixes=("", "_meta"))
    if args.ce_id is not None:
        merged = merged.loc[merged["CE_ID"] == args.ce_id].copy()
        if len(merged) == 0:
            raise ValueError(f"No {args.split} spectra with CE_ID={args.ce_id}")
    merged["InChIKey14"] = merged["InChIKey"].astype(str).str.split("-").str[0]
    merged = merged.sort_values(["cosine", "intensity_oov"], ascending=[True, False])

    show_cols = [
        "Spectrum",
        "cosine",
        "InChIKey14",
        "Precursor_type",
        "Dissociation_type",
        "eV",
        "intensity_in_vocab",
        "intensity_oov",
        "peak_frac_in_vocab",
        "Formula",
    ]
    top = merged.head(args.top_n)[show_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.output, index=False)

    label = args.split
    if args.ce_id is not None:
        label = f"{label} CE_ID={args.ce_id}"
    print(f"{label} low-cosine Top {args.top_n} -> {args.output}\n")
    print(top.to_string(index=False))

    print("\n## By dissociation / CE (Top N)")
    summary = (
        top.groupby(["Dissociation_type", "eV"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print(summary.to_string(index=False))

    high_oov = int((top["intensity_oov"] > 0.15).sum())
    print(f"\nHigh OOV (>15% intensity) in Top {args.top_n}: {high_oov}")


if __name__ == "__main__":
    main()
