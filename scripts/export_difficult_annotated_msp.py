#!/usr/bin/env python3
"""Export poorly annotated PFAS MSP spectra for manual review."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_filtered_train_from_mgf import (  # noqa: E402
    intensity_weighted_coverage,
    is_fillable,
    msp_row_to_block,
)
from src.io import read_msp  # noqa: E402


def score_msp(path: Path) -> pd.DataFrame:
    df = read_msp(str(path))
    rows = []
    for row in df.itertuples(index=False):
        int_cov, peak_cov = intensity_weighted_coverage(row.intensities, row.annots)
        unannot_peaks = sum(1 for a in row.annots if is_fillable(a))
        rows.append(
            {
                "Spectrum": str(row.Spectrum),
                "InChIKey": row.InChIKey,
                "InChIKey14": str(row.InChIKey).split("-")[0],
                "SMILES": row.SMILES,
                "Formula": row.Formula,
                "Precursor_type": row.Precursor_type,
                "PrecursorMZ": row.PrecursorMZ,
                "Dissociation_type": getattr(row, "Dissociation_type", ""),
                "Dissociation_id": getattr(row, "Dissociation_id", ""),
                "NCE": getattr(row, "NCE", ""),
                "eV": getattr(row, "eV", ""),
                "Source_file": getattr(row, "Source_file", path.name),
                "intensity_coverage": int_cov,
                "peak_coverage": peak_cov,
                "n_peaks": len(row.annots),
                "n_unannotated_peaks": unannot_peaks,
                "mzs": row.mzs,
                "intensities": row.intensities,
                "annots": row.annots,
            }
        )
    return pd.DataFrame(rows)


def export_difficult(
    input_msp: Path,
    output_dir: Path,
    *,
    stem: str,
    min_intensity_coverage: float,
    split_label: str,
) -> pd.DataFrame:
    scored = score_msp(input_msp)
    scored["split"] = split_label
    difficult = scored[scored["intensity_coverage"] < min_intensity_coverage].copy()
    difficult = difficult.sort_values(["intensity_coverage", "Spectrum"])

    output_dir.mkdir(parents=True, exist_ok=True)
    msp_path = output_dir / f"{stem}.msp"
    csv_path = output_dir / f"{stem}.csv"
    stats_path = output_dir / f"{stem}.stats.txt"

    blocks = []
    for row in difficult.itertuples(index=False):
        blocks.append(
            msp_row_to_block(
                {
                    "Spectrum": row.Spectrum,
                    "SMILES": row.SMILES,
                    "InChIKey": row.InChIKey,
                    "Formula": row.Formula,
                    "Precursor_type": row.Precursor_type,
                    "PrecursorMZ": row.PrecursorMZ,
                    "Dissociation_type": row.Dissociation_type,
                    "Dissociation_id": row.Dissociation_id,
                    "NCE": row.NCE,
                    "eV": row.eV,
                    "Source_file": row.Source_file,
                    "mzs": row.mzs,
                    "intensities": row.intensities,
                    "annots": row.annots,
                }
            )
        )

    with open(msp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))

    manifest_cols = [
        "split",
        "Spectrum",
        "InChIKey14",
        "InChIKey",
        "Precursor_type",
        "Dissociation_type",
        "eV",
        "intensity_coverage",
        "peak_coverage",
        "n_peaks",
        "n_unannotated_peaks",
        "Source_file",
    ]
    difficult[manifest_cols].to_csv(csv_path, index=False)

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"input_msp={input_msp}\n")
        f.write(f"split={split_label}\n")
        f.write(f"threshold=intensity_coverage<{min_intensity_coverage}\n")
        f.write(f"n_input={len(scored)}\n")
        f.write(f"n_difficult={len(difficult)}\n")
        f.write(f"n_structures_inchikey14={difficult['InChIKey14'].nunique()}\n")
        f.write(f"mean_coverage_all={scored['intensity_coverage'].mean():.4f}\n")
        f.write(f"mean_coverage_difficult={difficult['intensity_coverage'].mean():.4f}\n")

    print(f"{split_label}: {len(difficult)}/{len(scored)} spectra, "
          f"{difficult['InChIKey14'].nunique()} structures (InChIKey14)")
    print(f"  -> {msp_path}")
    return difficult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "pfas" / "nist_标注" / "train.msp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "pfas" / "difficult",
    )
    parser.add_argument("--stem", default="difficult_mask")
    parser.add_argument(
        "--min-intensity-coverage",
        type=float,
        default=0.50,
        help="Export spectra with intensity-weighted annotation coverage below this value.",
    )
    parser.add_argument(
        "--include-splits",
        default="train",
        help="Comma-separated splits to scan under nist_标注/ (train,val,test).",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.include_splits.split(",") if s.strip()]
    annotated_dir = REPO_ROOT / "data" / "pfas" / "nist_标注"
    all_difficult = []

    for split in splits:
        msp_path = args.input if len(splits) == 1 and args.input.name.startswith(split) else annotated_dir / f"{split}.msp"
        if not msp_path.is_file():
            print(f"skip missing: {msp_path}")
            continue
        part = export_difficult(
            msp_path,
            args.output_dir,
            stem=args.stem if len(splits) == 1 else f"{args.stem}_{split}",
            min_intensity_coverage=args.min_intensity_coverage,
            split_label=split,
        )
        all_difficult.append(part)

    if len(all_difficult) > 1:
        combined = pd.concat(all_difficult, ignore_index=True)
        combined = combined.sort_values(["split", "intensity_coverage", "Spectrum"])
        combined_path = args.output_dir / f"{args.stem}_all_splits.csv"
        combined[
            [
                "split",
                "Spectrum",
                "InChIKey14",
                "Precursor_type",
                "Dissociation_type",
                "eV",
                "intensity_coverage",
                "peak_coverage",
            ]
        ].to_csv(combined_path, index=False)
        print(
            f"\nCombined: {len(combined)} spectra, "
            f"{combined['InChIKey14'].nunique()} structures -> {combined_path}"
        )


if __name__ == "__main__":
    main()
