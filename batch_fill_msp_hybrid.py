#!/usr/bin/env python3
"""Hybrid MSP fill: sub-formula enumeration first, then MAGMa for remaining ``?``."""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_fill_msp_annotations import process_msp as process_enum
from batch_fill_msp_magma import process_msp as process_magma


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fill PFAS MSP: enum (broad) then MAGMa (structure-aware) for leftover ? peaks."
    )
    p.add_argument("msp", nargs="+", type=Path)
    p.add_argument("--enum-ppm", type=float, default=30.0)
    p.add_argument("--enum-da-tol", type=float, default=0.02)
    p.add_argument(
        "--enum-match-mode",
        choices=("ppm", "da", "hybrid"),
        default="hybrid",
    )
    p.add_argument("--magma-ppm", type=float, default=30.0)
    p.add_argument("--min-relative-intensity", type=float, default=0.01)
    p.add_argument("--max-candidates-per-peak", type=int, default=3)
    p.add_argument("--max-tree-depth", type=int, default=4)
    p.add_argument("--max-broken-bonds", type=int, default=8)
    p.add_argument("--spectrum-ids", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-dir", type=Path, default=None)
    args = p.parse_args()

    from batch_fill_msp_annotations import parse_spectrum_id_list

    sid_filter = parse_spectrum_id_list(args.spectrum_ids)

    for msp_path in args.msp:
        msp_path = msp_path.resolve()
        report_dir = args.report_dir or msp_path.parent

        print(f"\n=== Step 1/2: sub-formula enum -> {msp_path.name} ===")
        process_enum(
            msp_path,
            ppm=args.enum_ppm,
            da_tol=args.enum_da_tol,
            match_mode=args.enum_match_mode,
            min_relative_intensity=args.min_relative_intensity,
            max_candidates_per_peak=args.max_candidates_per_peak,
            spectrum_ids=sid_filter,
            dry_run=args.dry_run,
            report_path=report_dir / f"{msp_path.stem}_enum_report.csv",
        )

        print(f"\n=== Step 2/2: MAGMa -> {msp_path.name} ===")
        process_magma(
            msp_path,
            ppm=args.magma_ppm,
            min_relative_intensity=args.min_relative_intensity,
            max_candidates_per_peak=args.max_candidates_per_peak,
            max_tree_depth=args.max_tree_depth,
            max_broken_bonds=args.max_broken_bonds,
            spectrum_ids=sid_filter,
            dry_run=args.dry_run,
            report_path=report_dir / f"{msp_path.stem}_magma_report.csv",
        )


if __name__ == "__main__":
    main()
