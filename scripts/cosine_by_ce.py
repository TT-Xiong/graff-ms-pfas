#!/usr/bin/env python3
"""Per CE bucket cosine summary from diagnose CSV + pkl metadata."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import ci95  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    diag = pd.read_csv(args.diagnose_csv)
    diag = diag.query(f'split=="{args.split}"').copy()
    meta = pd.read_pickle(args.pkl)
    meta = meta.query(f'split=="{args.split}"')[["Spectrum", "CE_ID", "eV", "Dissociation_type"]]
    meta["Spectrum"] = meta["Spectrum"].astype(str)
    diag["Spectrum"] = diag["Spectrum"].astype(str)
    merged = diag.merge(meta, on="Spectrum", how="left")

    print(f"## Cosine by CE ({args.split}, N={len(merged)})")
    rows = []
    for ce_id, grp in merged.groupby("CE_ID", sort=True):
        ev = int(grp["eV"].iloc[0])
        m, err = ci95(grp["cosine"].values)
        rows.append({"CE_ID": ce_id, "eV": ev, "N": len(grp), "mean": m, "err": err})
        print(f"  CE_ID={ce_id}  eV={ev:2d}  N={len(grp):3d}  mean={m:.3f} +- {err:.3f}")

    print("\n## Cosine by Dissociation x CE")
    for (diss, ev), grp in merged.groupby(["Dissociation_type", "eV"], sort=True):
        m = grp["cosine"].mean()
        print(f"  {diss:3s}  eV={int(ev):2d}  N={len(grp):3d}  mean={m:.3f}")


if __name__ == "__main__":
    main()
