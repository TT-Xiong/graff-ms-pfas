#!/usr/bin/env python3
"""
Check whether predicted spectra satisfy CE monotonicity trends for the same PFAS molecule.

Expected trends as CE increases (15 -> 30 -> 45 -> 60 eV):
  - low_mass_fraction (m/z < 0.5 * precursor_mz) increases
  - high_mass_fraction (m/z > 0.8 * precursor_mz) decreases
  - normalized_centroid decreases
  - precursor_fraction (|m/z - precursor| <= tol) decreases

Usage on server (counterfactual 4-CE prediction per train molecule):

  CKPT=$(cat output/exp_ce_weighted_v1.2/best_ckpt.txt)
  python scripts/analyze_ce_monotonicity.py export-queries \\
      --pkl data/pfas/nist_标注/nist_pfas_annot.pkl --split train \\
      --dissociation HCD --output data/pfas/nist_标注/queries/train_4ce_hcd.tsv

  python run-graff-ms.py "$CKPT" data/pfas/nist_标注/queries/train_4ce_hcd.tsv \\
      output/ce_mono/pred_train_4ce_hcd.msp --has_isotopes 1 --gpus 1

  python scripts/analyze_ce_monotonicity.py report \\
      --pred-msp output/ce_mono/pred_train_4ce_hcd.msp \\
      --pkl data/pfas/nist_标注/nist_pfas_annot.pkl \\
      --dissociation HCD --output-csv output/ce_mono/train_hcd_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io import read_msp  # noqa: E402

ce_bins = [15, 30, 45, 60]
dissociation_id = {"HCD": 0, "CID": 1}

CE_ORDER = [0, 1, 2, 3]
QUERY_SEP = "|||"
METRICS = {
    "low_mass_fraction": +1,
    "high_mass_fraction": -1,
    "normalized_centroid": -1,
    "precursor_fraction": -1,
}


def spectrum_features(mzs, intensities, precursor_mz: float, *, precursor_tol: float = 0.1) -> dict:
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    total = float(intensities.sum())
    if total <= 0 or len(mzs) == 0:
        return {k: np.nan for k in METRICS}

    prec = float(precursor_mz)
    low = float(intensities[mzs < 0.5 * prec].sum() / total)
    high = float(intensities[mzs > 0.8 * prec].sum() / total)
    centroid = float((mzs * intensities).sum() / total / prec)
    precursor = float(intensities[np.abs(mzs - prec) <= precursor_tol].sum() / total)
    return {
        "low_mass_fraction": low,
        "high_mass_fraction": high,
        "normalized_centroid": centroid,
        "precursor_fraction": precursor,
    }


def monotonicity_stats(values_by_ce: np.ndarray, direction: int) -> dict:
    """values_by_ce shape (4,) in CE order 0..3."""
    signed = values_by_ce * direction
    diffs = np.diff(signed)
    result = spearmanr(CE_ORDER, values_by_ce * direction)
    rho = getattr(result, "statistic", getattr(result, "correlation", np.nan))
    return {
        "strict_monotone": bool(np.all(diffs >= 0)),
        "transition_ok_frac": float(np.mean(diffs >= 0)),
        "endpoint_ok": bool(signed[-1] >= signed[0]),
        "spearman_rho": float(rho) if rho == rho else np.nan,
    }


def export_queries(
    pkl_path: Path,
    output_path: Path,
    *,
    split: str,
    dissociation: str,
) -> None:
    df = pd.read_pickle(pkl_path)
    part = df.query(f'split=="{split}"').copy()
    dissociation = dissociation.upper()
    if dissociation not in dissociation_id:
        raise ValueError(f"Unknown dissociation: {dissociation}")

    base = (
        part.groupby(["InChIKey", "Precursor_type"], as_index=False)
        .agg(SMILES=("SMILES", "first"), PrecursorMZ=("PrecursorMZ", "first"))
    )
    rows = []
    for row in base.itertuples(index=False):
        for ce_id in CE_ORDER:
            rows.append(
                {
                    "Spectrum": (
                        f"{row.InChIKey}{QUERY_SEP}{row.Precursor_type}"
                        f"{QUERY_SEP}{dissociation}{QUERY_SEP}CE{ce_id}"
                    ),
                    "SMILES": row.SMILES,
                    "Precursor_type": row.Precursor_type,
                    "CE_ID": ce_id,
                    "Dissociation_type": dissociation,
                }
            )
    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep="\t", header=False, index=False)
    print(f"Exported {len(out)} queries ({len(base)} molecules x 4 CE) -> {output_path}")


def _group_key_from_pred(row, pkl_lut: pd.DataFrame) -> tuple | None:
    spec = str(row.Spectrum)
    if spec in pkl_lut.index:
        meta = pkl_lut.loc[spec]
        return (meta.InChIKey, meta.Precursor_type, int(meta.CE_ID))

    # Counterfactual query id: InChIKey|||PrecursorType|||DISS|||CE#
    if QUERY_SEP in spec:
        parts = spec.split(QUERY_SEP)
        if len(parts) != 4 or not parts[3].startswith("CE"):
            return None
        inchikey, precursor_type, diss, ce_tag = parts
        return (inchikey, precursor_type, int(ce_tag.replace("CE", "")))

    parts = spec.rsplit("_CE", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    prefix = parts[0]
    ce_id = int(parts[1])
    chunks = prefix.rsplit("_", 2)
    if len(chunks) != 3:
        return None
    inchikey, precursor_type, diss = chunks
    return (inchikey, precursor_type, ce_id)


def report(
    pred_msp: Path,
    pkl_path: Path,
    *,
    split: str | None,
    dissociation: str | None,
    target_msp: Path | None,
    output_csv: Path,
) -> pd.DataFrame:
    pred = read_msp(str(pred_msp))
    pred["Spectrum"] = pred["Spectrum"].astype(str)

    pkl = pd.read_pickle(pkl_path)
    if split:
        pkl = pkl.query(f'split=="{split}"').copy()
    pkl["Spectrum"] = pkl["Spectrum"].astype(str)
    pkl_lut = pkl.set_index("Spectrum")

    precursor_lut = {}
    if target_msp and target_msp.is_file():
        tgt = read_msp(str(target_msp))
        tgt["Spectrum"] = tgt["Spectrum"].astype(str)
        precursor_lut = tgt.set_index("Spectrum")["PrecursorMZ"].to_dict()
    elif "PrecursorMZ" in pred.columns:
        precursor_lut = pred.set_index("Spectrum")["PrecursorMZ"].to_dict()

    rows = []
    for row in pred.itertuples(index=False):
        key = _group_key_from_pred(row, pkl_lut)
        if key is None:
            continue
        inchikey, precursor_type, ce_id = key
        prec = precursor_lut.get(str(row.Spectrum))
        if prec is None or (isinstance(prec, float) and np.isnan(prec)):
            # fall back to any train spectrum for this molecule
            hits = pkl.query(
                "InChIKey == @inchikey and Precursor_type == @precursor_type"
            )
            if len(hits):
                prec = float(hits["PrecursorMZ"].iloc[0])
            else:
                continue
        feats = spectrum_features(row.mzs, row.intensities, float(prec))
        rows.append(
            {
                "InChIKey": inchikey,
                "Precursor_type": precursor_type,
                "CE_ID": ce_id,
                "eV": ce_bins[ce_id],
                **feats,
            }
        )

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        raise ValueError("No spectra parsed; check Spectrum ids or provide matching pkl split.")

    group_cols = ["InChIKey", "Precursor_type"]
    summaries = []
    for key, grp in feat_df.groupby(group_cols):
        if grp["CE_ID"].nunique() < 4:
            continue
        grp = grp.set_index("CE_ID").reindex(CE_ORDER)
        if grp.isna().any().any():
            continue
        rec = {"InChIKey": key[0], "Precursor_type": key[1], "n_ce": 4}
        metric_ok = []
        for metric, direction in METRICS.items():
            st = monotonicity_stats(grp[metric].values, direction)
            rec[f"{metric}_strict"] = st["strict_monotone"]
            rec[f"{metric}_transitions"] = st["transition_ok_frac"]
            rec[f"{metric}_endpoint"] = st["endpoint_ok"]
            rec[f"{metric}_rho"] = st["spearman_rho"]
            metric_ok.append(st["strict_monotone"])
        rec["all_metrics_strict"] = all(metric_ok)
        summaries.append(rec)

    out = pd.DataFrame(summaries)
    if out.empty:
        raise ValueError("No molecule groups with all 4 CE predictions.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    n = len(out)
    print("=" * 60)
    title = "CE monotonicity report"
    if dissociation:
        title += f" (Dissociation={dissociation.upper()})"
    if split:
        title += f" [{split}]"
    print(title)
    print(f"  molecule groups with 4 CE: {n}")
    print()
    for metric in METRICS:
        print(f"  {metric}:")
        print(f"    strict 4-point monotone : {out[f'{metric}_strict'].mean():.1%}")
        print(f"    mean transition OK rate : {out[f'{metric}_transitions'].mean():.1%}")
        print(f"    15->60 endpoint OK      : {out[f'{metric}_endpoint'].mean():.1%}")
        print(f"    mean Spearman rho       : {out[f'{metric}_rho'].mean():.3f}")
    print()
    print(f"  all 4 metrics strict      : {out['all_metrics_strict'].mean():.1%}")
    print(f"\nPer-molecule CSV -> {output_csv}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export-queries", help="Build 4-CE counterfactual query TSV.")
    exp.add_argument("--pkl", type=Path, required=True)
    exp.add_argument("--split", default="train")
    exp.add_argument("--dissociation", choices=["HCD", "CID"], default="HCD")
    exp.add_argument("--output", type=Path, required=True)

    rep = sub.add_parser("report", help="Summarize monotonicity from predicted MSP.")
    rep.add_argument("--pred-msp", type=Path, required=True)
    rep.add_argument("--pkl", type=Path, required=True)
    rep.add_argument("--split", default="train")
    rep.add_argument("--dissociation", choices=["HCD", "CID"], default=None)
    rep.add_argument("--target-msp", type=Path, default=None)
    rep.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "output/ce_monotonicity_summary.csv",
    )

    args = parser.parse_args()
    if args.cmd == "export-queries":
        export_queries(args.pkl, args.output, split=args.split, dissociation=args.dissociation)
    else:
        report(
            args.pred_msp,
            args.pkl,
            split=args.split,
            dissociation=args.dissociation,
            target_msp=args.target_msp,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()
