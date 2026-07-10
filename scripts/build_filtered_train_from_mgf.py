#!/usr/bin/env python3
"""
Build a quality-filtered PFAS train MSP from the full train MGF.

Phase export: merge full MGF with existing annotated train.msp (by Spectrum ID),
               new spectra start with ``?`` peak comments for hybrid fill.
Phase filter: drop poorly annotated spectra while keeping >=1 per condition group
              and >=1 per structure (InChIKey14).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyteomics.mass import Composition
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io import read_msp  # noqa: E402

PRECURSOR_TYPES = ["[M-H]-", "[M]+", "[M+H]+", "[M-2H]-"]
DISSOCIATION_TYPES = ["HCD", "CID"]
DISSOCIATION_ID = {"HCD": 0, "CID": 1}
CE_BINS = [15, 30, 45, 60]
CE_ID = {15: 0, 30: 1, 45: 2, 60: 3}

_MGF_META_RE = re.compile(r"^([A-Za-z0-9 _]+?)=(.*)$")
_MGF_PEAK_RE = re.compile(r"^(\d+\.?\d*)\s+(\d+\.?\d*)\s*$")
PEAK_LINE_RE = re.compile(r"^(\d+\.?\d*)\s+(\d+\.?\d*)(?:\s+(.+))?$")


def assign_ce_bin(ev: float, *, tolerance: float = 0.5) -> Optional[int]:
    if ev is None or (isinstance(ev, float) and np.isnan(ev)):
        return None
    best = min(CE_BINS, key=lambda b: abs(b - ev))
    if abs(best - ev) <= tolerance:
        return best
    return None


def parse_collision_energy(raw, precursor_mz: float) -> Tuple[float, float]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return float("nan"), float("nan")
    text = str(raw).strip()
    if not text:
        return float("nan"), float("nan")
    ev = float(text)
    nce = ev * 500.0 / precursor_mz if precursor_mz > 0 else float("nan")
    return ev, nce


def _parse_mgf_block(block: str) -> Optional[dict]:
    block = block.strip()
    if not block:
        return None
    meta: Dict[str, str] = {}
    mzs: List[float] = []
    intensities: List[float] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line in ("BEGIN IONS", "END IONS"):
            continue
        m_peak = _MGF_PEAK_RE.match(line)
        if m_peak:
            mzs.append(float(m_peak.group(1)))
            intensities.append(float(m_peak.group(2)))
            continue
        m_meta = _MGF_META_RE.match(line)
        if m_meta:
            key = m_meta.group(1).strip().upper().replace(" ", "_")
            meta[key] = m_meta.group(2).strip()
    if not mzs:
        return None
    return {**meta, "mzs": np.asarray(mzs, dtype=np.float32), "intensities": np.asarray(intensities, dtype=np.float32)}


def read_mgf(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    blocks = re.split(r"(?m)^BEGIN IONS\s*$", text)
    records = []
    for block in blocks:
        if "END IONS" in block:
            block = block.split("END IONS")[0]
        rec = _parse_mgf_block(block)
        if rec is None:
            continue
        rec["filename"] = os.path.basename(path)
        records.append(rec)
    if not records:
        raise ValueError(f"No spectra parsed from {path}")
    df = pd.DataFrame.from_records(records)
    df = df.rename(
        columns={
            "CANONICALSMILES": "SMILES",
            "INCHIKEY": "InChIKey",
            "PRECURSOR_TYPE": "Precursor_type",
            "PEPMASS": "PrecursorMZ",
            "COLLISION_ENERGY": "Collision_energy",
            "SOURCE_INSTRUMENT": "Dissociation_type",
            "MS_DATA_ID": "Spectrum",
            "TITLE": "Title",
        }
    )
    if "Spectrum" not in df.columns:
        df["Spectrum"] = df.get("Title", df.index.astype(str))
    df["Spectrum"] = df["Spectrum"].astype(str)
    return df


def is_fillable(annot: str) -> bool:
    a = (annot or "").strip().strip('"')
    return not a or a in ("?", "more")


def intensity_weighted_coverage(intensities: Sequence[float], annots: Sequence[str]) -> Tuple[float, float]:
    intens = np.asarray(intensities, dtype=np.float64)
    if len(intens) == 0:
        return 0.0, 0.0
    total = intens.sum()
    if total <= 0:
        total = len(intens)
        intens = np.ones_like(intens)
    mask = np.array([not is_fillable(a) for a in annots], dtype=bool)
    peak_cov = float(mask.mean())
    int_cov = float(intens[mask].sum() / total) if mask.any() else 0.0
    return int_cov, peak_cov


def condition_group_key(row) -> str:
    inchi14 = str(row["InChIKey"]).split("-")[0]
    diss = str(row["Dissociation_type"]).upper()
    ce = int(row["CE_bin"])
    return f"{inchi14}|{row['Precursor_type']}|{diss}|{ce}"


def normalize_mgf_frame(df: pd.DataFrame, *, source_file: str) -> pd.DataFrame:
    out = df.copy()
    out["PrecursorMZ"] = pd.to_numeric(out["PrecursorMZ"], errors="coerce")
    out["Dissociation_type"] = out["Dissociation_type"].astype(str).str.upper()
    out = out[out["Precursor_type"].isin(PRECURSOR_TYPES)]
    out = out[out["Dissociation_type"].isin(DISSOCIATION_TYPES)]
    out = out[out["PrecursorMZ"].notna() & (out["PrecursorMZ"] <= 1000)]
    out = out[out["SMILES"].notna() & (out["SMILES"].astype(str).str.len() > 0)]
    out = out[~out["SMILES"].astype(str).str.contains(r"\.")]

    ev_nce = [
        parse_collision_energy(ce, float(pmz))
        for ce, pmz in zip(out["Collision_energy"], out["PrecursorMZ"])
    ]
    out["eV"] = [x[0] for x in ev_nce]
    out["NCE"] = [x[1] for x in ev_nce]
    out["CE_bin"] = out["eV"].map(lambda x: assign_ce_bin(x))
    out["CE_ID"] = out["CE_bin"].map(lambda x: CE_ID.get(x) if x is not None else None)
    out["Dissociation_id"] = out["Dissociation_type"].map(DISSOCIATION_ID)
    out = out[out["CE_bin"].notna()].copy()

    out["Formula"] = out["SMILES"].map(
        lambda s: CalcMolFormula(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else None
    )
    out = out[out["Formula"].notna()].copy()
    out["InChIKey14"] = out["InChIKey"].astype(str).str.split("-").str[0]
    out["Source_file"] = source_file
    out["condition_group"] = out.apply(condition_group_key, axis=1)
    return out.reset_index(drop=True)


def msp_row_to_block(row) -> str:
    meta = {
        "Name": str(row["Spectrum"]),
        "Spectrum": str(row["Spectrum"]),
        "SMILES": row["SMILES"],
        "InChIKey": row["InChIKey"],
        "Formula": row["Formula"],
        "Precursor_type": row["Precursor_type"],
        "PrecursorMZ": f"{float(row['PrecursorMZ']):.6f}",
        "Dissociation_type": row["Dissociation_type"],
        "Dissociation_id": str(int(row["Dissociation_id"])),
        "NCE": f"{float(row['NCE']):.4f}",
        "eV": str(int(row.get("CE_bin", row.get("eV", 0)))),
        "Source_file": row.get("Source_file", row.get("filename", "")),
    }
    lines = [f"{k}: {v}" for k, v in meta.items()]
    mzs = np.asarray(row["mzs"], dtype=np.float64)
    intens = np.asarray(row["intensities"], dtype=np.float64)
    annots = row.get("annots", np.array(["?"] * len(mzs), dtype=object))
    if len(mzs) > 1:
        idx = np.argsort(mzs)
        mzs, intens, annots = mzs[idx], intens[idx], np.asarray(annots)[idx]
    if intens.max() > 0:
        intens_norm = intens / intens.max() * 999.0
    else:
        intens_norm = intens
    for mz, inten, annot in zip(mzs, intens_norm, annots):
        line = f"{mz:.5f} {inten:.2f}"
        a = str(annot).strip().strip('"')
        if a:
            line += f' "{a}"'
        lines.append(line)
    return "\n".join(lines) + "\n"


def load_seed_msp_blocks(seed_msp: Path) -> Dict[str, dict]:
    df = read_msp(str(seed_msp))
    blocks: Dict[str, dict] = {}
    for row in df.itertuples(index=False):
        sid = str(row.Spectrum)
        blocks[sid] = {
            "Spectrum": sid,
            "SMILES": row.SMILES,
            "InChIKey": row.InChIKey,
            "Formula": row.Formula,
            "Precursor_type": row.Precursor_type,
            "PrecursorMZ": row.PrecursorMZ,
            "Dissociation_type": getattr(row, "Dissociation_type", None),
            "Dissociation_id": getattr(row, "Dissociation_id", None),
            "NCE": getattr(row, "NCE", None),
            "eV": getattr(row, "eV", None),
            "Source_file": getattr(row, "Source_file", seed_msp.name),
            "mzs": row.mzs,
            "intensities": row.intensities,
            "annots": row.annots,
        }
    return blocks


def export_full_train_msp(
    mgf_path: Path,
    seed_msp: Path,
    output_msp: Path,
) -> pd.DataFrame:
    mgf_df = normalize_mgf_frame(read_mgf(str(mgf_path)), source_file=mgf_path.name)
    seed_blocks = load_seed_msp_blocks(seed_msp)

    reused = 0
    new_skeleton = 0
    blocks: List[str] = []
    manifest_rows = []

    for row in mgf_df.itertuples(index=False):
        sid = str(row.Spectrum)
        if sid in seed_blocks:
            seed = seed_blocks[sid]
            block = msp_row_to_block(seed)
            reused += 1
            source = "seed_msp"
            annots = seed["annots"]
        else:
            rec = row._asdict()
            rec["annots"] = np.array(["?"] * len(row.mzs), dtype=object)
            block = msp_row_to_block(rec)
            new_skeleton += 1
            source = "mgf_new"
            annots = rec["annots"]

        int_cov, peak_cov = intensity_weighted_coverage(row.intensities, annots)
        manifest_rows.append(
            {
                "Spectrum": sid,
                "InChIKey14": row.InChIKey14,
                "condition_group": row.condition_group,
                "source": source,
                "intensity_coverage": int_cov,
                "peak_coverage": peak_cov,
                "n_peaks": len(row.mzs),
            }
        )
        blocks.append(block)

    output_msp.parent.mkdir(parents=True, exist_ok=True)
    with open(output_msp, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_msp.with_suffix(".manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"Wrote {output_msp} ({len(blocks)} spectra)", flush=True)
    print(f"  reused annotated from seed: {reused}", flush=True)
    print(f"  new skeleton (? peaks):     {new_skeleton}", flush=True)
    print(f"  manifest: {manifest_path}", flush=True)
    print(
        f"  structures={mgf_df['InChIKey14'].nunique()} "
        f"condition_groups={mgf_df['condition_group'].nunique()}",
        flush=True,
    )
    return manifest


def filter_train_msp(
    input_msp: Path,
    output_msp: Path,
    *,
    min_intensity_coverage: float,
    max_per_group: int,
) -> pd.DataFrame:
    df = read_msp(str(input_msp))
    df["Spectrum"] = df["Spectrum"].astype(str)
    df["InChIKey14"] = df["InChIKey"].astype(str).str.split("-").str[0]
    df["Dissociation_type"] = df["Dissociation_type"].astype(str).str.upper()
    df["CE_bin"] = pd.to_numeric(df["eV"], errors="coerce")
    df["condition_group"] = df.apply(
        lambda r: f"{r['InChIKey14']}|{r['Precursor_type']}|{r['Dissociation_type']}|{int(r['CE_bin'])}",
        axis=1,
    )

    scores = []
    for row in df.itertuples(index=False):
        int_cov, peak_cov = intensity_weighted_coverage(row.intensities, row.annots)
        inchi14 = str(row.InChIKey).split("-")[0]
        diss = str(getattr(row, "Dissociation_type", "")).upper()
        ce = int(float(getattr(row, "eV", 0)))
        scores.append(
            {
                "Spectrum": str(row.Spectrum),
                "InChIKey14": inchi14,
                "condition_group": f"{inchi14}|{row.Precursor_type}|{diss}|{ce}",
                "intensity_coverage": int_cov,
                "peak_coverage": peak_cov,
                "quality_score": int_cov,
            }
        )
    scored = pd.DataFrame(scores)

    kept_ids: set = set()
    group_reports = []

    for cg, group in scored.groupby("condition_group"):
        group = group.sort_values(["quality_score", "intensity_coverage"], ascending=False)
        passing = group[group["intensity_coverage"] >= min_intensity_coverage]
        if len(passing) == 0:
            selected = group.head(1)
            reason = "forced_best_in_group"
        else:
            selected = passing.head(max_per_group)
            reason = "passed_threshold"
        kept_ids.update(selected["Spectrum"].tolist())
        group_reports.append(
            {
                "condition_group": cg,
                "n_input": len(group),
                "n_kept": len(selected),
                "reason": reason,
                "best_coverage": float(group["intensity_coverage"].max()),
            }
        )

    # Ensure each structure keeps at least one spectrum.
    for struct, group in scored.groupby("InChIKey14"):
        if not any(sid in kept_ids for sid in group["Spectrum"]):
            best = group.sort_values("quality_score", ascending=False).head(1)
            kept_ids.update(best["Spectrum"].tolist())

    kept_df = df[df["Spectrum"].isin(kept_ids)].copy()
    blocks = []
    for row in kept_df.itertuples(index=False):
        blocks.append(
            msp_row_to_block(
                {
                    "Spectrum": row.Spectrum,
                    "SMILES": row.SMILES,
                    "InChIKey": row.InChIKey,
                    "Formula": row.Formula,
                    "Precursor_type": row.Precursor_type,
                    "PrecursorMZ": row.PrecursorMZ,
                    "Dissociation_type": getattr(row, "Dissociation_type", ""),
                    "Dissociation_id": getattr(row, "Dissociation_id", 0),
                    "NCE": getattr(row, "NCE", 0),
                    "CE_bin": getattr(row, "eV", 0),
                    "Source_file": getattr(row, "Source_file", input_msp.name),
                    "mzs": row.mzs,
                    "intensities": row.intensities,
                    "annots": row.annots,
                }
            )
        )

    output_msp.parent.mkdir(parents=True, exist_ok=True)
    with open(output_msp, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))

    report_path = output_msp.with_suffix(".filter_report.csv")
    pd.DataFrame(group_reports).to_csv(report_path, index=False)
    summary_path = output_msp.with_suffix(".filter_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"input_spectra={len(df)}\n")
        f.write(f"kept_spectra={len(kept_df)}\n")
        f.write(f"structures_in={scored['InChIKey14'].nunique()}\n")
        f.write(f"structures_kept={kept_df['InChIKey'].astype(str).str.split('-').str[0].nunique()}\n")
        f.write(f"condition_groups_in={scored['condition_group'].nunique()}\n")
        f.write(f"condition_groups_kept={kept_df['condition_group'].nunique()}\n")
        f.write(f"min_intensity_coverage={min_intensity_coverage}\n")
        f.write(f"max_per_group={max_per_group}\n")

    print(f"Filtered {len(df)} -> {len(kept_df)} spectra", flush=True)
    print(f"  structures kept: {kept_df['InChIKey'].astype(str).str.split('-').str[0].nunique()}", flush=True)
    print(f"  report: {report_path}", flush=True)
    print(f"  summary: {summary_path}", flush=True)
    return kept_df


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="MGF + seed MSP -> full train MSP")
    p_export.add_argument("--mgf", type=Path, required=True)
    p_export.add_argument("--seed-msp", type=Path, required=True)
    p_export.add_argument("--output", type=Path, required=True)

    p_filter = sub.add_parser("filter", help="Quality-filter an annotated MSP")
    p_filter.add_argument("--input", type=Path, required=True)
    p_filter.add_argument("--output", type=Path, required=True)
    p_filter.add_argument("--min-intensity-coverage", type=float, default=0.35)
    p_filter.add_argument("--max-per-group", type=int, default=10)

    args = parser.parse_args(argv)
    if args.command == "export":
        export_full_train_msp(args.mgf, args.seed_msp, args.output)
    elif args.command == "filter":
        filter_train_msp(
            args.input,
            args.output,
            min_intensity_coverage=args.min_intensity_coverage,
            max_per_group=args.max_per_group,
        )


if __name__ == "__main__":
    main()
