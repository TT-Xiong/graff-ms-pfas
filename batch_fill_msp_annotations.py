#!/usr/bin/env python3
"""
Batch-fill PFAS MSP peak annotations using sub-formula enumeration.

Only replaces blank / ``?`` / ``more`` peak comments; existing manual labels
are preserved. Writes a ``.bak`` backup before overwriting unless ``--dry-run``.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

# annotate_pfas only needs constants from graff; avoid importing pytorch_lightning.
import sys
from types import ModuleType

_graff_stub = ModuleType("src.graff")
_graff_stub.atom_types = sorted(["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"])
_graff_stub.isotope_types = [0, 1, 2]
_graff_stub.neutron_mass = 1.008665
sys.modules["src.graff"] = _graff_stub

from src.annotate_pfas import annotate_spectrum, precursor_composition

PEAK_LINE_RE = re.compile(r"^(\d+\.?\d*)\s+(\d+\.?\d*)(?:\s+(.+))?$")
SPECTRUM_ID_RE = re.compile(r"(\d+)\s*$")


def is_fillable(annot: str) -> bool:
    a = (annot or "").strip().strip('"')
    return not a or a in ("?", "more")


def coverage(annots: Sequence[str]) -> float:
    if not annots:
        return 0.0
    return sum(1 for a in annots if not is_fillable(a)) / len(annots)


def parse_spectrum_id(name: str) -> Optional[str]:
    if not name:
        return None
    m = SPECTRUM_ID_RE.search(name.strip())
    return m.group(1) if m else name.strip()


def group_auto_annotations(
    peaks: Sequence[int],
    products: Sequence[str],
    losses: Sequence[str],
    isotopes: Sequence[int],
) -> Dict[int, List[str]]:
    by_peak: Dict[int, List[str]] = defaultdict(list)
    seen: Dict[int, set] = defaultdict(set)

    for peak_idx, prod, loss, iso in zip(peaks, products, losses, isotopes):
        token = f"{prod}={loss}" if loss else f"{prod}="
        if iso != 0:
            token += f";i={iso}"
        if token in seen[peak_idx]:
            continue
        seen[peak_idx].add(token)
        by_peak[int(peak_idx)].append(token)

    return by_peak


def parse_msp_block(block: str) -> Tuple[Dict[str, List[str]], List[str], List[str], List[str]]:
    meta: Dict[str, List[str]] = defaultdict(list)
    mzs: List[str] = []
    intens: List[str] = []
    annots: List[str] = []

    for line in block.splitlines():
        if not line.strip():
            continue
        if ":" in line and not PEAK_LINE_RE.match(line):
            k, v = line.split(":", maxsplit=1)
            meta[k.strip()].append(v.strip())
            continue
        m = PEAK_LINE_RE.match(line)
        if m:
            mzs.append(m.group(1))
            intens.append(m.group(2))
            raw = m.group(3)
            if raw is None:
                annots.append("")
            else:
                annots.append(raw.strip().strip('"'))
            continue

    return meta, mzs, intens, annots


def neutral_formula(meta: Dict[str, List[str]], smiles: Optional[str]) -> Optional[str]:
    if "Formula" in meta and meta["Formula"]:
        formula = meta["Formula"][0].strip()
        formula = re.sub(r"[+\-]+$", "", formula)
        return formula
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.rdMolDescriptors.CalcMolFormula(mol)
    return None


def render_msp_block(
    meta: Dict[str, List[str]],
    mzs: Sequence[str],
    intens: Sequence[str],
    annots: Sequence[str],
) -> str:
    lines: List[str] = []
    for key, values in meta.items():
        for v in values:
            lines.append(f"{key}: {v}")
    for mz, inten, annot in zip(mzs, intens, annots):
        line = f"{mz} {inten}"
        if annot and not is_fillable(annot):
            line += f' "{annot}"'
        elif annot and annot not in ("", "?"):
            line += f' "{annot}"'
        elif annot == "?":
            line += ' "?"'
        lines.append(line)
    return "\n".join(lines)


def fill_block(
    block: str,
    *,
    ppm: float,
    da_tol: float,
    match_mode: str,
    min_relative_intensity: float,
    max_candidates_per_peak: int,
    spectrum_ids: Optional[set],
) -> Tuple[str, dict]:
    meta, mzs, intens, annots = parse_msp_block(block)
    name = meta.get("Name", [""])[0]
    sid = parse_spectrum_id(name)
    cov_before = coverage(annots)

    stats = {
        "spectrum_id": sid or "",
        "name": name,
        "n_peaks": len(annots),
        "cov_before": cov_before,
        "cov_after": cov_before,
        "n_filled": 0,
        "skipped": False,
        "reason": "",
    }

    if spectrum_ids is not None and sid not in spectrum_ids:
        stats["skipped"] = True
        stats["reason"] = "not_in_filter"
        return block, stats

    smiles = None
    for key in ("SMILES", "Smiles", "smiles"):
        if key in meta and meta[key]:
            smiles = meta[key][0]
            break

    precursor_type = meta.get("Precursor_type", meta.get("PrecursorType", [""]))[0]
    if not precursor_type:
        stats["skipped"] = True
        stats["reason"] = "missing_precursor_type"
        return block, stats

    formula = neutral_formula(meta, smiles)
    if not formula:
        stats["skipped"] = True
        stats["reason"] = "missing_formula"
        return block, stats

    try:
        prec_comp = precursor_composition(formula, precursor_type)
    except Exception as exc:
        stats["skipped"] = True
        stats["reason"] = f"precursor_error:{exc}"
        return block, stats

    mz_arr = np.asarray([float(x) for x in mzs], dtype=np.float64)
    inten_arr = np.asarray([float(x) for x in intens], dtype=np.float64)

    peaks, products, losses, isotopes = annotate_spectrum(
        mz_arr,
        inten_arr,
        prec_comp,
        ppm_tolerance=ppm,
        da_tolerance=da_tol,
        match_mode=match_mode,
        min_relative_intensity=min_relative_intensity,
        max_candidates_per_peak=max_candidates_per_peak,
    )
    by_peak = group_auto_annotations(peaks, products, losses, isotopes)

    new_annots = list(annots)
    n_filled = 0
    for i, annot in enumerate(new_annots):
        if not is_fillable(annot):
            continue
        auto = by_peak.get(i)
        if not auto:
            continue
        new_annots[i] = ";".join(auto)
        n_filled += 1

    stats["n_filled"] = n_filled
    stats["cov_after"] = coverage(new_annots)
    if n_filled == 0:
        stats["reason"] = "no_auto_matches"

    return render_msp_block(meta, mzs, intens, new_annots), stats


def process_msp(
    path: Path,
    *,
    ppm: float,
    da_tol: float,
    match_mode: str,
    min_relative_intensity: float,
    max_candidates_per_peak: int,
    spectrum_ids: Optional[set],
    dry_run: bool,
    report_path: Optional[Path],
) -> None:
    text = path.read_text(encoding="utf-8")
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]

    out_blocks: List[str] = []
    rows: List[dict] = []
    total_filled = 0
    cov_before_sum = 0.0
    cov_after_sum = 0.0

    for block in blocks:
        new_block, stats = fill_block(
            block,
            ppm=ppm,
            da_tol=da_tol,
            match_mode=match_mode,
            min_relative_intensity=min_relative_intensity,
            max_candidates_per_peak=max_candidates_per_peak,
            spectrum_ids=spectrum_ids,
        )
        out_blocks.append(new_block)
        rows.append(stats)
        if not stats["skipped"]:
            total_filled += stats["n_filled"]
            cov_before_sum += stats["cov_before"]
            cov_after_sum += stats["cov_after"]

    n_proc = sum(1 for r in rows if not r["skipped"])
    mean_before = cov_before_sum / n_proc if n_proc else 0.0
    mean_after = cov_after_sum / n_proc if n_proc else 0.0

    print(f"{path.name}: {len(blocks)} spectra, {n_proc} processed")
    print(f"  match: mode={match_mode}, ppm={ppm}, da={da_tol}")
    print(f"  peaks filled: {total_filled}")
    print(f"  mean coverage: {mean_before:.1%} -> {mean_after:.1%}")

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "spectrum_id",
                    "name",
                    "n_peaks",
                    "cov_before",
                    "cov_after",
                    "n_filled",
                    "skipped",
                    "reason",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  report: {report_path}")

    if dry_run:
        print("  dry-run: no file written")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  backup: {backup}")

    path.write_text("\n\n".join(out_blocks) + "\n\n", encoding="utf-8")
    print(f"  wrote: {path}")


def parse_spectrum_id_list(raw: Optional[str]) -> Optional[set]:
    if not raw:
        return None
    ids = set()
    for part in re.split(r"[\s,]+", raw.strip()):
        if part:
            ids.add(part)
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description="Batch-fill PFAS MSP annotations from sub-formulas.")
    p.add_argument("msp", nargs="+", type=Path, help="MSP file(s) to update (e.g. train.msp val.msp)")
    p.add_argument("--ppm", type=float, default=30.0, help="Mass tolerance in ppm (default: 30)")
    p.add_argument(
        "--da-tol",
        type=float,
        default=0.02,
        help="Absolute |Δm| tolerance in Da (default: 0.02)",
    )
    p.add_argument(
        "--match-mode",
        choices=("ppm", "da", "hybrid"),
        default="hybrid",
        help="Peak matching: ppm, da, or hybrid (ppm OR da, default: hybrid)",
    )
    p.add_argument(
        "--min-relative-intensity",
        type=float,
        default=0.01,
        help="Skip peaks below this fraction of base peak (default: 0.01)",
    )
    p.add_argument(
        "--max-candidates-per-peak",
        type=int,
        default=3,
        help="Max formula hypotheses per peak (default: 3, 0=unlimited)",
    )
    p.add_argument(
        "--spectrum-ids",
        type=str,
        default=None,
        help="Comma/space-separated spectrum IDs to process (default: all)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print stats only, do not write MSP")
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for per-file CSV reports (default: next to MSP)",
    )
    args = p.parse_args()

    sid_filter = parse_spectrum_id_list(args.spectrum_ids)

    for msp_path in args.msp:
        msp_path = msp_path.resolve()
        report = None
        if args.report_dir:
            report = args.report_dir / f"{msp_path.stem}_fill_report.csv"
        else:
            report = msp_path.with_name(f"{msp_path.stem}_fill_report.csv")

        process_msp(
            msp_path,
            ppm=args.ppm,
            da_tol=args.da_tol,
            match_mode=args.match_mode,
            min_relative_intensity=args.min_relative_intensity,
            max_candidates_per_peak=args.max_candidates_per_peak,
            spectrum_ids=sid_filter,
            dry_run=args.dry_run,
            report_path=report,
        )


if __name__ == "__main__":
    main()
