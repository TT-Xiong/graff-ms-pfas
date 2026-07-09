"""
Annotate PFAS (and compatible) MGF spectra for GrAFF-MS training.

Uses compositional sub-formula enumeration (Cartesian product over element
counts) and matches experimental peaks to candidate product formulas at M+0,
M+1, and M+2 (via a fixed neutron-mass shift, consistent with GrAFF).

Peaks may receive multiple formula hypotheses within the ppm window (like NIST
MSP annotations joined with ``;``), which GrAFF training marginalizes via
logsumexp in ``graff.py`` step().

Output columns extend preprocess-nist.py for PFAS training with:
  - adducts: [M-H]-, [M]+, [M+H]+, [M-2H]-
  - dissociation: HCD / CID (from SOURCE_INSTRUMENT) + Dissociation_id
  - collision energy: discrete CE bins 15/30/45/60 eV + CE_ID
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.random as npr
import pandas as pd
from pyteomics.mass import Composition
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# Allow running as `python src/annotate_pfas.py` from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.graff import atom_types, isotope_types, neutron_mass
from src.io import write_msp


# PFAS training covariates (written to .pkl for downstream featurization)
PRECURSOR_TYPES = ["[M-H]-", "[M]+", "[M+H]+", "[M-2H]-"]
DISSOCIATION_TYPES = ["HCD", "CID"]
DISSOCIATION_ID = {"HCD": 0, "CID": 1}
CE_BINS = [15, 30, 45, 60]
CE_ID = {15: 0, 30: 1, 45: 2, 60: 3}


# ---------------------------------------------------------------------------
# Shared helpers (aligned with preprocess-nist.py)
# ---------------------------------------------------------------------------


def composition_to_string(comp) -> str:
    if isinstance(comp, str):
        comp = Composition(formula=comp)
    return "".join(f"{a}{comp[a]}" for a in sorted(comp) if comp[a] > 0)


def composition_to_counts(comp: Composition) -> Dict[str, int]:
    return {a: int(comp[a]) for a in atom_types if comp[a] > 0}


def counts_to_composition(counts: Dict[str, int]) -> Composition:
    comp = Composition("")
    for el, n in counts.items():
        if n > 0:
            comp[el] += n
    return comp


def precursor_composition(formula: str, precursor_type: str) -> Composition:
    comp = Composition(formula=formula)
    if precursor_type == "[M+H]+":
        comp["H"] += 1
    elif precursor_type == "[M-H]-":
        comp["H"] -= 1
    elif precursor_type == "[M]+":
        pass
    elif precursor_type == "[M-2H]-":
        comp["H"] -= 2
    else:
        raise ValueError(f"Unsupported precursor type: {precursor_type}")
    if comp["H"] < 0:
        raise ValueError(f"Invalid precursor H count for {formula} / {precursor_type}")
    return comp


def parse_collision_energy(raw, precursor_mz: float) -> Tuple[float, float]:
    """
    Parse MGF COLLISION_ENERGY to (eV, NCE).

    PFAS MGF stores eV directly (15/30/45/60). NIST-style ``NCE=xx%`` strings
    are converted via eV = NCE * PrecursorMZ / 500 (preprocess-nist.py).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan, np.nan

    s = str(raw).strip()
    ev = np.nan
    nce = np.nan

    if s.upper().startswith("NCE="):
        s = s[4:].strip()
        if s.endswith("%"):
            s = s[:-1]
        if " " in s:
            s = s.split()[0]
        try:
            nce = float(s)
        except ValueError:
            return np.nan, np.nan
    else:
        try:
            ev = float(s)
        except ValueError:
            return np.nan, np.nan

    if np.isnan(ev) and not np.isnan(nce) and precursor_mz > 0:
        ev = nce * precursor_mz / 500.0
    if not np.isnan(ev) and np.isnan(nce) and precursor_mz > 0:
        nce = ev * 500.0 / precursor_mz

    return ev, nce


def assign_ce_bin(ev: float, *, tolerance: float = 0.5) -> Optional[int]:
    """
    Map continuous eV to a discrete CE bin in CE_BINS.

    Returns the bin value (15/30/45/60) or None if no bin is within tolerance.
    """
    if ev is None or np.isnan(ev):
        return None
    best = min(CE_BINS, key=lambda b: abs(b - ev))
    if abs(best - ev) <= tolerance:
        return best
    return None


@lru_cache(maxsize=4096)
def _element_masses() -> Dict[str, float]:
    pt = Chem.GetPeriodicTable()
    return {
        pt.GetElementSymbol(z): pt.GetMostCommonIsotopeMass(z)
        for z in range(1, 119)
    }


def formula_mass(comp: Composition) -> float:
    mws = _element_masses()
    return float(sum(mws[a] * comp[a] for a in comp if comp[a] > 0))


def enumerate_subformulas(max_counts: Dict[str, int]) -> List[Composition]:
    """All non-empty sub-compositions: prod_e (0..max_e) for each element."""
    elements = sorted(max_counts.keys())
    if not elements:
        return []
    ranges = [range(max_counts[e] + 1) for e in elements]
    out: List[Composition] = []
    for counts in itertools.product(*ranges):
        if sum(counts) == 0:
            continue
        comp = counts_to_composition(dict(zip(elements, counts)))
        out.append(comp)
    return out


def build_candidates(
    precursor_comp: Composition,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Build monotonically sorted candidate m/z array and parallel metadata.

    Each sub-formula yields len(isotope_types) m/z values:
        mass(product) + isotope * neutron_mass
    """
    max_counts = composition_to_counts(precursor_comp)
    subformulas = enumerate_subformulas(max_counts)

    masses: List[float] = []
    products: List[str] = []
    isotopes: List[int] = []

    for sub in subformulas:
        product_str = composition_to_string(sub)
        base = formula_mass(sub)
        for iso in isotope_types:
            masses.append(base + iso * neutron_mass)
            products.append(product_str)
            isotopes.append(iso)

    if not masses:
        return np.array([], dtype=np.float64), [], []

    order = np.argsort(masses)
    mass_arr = np.asarray(masses, dtype=np.float64)[order]
    prod_arr = [products[i] for i in order]
    iso_arr = [isotopes[i] for i in order]
    return mass_arr, prod_arr, iso_arr


def _ppm(obs: float, theo: float) -> float:
    return abs(obs - theo) / theo * 1e6


def match_peak_candidates(
    mz_obs: float,
    cand_mz: np.ndarray,
    cand_products: Sequence[str],
    cand_isotopes: Sequence[int],
    *,
    ppm_tolerance: float,
    max_candidates: int = 0,
) -> List[Tuple[str, int, float]]:
    """
    Return all (product, isotope, ppm) matches within ±ppm_tolerance.

    Mirrors NIST MSP behaviour where a single peak may list multiple formula
    hypotheses (``;``-separated). Duplicate (product, isotope) pairs are
    collapsed, keeping the best ppm match.
    """
    if cand_mz.size == 0:
        return []

    ppm_frac = ppm_tolerance * 1e-6
    lo = mz_obs / (1.0 + ppm_frac)
    hi = mz_obs / (1.0 - ppm_frac) if ppm_frac < 1.0 else mz_obs * (1.0 + ppm_frac)

    left = int(np.searchsorted(cand_mz, lo, side="left"))
    right = int(np.searchsorted(cand_mz, hi, side="right"))

    best: Dict[Tuple[str, int], float] = {}
    for i in range(left, right):
        ppm = _ppm(mz_obs, float(cand_mz[i]))
        if ppm <= ppm_tolerance:
            key = (cand_products[i], cand_isotopes[i])
            if key not in best or ppm < best[key]:
                best[key] = ppm

    matches = sorted(
        ((prod, iso, ppm) for (prod, iso), ppm in best.items()),
        key=lambda x: x[2],
    )
    if max_candidates > 0:
        matches = matches[:max_candidates]
    return matches


def annotate_spectrum(
    mzs: np.ndarray,
    intensities: np.ndarray,
    precursor_comp: Composition,
    *,
    ppm_tolerance: float = 20.0,
    min_mz: float = 0.0,
    max_mz: Optional[float] = None,
    min_relative_intensity: float = 0.0,
    max_candidates_per_peak: int = 0,
) -> Tuple[List[int], List[str], List[str], List[int]]:
    """
    Annotate peaks via sub-formula enumeration + isotope shifts.

    Each experimental peak may produce multiple (product, loss, isotope) rows
    when several sub-formulas fall within the ppm window (NIST-style).
    """
    cand_mz, cand_products, cand_isotopes = build_candidates(precursor_comp)

    if intensities.size and intensities.max() > 0:
        rel = intensities / intensities.max()
    else:
        rel = intensities

    peaks: List[int] = []
    products: List[str] = []
    losses: List[str] = []
    isotopes: List[int] = []

    for peak_idx, (mz_obs, inten, rint) in enumerate(zip(mzs, intensities, rel)):
        if mz_obs < min_mz:
            continue
        if max_mz is not None and mz_obs > max_mz:
            continue
        if rint < min_relative_intensity:
            continue

        matches = match_peak_candidates(
            float(mz_obs),
            cand_mz,
            cand_products,
            cand_isotopes,
            ppm_tolerance=ppm_tolerance,
            max_candidates=max_candidates_per_peak,
        )
        if not matches:
            continue

        for product_str, iso, _ in matches:
            loss_comp = precursor_comp - Composition(formula=product_str)
            if any(loss_comp[a] < 0 for a in loss_comp):
                continue
            loss_str = composition_to_string(loss_comp)

            peaks.append(int(peak_idx))
            products.append(product_str)
            losses.append(loss_str)
            isotopes.append(int(iso))

    return peaks, products, losses, isotopes


# ---------------------------------------------------------------------------
# MGF I/O (PFAS / dimspec format)
# ---------------------------------------------------------------------------

_MGF_META_RE = re.compile(r"^([A-Za-z0-9 _]+?)=(.*)$")
_MGF_PEAK_RE = re.compile(r"^(\d+\.?\d*)\s+(\d+\.?\d*)\s*$")


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

    return {
        **meta,
        "mzs": np.asarray(mzs, dtype=np.float32),
        "intensities": np.asarray(intensities, dtype=np.float32),
    }


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


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def _prepare_row(
    row: pd.Series,
    *,
    ppm_tolerance: float,
    min_mz: float,
    min_relative_intensity: float,
    max_candidates_per_peak: int,
    max_precursor_mz: float,
    ce_tolerance: float,
) -> Optional[dict]:
    if str(row.get("MSLEVEL", "2")) not in ("2", 2):
        return None

    smiles = row.get("SMILES")
    precursor_type = row.get("Precursor_type")
    if pd.isna(smiles) or pd.isna(precursor_type):
        return None
    if precursor_type not in PRECURSOR_TYPES:
        return None

    dissociation_type = str(row.get("Dissociation_type", "")).strip().upper()
    if dissociation_type not in DISSOCIATION_TYPES:
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    if "." in str(smiles):
        return None

    neutral_formula = CalcMolFormula(mol)
    neutral_comp = Composition(formula=neutral_formula)
    if not set(neutral_comp).issubset(set(atom_types)):
        return None

    try:
        prec_comp = precursor_composition(neutral_formula, precursor_type)
    except ValueError:
        return None

    precursor_mz = float(row.get("PrecursorMZ", np.nan))
    if np.isnan(precursor_mz) or precursor_mz > max_precursor_mz:
        return None

    ev, nce = parse_collision_energy(row.get("Collision_energy"), precursor_mz)
    ce_bin = assign_ce_bin(ev, tolerance=ce_tolerance)
    if ce_bin is None:
        return None
    ce_id = CE_ID[ce_bin]

    mzs = np.asarray(row["mzs"], dtype=np.float64)
    intensities = np.asarray(row["intensities"], dtype=np.float64)

    peaks, products, losses, isotopes = annotate_spectrum(
        mzs,
        intensities,
        prec_comp,
        ppm_tolerance=ppm_tolerance,
        min_mz=min_mz,
        max_mz=precursor_mz + neutron_mass * max(isotope_types) + 0.5,
        min_relative_intensity=min_relative_intensity,
        max_candidates_per_peak=max_candidates_per_peak,
    )

    if len(peaks) == 0:
        return None

    intensities_norm = intensities / intensities.sum() if intensities.sum() > 0 else intensities

    return {
        "Spectrum": str(row["Spectrum"]),
        "SMILES": str(smiles),
        "InChIKey": row.get("InChIKey"),
        "Formula": neutral_formula,
        "Precursor_type": precursor_type,
        "PrecursorMZ": precursor_mz,
        "Dissociation_type": dissociation_type,
        "Dissociation_id": DISSOCIATION_ID[dissociation_type],
        "eV": float(ce_bin),
        "NCE": float(nce) if not np.isnan(nce) else float(ce_bin * 500.0 / precursor_mz),
        "CE_ID": ce_id,
        "mzs": mzs.astype(np.float32),
        "intensities": intensities_norm.astype(np.float32),
        "peaks": peaks,
        "products": products,
        "losses": losses,
        "isotopes": isotopes,
        "has_isotopes": any(i != 0 for i in isotopes),
        "filename": row.get("filename"),
    }


def structure_disjoint_split(
    df: pd.DataFrame,
    *,
    seed: int,
    train_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    df = df.copy()
    df["InChIKey2D"] = df["InChIKey"].astype(str).str.split("-").str[0]

    keys = df["InChIKey2D"].unique()
    npr.seed(seed)
    npr.shuffle(keys)

    n = len(keys)
    n_train = int(train_frac * n)
    n_test = int(test_frac * n)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n - n_test])
    test_keys = set(keys[n - n_test :])

    df["split"] = ""
    df.loc[df["InChIKey2D"].isin(train_keys), "split"] = "train"
    df.loc[df["InChIKey2D"].isin(val_keys), "split"] = "val"
    df.loc[df["InChIKey2D"].isin(test_keys), "split"] = "test"
    return df


def annotate_mgf_paths(
    mgf_paths: Sequence[str],
    *,
    ppm_tolerance: float = 20.0,
    min_mz: float = 0.0,
    min_relative_intensity: float = 0.001,
    max_candidates_per_peak: int = 0,
    max_precursor_mz: float = 1000.0,
    min_annotation_coverage: float = 0.0,
    dissociation_types: Optional[Sequence[str]] = None,
    ce_tolerance: float = 0.5,
    sample: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if dissociation_types is None:
        dissociation_types = DISSOCIATION_TYPES

    frames = []
    for path in mgf_paths:
        if verbose:
            print(f"Reading {path}...", flush=True)
        raw = read_mgf(path)
        if "Dissociation_type" in raw.columns:
            raw["Dissociation_type"] = raw["Dissociation_type"].str.upper()
            raw = raw[raw["Dissociation_type"].isin(dissociation_types)]
        frames.append(raw)

    df_in = pd.concat(frames, ignore_index=True)
    if sample is not None and sample > 0:
        df_in = df_in.iloc[:sample].copy()
    if verbose:
        print(f"Annotating {len(df_in)} spectra...", flush=True)

    records = []
    iterator = tqdm(df_in.iterrows(), total=len(df_in), disable=not verbose)
    for _, row in iterator:
        rec = _prepare_row(
            row,
            ppm_tolerance=ppm_tolerance,
            min_mz=min_mz,
            min_relative_intensity=min_relative_intensity,
            max_candidates_per_peak=max_candidates_per_peak,
            max_precursor_mz=max_precursor_mz,
            ce_tolerance=ce_tolerance,
        )
        if rec is None:
            continue
        if min_annotation_coverage > 0:
            coverage = len(set(rec["peaks"])) / max(len(rec["mzs"]), 1)
            if coverage < min_annotation_coverage:
                continue
        records.append(rec)

    if not records:
        raise RuntimeError("No spectra survived annotation filters.")

    df = pd.DataFrame.from_records(records)
    return df


def export_artifacts(
    df: pd.DataFrame,
    output_prefix: str,
    *,
    verbose: bool = True,
) -> str:
    pkl_path = output_prefix + ".pkl"
    df.to_pickle(pkl_path)
    if verbose:
        print(f"Saved {pkl_path} ({len(df)} spectra)", flush=True)

    for split in ("train", "val", "test"):
        part = df.query(f'split=="{split}"')
        if len(part) == 0:
            continue

        tsv_path = f"{output_prefix}_{split}.tsv"
        tsv_cols = ["Spectrum", "SMILES", "Precursor_type", "CE_ID", "Dissociation_type"]
        part[tsv_cols].to_csv(tsv_path, sep="\t", header=False, index=False)

        msp_path = f"{output_prefix}_{split}.msp"
        msp_cols = tsv_cols + ["InChIKey", "Formula", "PrecursorMZ", "eV", "Dissociation_id"]
        write_msp(
            msp_path,
            part["mzs"].tolist(),
            part["intensities"].tolist(),
            **{c: part[c].tolist() for c in msp_cols},
        )
        if verbose:
            print(f"Exported {split}: {len(part)} spectra -> {tsv_path}, {msp_path}", flush=True)

    return pkl_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Annotate PFAS MGF spectra via compositional formula enumeration for GrAFF-MS training.",
    )
    p.add_argument(
        "mgf_paths",
        nargs="+",
        help="One or more MGF files (e.g. data/pfas/nist_pfas_ms2_train.mgf)",
    )
    p.add_argument(
        "output_prefix",
        help="Output prefix without extension (writes .pkl and optional split exports)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument(
        "--dissociation-types",
        default="HCD,CID",
        help="Comma-separated SOURCE_INSTRUMENT values (HCD/CID) to keep from MGF",
    )
    p.add_argument(
        "--ce-tolerance",
        type=float,
        default=0.5,
        help="Max |eV - bin| when mapping COLLISION_ENERGY to CE bins 15/30/45/60",
    )
    p.add_argument(
        "--ppm",
        type=float,
        default=20.0,
        help="± m/z matching tolerance in ppm (default 20, NIST-like multi-hypothesis within window)",
    )
    p.add_argument("--min-mz", type=float, default=0.0)
    p.add_argument("--max-precursor-mz", type=float, default=1000.0)
    p.add_argument(
        "--min-relative-intensity",
        type=float,
        default=0.001,
        help="Skip peaks below this fraction of base peak",
    )
    p.add_argument(
        "--min-annotation-coverage",
        type=float,
        default=0.0,
        help="Drop spectra where annotated peaks / total peaks is below this fraction",
    )
    p.add_argument(
        "--max-candidates-per-peak",
        type=int,
        default=0,
        help="Cap formula hypotheses per peak (0 = no limit, keep all within ppm)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Process only the first N spectra (0 = all)",
    )
    p.add_argument("--no-export-splits", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    dissoc_types = tuple(s.strip().upper() for s in args.dissociation_types.split(",") if s.strip())

    df = annotate_mgf_paths(
        args.mgf_paths,
        ppm_tolerance=args.ppm,
        min_mz=args.min_mz,
        min_relative_intensity=args.min_relative_intensity,
        max_candidates_per_peak=args.max_candidates_per_peak,
        max_precursor_mz=args.max_precursor_mz,
        min_annotation_coverage=args.min_annotation_coverage,
        dissociation_types=dissoc_types,
        ce_tolerance=args.ce_tolerance,
        sample=args.sample or None,
    )

    df = structure_disjoint_split(
        df,
        seed=args.seed,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
    )

    if args.no_export_splits:
        pkl_path = args.output_prefix + ".pkl"
        df.to_pickle(pkl_path)
        print(f"Saved {pkl_path} ({len(df)} spectra)", flush=True)
    else:
        pkl_path = export_artifacts(df, args.output_prefix, verbose=True)

    for split in ("train", "val", "test"):
        n_spec = (df["split"] == split).sum()
        n_struct = df.loc[df["split"] == split, "InChIKey"].nunique()
        print(f"{split}: {n_spec} spectra, {n_struct} structures")

    print(f"Done. Training pickle: {pkl_path}")
    print("Next: python train-graff-ms.py", pkl_path)


if __name__ == "__main__":
    main()
