"""
Parse manually annotated PFAS MSP files into a GrAFF-MS training dataframe (.pkl).

Reads train/val/test MSP splits from ``data/pfas/nist_标注/`` (or custom paths),
parses ``product=loss`` peak comments with optional ``;i=N`` isotope tags, and
writes a single pickle aligned with ``train-graff-ms.py`` expectations.

Annotation syntax (per peak, third column):
  F1O3=C5F10S1;i=2     product + explicit loss, isotope 2
  C5F11O3S1=           product only (empty loss), e.g. precursor peak
  C3F7=C2F4O3S1;...    multiple hypotheses separated by ``;``
  ?                    unannotated peak (skipped)

Unlike ``preprocess-nist.py``, product and loss formulas are taken directly from
the comment (not recomputed from the precursor). Splits come from input filenames
(no structure-disjoint re-splitting).
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.random as npr
import pandas as pd
from pyteomics.mass import Composition
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from src.io import read_msp, write_msp

# Keep in sync with src/graff.py (avoid importing torch / lightning here)
atom_types = sorted(["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"])
isotope_types = [0, 1, 2]

# PFAS covariates (for downstream PFAS-specific featurization)
PRECURSOR_TYPES = ["[M-H]-", "[M]+", "[M+H]+", "[M-2H]-"]
DISSOCIATION_TYPES = ["HCD", "CID"]
DISSOCIATION_ID = {"HCD": 0, "CID": 1}
CE_BINS = [15, 30, 45, 60]
CE_ID = {15: 0, 30: 1, 45: 2, 60: 3}

# Placeholder for legacy train-graff-ms.py (expects an Orbitrap instrument name)
LEGACY_INSTRUMENT = "Orbitrap Fusion Lumos"

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

DEFAULT_MSP_PATHS = [
    os.path.join("data", "pfas", "nist_标注", "train.msp"),
    os.path.join("data", "pfas", "nist_标注", "val.msp"),
    os.path.join("data", "pfas", "nist_标注", "test.msp"),
]


def composition_to_string(comp: Union[str, Composition, None]) -> str:
    if comp is None:
        return ""
    if isinstance(comp, str):
        comp = comp.strip()
        if not comp:
            return ""
        comp = parse_compact_formula(comp)
    return "".join(f"{a}{comp[a]}" for a in sorted(comp) if comp[a] > 0)


def parse_compact_formula(formula_str: str) -> Composition:
    """Parse ``C5F10S1`` / ``F1O3`` style formulas into a Composition."""
    formula_str = formula_str.strip()
    comp = Composition("")
    if not formula_str:
        return comp
    for element, count in _FORMULA_TOKEN_RE.findall(formula_str):
        if not element:
            continue
        n = int(count) if count else 1
        if n <= 0:
            raise ValueError(f"Invalid count in formula: {formula_str}")
        comp[element] += n
    return comp


def formula_within_atom_types(formula: str) -> bool:
    try:
        comp = parse_compact_formula(formula)
    except ValueError:
        return False
    return set(comp) <= set(atom_types) and all(comp[a] >= 0 for a in comp)


def assign_ce_id(ev: float, *, tolerance: float = 0.5) -> Optional[int]:
    if ev is None or (isinstance(ev, float) and np.isnan(ev)):
        return None
    best = min(CE_BINS, key=lambda b: abs(b - ev))
    if abs(best - ev) <= tolerance:
        return CE_ID[best]
    return None


def assign_ce_bin(ev: float, *, tolerance: float = 0.5) -> Optional[int]:
    if ev is None or (isinstance(ev, float) and np.isnan(ev)):
        return None
    best = min(CE_BINS, key=lambda b: abs(b - ev))
    if abs(best - ev) <= tolerance:
        return best
    return None


def extract_pfas_annotations(item) -> Optional[Tuple[List[int], List[str], List[str], List[int]]]:
    """
    Parse PFAS MSP peak comments into training labels.

    Returns (peaks, products, losses, isotopes) or None if no valid annotations.
    """
    annots = item["annots"]
    peaks: List[int] = []
    products: List[str] = []
    losses: List[str] = []
    isotopes: List[int] = []

    for peak_idx, annot in enumerate(annots):
        if not annot or annot in ("?", "more"):
            continue

        for token in str(annot).split(";"):
            token = token.strip()
            if not token:
                continue

            if token.startswith("i="):
                try:
                    iso = int(token[2:])
                except ValueError:
                    continue
                if iso not in isotope_types:
                    continue
                if peaks:
                    isotopes[-1] = iso
                continue

            if "=" not in token:
                continue

            product_str, loss_str = token.split("=", 1)
            product_str = product_str.strip()
            loss_str = loss_str.strip()

            if not product_str:
                continue

            try:
                product = composition_to_string(product_str)
                loss = composition_to_string(loss_str) if loss_str else ""
            except (ValueError, KeyError):
                continue

            if not formula_within_atom_types(product):
                continue
            if loss and not formula_within_atom_types(loss):
                continue

            peaks.append(peak_idx)
            products.append(product)
            losses.append(loss)
            isotopes.append(0)

    if not peaks:
        return None
    return peaks, products, losses, isotopes


def split_from_path(path: str) -> str:
    base = os.path.basename(path).lower()
    for split in ("train", "val", "test"):
        if split in base:
            return split
    raise ValueError(f"Cannot infer split from filename: {path}")


def load_msp_splits(msp_paths: Sequence[str], *, parallel: bool) -> pd.DataFrame:
    frames = []
    for path in msp_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        split = split_from_path(path)
        print(f"Parsing {path} ({split})... ", end="", flush=True)
        part = read_msp(path, parallel=parallel)
        part["split"] = split
        frames.append(part)
        print(f"{len(part)} spectra", flush=True)
    return pd.concat(frames, ignore_index=True)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["PrecursorMZ"] = pd.to_numeric(df["PrecursorMZ"], errors="coerce")
    df["NCE"] = pd.to_numeric(df.get("NCE", np.nan), errors="coerce")
    df["eV"] = pd.to_numeric(df.get("eV", np.nan), errors="coerce")
    df["Dissociation_id"] = pd.to_numeric(
        df.get("Dissociation_id", np.nan), errors="coerce"
    )

    if "Dissociation_type" in df.columns:
        df["Dissociation_type"] = df["Dissociation_type"].astype(str).str.upper()
    else:
        id_to_type = {v: k for k, v in DISSOCIATION_ID.items()}
        df["Dissociation_type"] = df["Dissociation_id"].map(id_to_type)

    df = df[df["Dissociation_type"].isin(DISSOCIATION_TYPES)]
    df = df[df["Precursor_type"].isin(PRECURSOR_TYPES)]
    df = df[df["PrecursorMZ"].notna() & (df["PrecursorMZ"] <= 1000)]

    df = df[df["Formula"].map(formula_within_atom_types)]
    df = df[df["SMILES"].notna() & (df["SMILES"].astype(str).str.len() > 0)]
    df = df[~df["SMILES"].astype(str).str.contains(r"\.")]

    df["mol"] = df["SMILES"].map(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"])

    # NCE / eV consistency (Thermo PSB104-style conversion used in preprocess-nist.py)
    mask = df["eV"].isna() & df["NCE"].notna() & (df["PrecursorMZ"] > 0)
    df.loc[mask, "eV"] = df.loc[mask, "NCE"] * df.loc[mask, "PrecursorMZ"] / 500.0
    mask = df["NCE"].isna() & df["eV"].notna() & (df["PrecursorMZ"] > 0)
    df.loc[mask, "NCE"] = df.loc[mask, "eV"] * 500.0 / df.loc[mask, "PrecursorMZ"]

    df["CE_bin"] = df["eV"].map(lambda x: assign_ce_bin(x))
    df["CE_ID"] = df["eV"].map(lambda x: assign_ce_id(x))

    if "Dissociation_id" not in df.columns or df["Dissociation_id"].isna().all():
        df["Dissociation_id"] = df["Dissociation_type"].map(DISSOCIATION_ID)

    df["Spectrum"] = df["Spectrum"].astype(str)
    df["InChIKey2D"] = df["InChIKey"].astype(str).str.split("-").str[0]

    # Legacy columns for current train-graff-ms.py (PFAS-specific featurization still TBD)
    df["Instrument"] = LEGACY_INSTRUMENT
    df["Instrument_type"] = df["Dissociation_type"]

    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert annotated PFAS MSP splits to a GrAFF-MS training .pkl",
    )
    parser.add_argument(
        "msp_paths",
        nargs="*",
        help="Annotated MSP files (default: data/pfas/nist_标注/{train,val,test}.msp)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "pfas", "nist_标注", "nist_pfas_annot.pkl"),
        help="Output pickle path",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable pandarallel when parsing MSP blocks",
    )
    parser.add_argument(
        "--export-msp",
        action="store_true",
        help="Also export combined train/val/test .tsv and .msp next to --output",
    )
    args = parser.parse_args()

    msp_paths = args.msp_paths or DEFAULT_MSP_PATHS
    parallel = not args.no_parallel

    if parallel:
        from multiprocessing import cpu_count

        from pandarallel import pandarallel

        pandarallel.initialize(
            progress_bar=False, verbose=0, nb_workers=max(1, cpu_count() // 2)
        )

    df = load_msp_splits(msp_paths, parallel=parallel)
    df = prepare_dataframe(df)

    print("Parsing peak annotations... ", end="", flush=True)
    if parallel:
        annots = df.parallel_apply(extract_pfas_annotations, axis=1)
    else:
        annots = df.apply(extract_pfas_annotations, axis=1)

    annots = pd.DataFrame([*annots], columns=["peaks", "products", "losses", "isotopes"])
    annots.index = df.index
    annots = annots.dropna()
    df = df.join(annots, how="inner")
    print(f"{len(df)} spectra with annotations", flush=True)

    df["has_isotopes"] = df["isotopes"].map(lambda xs: any(i != 0 for i in xs))
    df["intensities"] = df["intensities"].map(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"Saving {args.output}... ", end="", flush=True)
    # Drop RDKit mol objects (version-sensitive) and raw MSP annot strings; training
    # rebuilds graphs from SMILES in train-graff-ms.py.
    export_df = df.drop(columns=["mol", "annots"], errors="ignore")
    export_df.to_pickle(args.output)
    print("done", flush=True)

    if args.export_msp:
        prefix, _ = os.path.splitext(args.output)
        print("Exporting split artifacts... ", end="", flush=True)
        for split in ("train", "val", "test"):
            part = df.query(f'split=="{split}"')
            if len(part) == 0:
                continue
            tsv_cols = [
                "Spectrum",
                "SMILES",
                "Precursor_type",
                "CE_ID",
                "Dissociation_type",
            ]
            part[tsv_cols].to_csv(
                f"{prefix}_{split}.tsv", sep="\t", header=False, index=False
            )
            msp_cols = tsv_cols + [
                "InChIKey",
                "Formula",
                "PrecursorMZ",
                "eV",
                "Dissociation_id",
            ]
            write_msp(
                f"{prefix}_{split}.msp",
                part["mzs"].tolist(),
                part["intensities"].tolist(),
                **{c: part[c].tolist() for c in msp_cols},
            )
        print("done", flush=True)

    num_spectra = df["split"].value_counts()
    num_structures = df.groupby("split")["InChIKey"].nunique()
    for split in ("train", "val", "test"):
        if split in num_spectra.index:
            print(
                f"{split}: {num_spectra[split]} spectra, "
                f"{num_structures.get(split, 0)} structures"
            )


if __name__ == "__main__":
    main()
