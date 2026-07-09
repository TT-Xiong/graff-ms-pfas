"""Minimal chemistry helpers for MAGMa (adapted from ms-pred chem_utils)."""

from __future__ import annotations

import re

import numpy as np
from rdkit import Chem

P_TBL = Chem.GetPeriodicTable()

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = r"([A-Z][a-z]*)([0-9]*)"

# PFAS / GrAFF-MS element set (superset of ms-pred subset used here).
VALID_ELEMENTS = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"]

VALID_MONO_MASSES = np.array([P_TBL.GetMostCommonIsotopeMass(el) for el in VALID_ELEMENTS])
ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, VALID_MONO_MASSES))

element_to_ind = {el: i for i, el in enumerate(VALID_ELEMENTS)}
element_to_position = {el: ELEMENT_VECTORS[i] for el, i in element_to_ind.items()}

ion2mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M]+": -ELECTRON_MASS,
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M-2H]-": -2 * ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
}


def is_positive_adduct(adduct: str) -> bool:
    return adduct.endswith("+")


def formula_to_dense(chem_formula: str) -> np.ndarray:
    total_onehot = []
    for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        total_onehot.append(np.repeat(one_hot, repeats=num, axis=0))

    if not total_onehot:
        return np.zeros(len(VALID_ELEMENTS))
    return np.vstack(total_onehot).sum(0)


def vec_to_formula(form_vec) -> str:
    parts = []
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[int(i)]
        ct = int(form_vec[i])
        parts.append(f"{el}{ct}" if ct > 1 else el)
    return "".join(parts)


def canonical_mol_from_inchi(inchi: str):
    return Chem.MolFromInchi(inchi)
