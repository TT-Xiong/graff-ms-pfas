"""Compare ppm vs fixed Da tolerance for subformula peak matching."""
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_graff_stub = ModuleType("src.graff")
_graff_stub.atom_types = sorted(["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"])
_graff_stub.isotope_types = [0, 1, 2]
_graff_stub.neutron_mass = 1.008665
sys.modules["src.graff"] = _graff_stub

from src.annotate_pfas import build_candidates, precursor_composition

PEAK = re.compile(r"^(\d+\.?\d*)\s+(\d+\.?\d*)(?:\s+(.+))?$")


def parse_blocks(path):
    text = path.read_text(encoding="utf-8")
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        meta = {}
        mzs, annots = [], []
        for line in block.splitlines():
            if ":" in line and not PEAK.match(line):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
            elif PEAK.match(line):
                m = PEAK.match(line)
                mzs.append(float(m.group(1)))
                a = m.group(3)
                annots.append("" if a is None else a.strip('"'))
        yield meta, mzs, annots


def unann(a):
    return not a or a in ("?", "more")


def best_delta(mz_obs, cand_mz):
    if len(cand_mz) == 0:
        return float("inf"), float("inf")
    deltas = abs(cand_mz - mz_obs)
    i = int(deltas.argmin())
    theo = float(cand_mz[i])
    da = float(deltas[i])
    ppm = da / theo * 1e6
    return da, ppm


def count_matches(meta, mzs, annots, *, ppm=None, da=None):
    formula = re.sub(r"[+\-]+$", "", meta.get("Formula", ""))
    pt = meta.get("Precursor_type", "")
    try:
        prec = precursor_composition(formula, pt)
    except Exception:
        return 0, 0
    cm, _, _ = build_candidates(prec)
    n = 0
    total = 0
    for mz, a in zip(mzs, annots):
        if not unann(a):
            continue
        total += 1
        d, p = best_delta(mz, cm)
        ok = True
        if ppm is not None and p > ppm:
            ok = False
        if da is not None and d > da:
            ok = False
        if ok:
            n += 1
    return n, total


def sweep_spectrum_9885():
    path = ROOT / "data/pfas/nist_标注/train.msp"
    for meta, mzs, annots in parse_blocks(path):
        if meta.get("Name") != "9885":
            continue
        prec = precursor_composition("C5HF9O2", "[M-H]-")
        cm, cp, ci = build_candidates(prec)
        idxs = [i for i, a in enumerate(annots) if unann(a)]
        print("=== Spectrum 9885 (21 unannotated peaks) ===")
        print(f"{'m/z':>10} {'best_dDa':>9} {'best_ppm':>9} | match@30ppm | match@0.02Da | match@both")
        for i in idxs:
            da, ppm = best_delta(mzs[i], cm)
            m30 = ppm <= 30
            m02 = da <= 0.02
            print(
                f"{mzs[i]:10.4f} {da:9.4f} {ppm:9.1f} | "
                f"{'Y' if m30 else 'N':>10} | {'Y' if m02 else 'N':>11} | "
                f"{'Y' if m30 and m02 else 'N' if not (m30 or m02) else 'partial':>8}"
            )
        for ppm_cut in [30, 50, 100]:
            n = sum(1 for i in idxs if best_delta(mzs[i], cm)[1] <= ppm_cut)
            print(f"  ppm<={ppm_cut}: {n}/21")
        for da_cut in [0.01, 0.02, 0.05, 0.1]:
            n = sum(1 for i in idxs if best_delta(mzs[i], cm)[0] <= da_cut)
            print(f"  |d|<={da_cut} Da: {n}/21")
        break


def sweep_train():
    path = ROOT / "data/pfas/nist_标注/train.msp"
    rules = [
        ("ppm<=30", dict(ppm=30)),
        ("ppm<=50", dict(ppm=50)),
        ("|d|<=0.02", dict(da=0.02)),
        ("|d|<=0.05", dict(da=0.05)),
        ("|d|<=0.1", dict(da=0.1)),
        ("ppm<=30 AND |d|<=0.02", dict(ppm=30, da=0.02)),
        ("ppm<=100 OR |d|<=0.02", None),  # special
    ]
    counts = {name: 0 for name, _ in rules}
    total = 0
    for meta, mzs, annots in parse_blocks(path):
        formula = re.sub(r"[+\-]+$", "", meta.get("Formula", ""))
        pt = meta.get("Precursor_type", "")
        try:
            prec = precursor_composition(formula, pt)
            cm, _, _ = build_candidates(prec)
        except Exception:
            continue
        for mz, a in zip(mzs, annots):
            if not unann(a):
                continue
            total += 1
            da, ppm = best_delta(mz, cm)
            if ppm <= 30:
                counts["ppm<=30"] += 1
            if ppm <= 50:
                counts["ppm<=50"] += 1
            if da <= 0.02:
                counts["|d|<=0.02"] += 1
            if da <= 0.05:
                counts["|d|<=0.05"] += 1
            if da <= 0.1:
                counts["|d|<=0.1"] += 1
            if ppm <= 30 and da <= 0.02:
                counts["ppm<=30 AND |d|<=0.02"] += 1
            if ppm <= 100 or da <= 0.02:
                counts["ppm<=100 OR |d|<=0.02"] += 1
    print()
    print(f"=== All train.msp fillable ? peaks: {total} ===")
    for name in counts:
        print(f"  {name:28s}: {counts[name]:4d} ({100*counts[name]/total:.1f}%)")


if __name__ == "__main__":
    sweep_spectrum_9885()
    sweep_train()
