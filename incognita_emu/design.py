#!/usr/bin/env python3
"""Run designs (which engine runs to make). Every generator returns item lines "RUNID Z A u1,...,uk" in queue order;
the item files behind the shipped models are reproduced byte for byte (tests/test_design.py; md5s in ITEM_MD5).

Layout for ONE nucleus KEY = "ZZZ_AAA" (what a new nucleus needs before it can be emulated, `nucleus_items`):
  c_KEY                    centre (u = 0)
  a<kk><p|m>_KEY           one-at-a-time axis runs, knob k at u = +-0.5 (2 nk runs; A0 anchors, exact sensitivities)
  s<jjj>_KEY               scrambled Sobol (scipy, d = 12, seed 20261012 + 1000 Z + A), points 0..n-1 in [-1, 1]^12
  t<jjj>_KEY               16-knob Sobol (seed + 500000; d = 16 for Z >= 90, 14 otherwise, inactive fission dims = 0)
Held-out points: Sobol index j % 5 == 4. Nucleus splits: nuclei sorted by (A, Z); HN = [4::10], VAL = [9::10].

usage: python -m incognita_emu.design {pilot|stage1|2b|fresh|nucleus} [--key 026_056 --nsob 32 --nk 12] [--out DIR]"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from .config import META

D = 12
fmt = lambda u: ",".join(f"{x:.6f}" for x in u)
za = lambda s: (int(s[:3]), int(s[4:]))
ITEM_MD5 = {"pilot/items_all.txt": "50e743a4685c51e28f425c7c06b3031a", "stage1/items_r1.txt": "0b685965d4a9ff499c6c7ef75474bb8b",
            "stage1/items_r2.txt": "487425d724aa74eae97c403ef9bc6308", "2b/items_2b.txt": "c451bbb2ea2b3a20bae3adff4bcfeaeb",
            "2b/items_2b_all.txt": "3677c9c3289dadde7730f7c788d96de8", "fresh/items_na.txt": "e05ddf43db7f53ce204327a026ba3521"}


def meta(name):
    return json.load(open(META / name))


# ---------------- pilot design (60 nuclei, one global Sobol) ----------------
def pilot_items():
    """centre (60), Jacobian trio +-0.5 (72), design runs d0000..d3999 (Sobol seed 20261011, nucleus NUC[i % 60])"""
    M = meta("design_meta_pilot.json"); NUC, TRIO, NDES = M["NUC"], M["TRIO"], M["NDES"]
    U = qmc.Sobol(d=D, scramble=True, seed=20261011).random(4096)[:NDES] * 2 - 1
    rows = [f"c_{s} {za(s)[0]} {za(s)[1]} {fmt(np.zeros(D))}" for s in NUC]
    for s in TRIO:
        for k in range(D):
            for sg, nm in ((0.5, "p"), (-0.5, "m")):
                u = np.zeros(D); u[k] = sg; rows.append(f"j{k:02d}{nm}_{s} {za(s)[0]} {za(s)[1]} {fmt(u)}")
    for i in range(NDES):
        s = NUC[i % 60]; rows.append(f"d{i:04d} {za(s)[0]} {za(s)[1]} {fmt(U[i])}")
    return rows


# ---------------- per-nucleus layout ----------------
def sobol(key, n=64):
    z, a = za(key)
    return qmc.Sobol(d=D, scramble=True, seed=20261012 + 1000 * z + a).random(n) * 2 - 1


def axis(key, ks=range(D), nk=D, tag="a"):
    rows = []
    for k in ks:
        for sg, nm in ((0.5, "p"), (-0.5, "m")):
            u = np.zeros(nk); u[k] = sg; rows.append(f"{tag}{k:02d}{nm}_{key} {za(key)[0]} {za(key)[1]} {fmt(u)}")
    return rows


def sobol16(key, n=16):
    z, a = za(key); d = 16 if z >= 90 else 14
    S = qmc.Sobol(d=d, scramble=True, seed=20261012 + 1000 * z + a + 500000).random(n) * 2 - 1
    U = np.zeros((n, 16)); U[:, :d] = S
    return U


def nucleus_items(key, nsob=32, nk=12, sob0=0):
    """everything a NEW nucleus needs for a per-nucleus emulator: centre + 2 nk axis runs + nsob Sobol points
    (nk = 12: s-points; nk = 16: the 12-knob s-points padded with 0 plus 16-knob t-points; fission axis Z >= 90 only)"""
    z, a = za(key)
    rows = [f"c_{key} {z} {a} {fmt(np.zeros(nk))}"]
    if nk == 12:
        rows += axis(key)
        U = sobol(key, sob0 + nsob)
        rows += [f"s{j:03d}_{key} {z} {a} {fmt(U[j])}" for j in range(sob0, sob0 + nsob)]
    else:
        ks = list(range(16)) if z >= 90 else list(range(14))
        rows += axis(key, ks, 16)
        U = sobol16(key, sob0 + nsob)
        rows += [f"t{j:03d}_{key} {z} {a} {fmt(U[j])}" for j in range(sob0, sob0 + nsob)]
    return rows


def stage1_items():
    """(r1, r2, meta) of the 185-nucleus design: queue order HN, VAL, TRAIN alternating light/heavy"""
    pilot = meta("design_meta_pilot.json"); PNUC, TRIO = pilot["NUC"], pilot["TRIO"]
    DEVC = meta("nuclei176.json")
    ALL = sorted(set(DEVC) | set(PNUC), key=lambda s: (int(s[4:]), int(s[:3])))
    HN, VAL = ALL[4::10], ALL[9::10]
    rest = [k for k in ALL if k not in HN and k not in VAL]
    order = []
    for grp in (HN, VAL, rest):
        g = list(grp); order += [g[i // 2] if i % 2 == 0 else g[-1 - i // 2] for i in range(len(g))]
    r1, r2 = [], []
    for key in order:
        z, a = za(key)
        if key in PNUC:
            if key not in TRIO: r1 += axis(key)
            continue
        U = sobol(key)
        r1.append(f"c_{key} {z} {a} {fmt(np.zeros(D))}")
        r1 += axis(key)
        r1 += [f"s{j:03d}_{key} {z} {a} {fmt(U[j])}" for j in range(32)]
        r2 += [f"s{j:03d}_{key} {z} {a} {fmt(U[j])}" for j in range(32, 64)]
    m = {"ALL": ALL, "HN": HN, "VAL": VAL, "TRIO": TRIO, "PILOT": PNUC, "order": order, "n_r1": len(r1), "n_r2": len(r2)}
    return r1, r2, m


WIRED_KEYS = (("026_056", "040_090", "082_208", "092_238", "090_232"), ("092_238", "090_232"))


def wired():
    """16-knob axis runs that were made in an earlier wiring check (2b rid -> wiring rid)"""
    return {**{f"a{k}{s}_{key}": f"w{k}{s}_{key}" for key in WIRED_KEYS[0] for k in (12, 13) for s in "pm"},
            **{f"a{k}{s}_{key}": f"v{k}{s}_{key}" for key in WIRED_KEYS[1] for k in (14, 15) for s in "pm"}}


def items_2b():
    """(queue, all) of the 16-knob design: per nucleus 2b axis (12,13 [+14,15 for Z >= 90]) + 16 t-points"""
    M = meta("design_meta_stage1.json"); W = wired(); allrows, q = [], []
    for key in M["order"]:
        z, a = za(key); act = z >= 90; ks = range(12, 16) if act else range(12, 14)
        rows = axis(key, ks, 16)
        U = sobol16(key, 16)
        rows += [f"t{j:03d}_{key} {z} {a} {fmt(U[j])}" for j in range(16)]
        allrows += rows; q += [r for r in rows if r.split()[0] not in W]
    return q, allrows


def fresh_items(j0=64, j1=73):
    """fresh test points: Sobol points j0..j1-1 of every non-HN stage-1 nucleus (queue alternates light/heavy)"""
    M = meta("design_meta_stage1.json")
    TV = [k for k in M["ALL"] if k not in M["HN"]]
    order = [TV[i // 2] if i % 2 == 0 else TV[-1 - i // 2] for i in range(len(TV))]
    L = []
    for key in order:
        z, a = za(key); U = qmc.Sobol(d=12, scramble=True, seed=20261012 + 1000 * z + a).random(max(80, j1)) * 2 - 1
        L += [f"s{j:03d}_{key} {z} {a} " + fmt(U[j]) for j in range(j0, j1)]
    return L


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text("\n".join(rows) + "\n"); return path


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["pilot", "stage1", "2b", "fresh", "nucleus"])
    ap.add_argument("--out", default="."); ap.add_argument("--key"); ap.add_argument("--nsob", type=int, default=32)
    ap.add_argument("--nk", type=int, default=12)
    a = ap.parse_args(argv); o = Path(a.out)
    if a.which == "pilot": print(write(o / "items_all.txt", pilot_items()))
    elif a.which == "stage1":
        r1, r2, m = stage1_items(); write(o / "items_r1.txt", r1); write(o / "items_r2.txt", r2)
        json.dump(m, open(o / "design_meta.json", "w"), indent=1); print(len(r1), len(r2))
    elif a.which == "2b":
        q, al = items_2b(); write(o / "items_2b.txt", q); write(o / "items_2b_all.txt", al); print(len(q), len(al))
    elif a.which == "fresh": print(write(o / "items_na.txt", fresh_items()))
    else:
        rows = nucleus_items(a.key, a.nsob, a.nk); print(write(o / f"items_{a.key}.txt", rows), len(rows))


if __name__ == "__main__":
    main()
