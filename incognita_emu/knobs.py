"""Knobs: 12 global model-ingredient directions + 4 optional extras (16-knob set), in design units u in [-1, 1],
mapped to deck lines on top of the production lines file. ONE value per knob for every nucleus the run touches
(field cells of the lines file), so a knob point is a global parameter point, never a per-nuclide fit.
Ranges = physical priors (published parameter uncertainties; KD03 paper uncertainties for the OMP).
With the 4 extras at 0, deck(u16) == deck12(u12) byte for byte (tests/test_knobs.py)."""
import math

# name: (kind, half-range)  lin: x = x0 + h*u ; log: x = x0 * h**u
KNOBS12 = {
    "sA":   ("lin", 0.05),   # alphald & betald scale (a-parameter systematics), x0 = 1
    "dC":   ("add", 0.15),   # ctable field level shift (ln rho), x0 = 0
    "dPs":  ("add", 0.30),   # pshiftconstant (MeV), x0 = 0
    "rsc":  ("log", 1.15),   # rspincut
    "fE1":  ("log", 1.25),   # ftableadjust e1
    "wE1":  ("log", 1.10),   # wtableadjust e1
    "fM1":  ("log", 1.50),   # ftableadjust m1
    "nv1":  ("lin", 0.02),   # v1adjust n
    "nrv":  ("lin", 0.01),   # rvadjust n
    "nav":  ("lin", 0.05),   # avadjust n
    "nd1":  ("lin", 0.10),   # d1adjust n
    "m2c":  ("log", 1.25),   # m2constant (exciton matrix element)
}
NAMES12 = list(KNOBS12)
ALD = (0.0692559, 0.282769)  # alphald, betald defaults (ldmodel 1, no colenhance)
CELL_KEYS = ("ctable", "ptable", "ftable", "wtable")
OMP_PARTS = ("n", "g")  # "n" alone moves only the incident channel ((n,2n) 0.003 vs 0.042 dex at nv1 +1); TALYS index 0 = outgoing n


def phys12(u):
    """design vector (dict or list in NAMES order) -> physical values"""
    if not isinstance(u, dict): u = dict(zip(NAMES12, u))
    out = {}
    for k, (kind, h) in KNOBS12.items():
        x = float(u.get(k, 0.0))
        out[k] = (1.0 + h * x) if kind == "lin" else (h * x if kind == "add" else h ** x)
    return out


def field_cells(text):
    cells = set()
    for l in text.splitlines():
        w = l.split()
        if len(w) >= 4 and w[0].lower() in CELL_KEYS: cells.add((int(w[1]), int(w[2])))
    return cells


def shift_ctable(text, cells, dC):
    if dC == 0.0: return text
    out, pinned = [], set()
    for l in text.splitlines():
        w = l.split()
        if len(w) >= 4 and w[0].lower() == "ctable" and (len(w) == 4 or w[4] in ("0", "0.0")):
            pinned.add((int(w[1]), int(w[2]))); l = f"ctable {w[1]} {w[2]} {float(w[3]) + dC:.6f}"
        out.append(l)
    out += [f"ctable {z} {a} {dC:.6f}" for z, a in sorted(cells - pinned)]
    return "\n".join(out)


def deck12(base_text, u, omp_parts=OMP_PARTS):
    """base lines text + knob point u -> new lines text"""
    p = phys12(u); cells = field_cells(base_text)
    txt = shift_ctable(base_text.rstrip("\n"), cells, p["dC"])
    out = []
    for z, a in sorted(cells):
        if p["sA"] != 1.0:
            out += [f"alphald {z} {a} {ALD[0] * p['sA']:.8f}", f"betald {z} {a} {ALD[1] * p['sA']:.8f}"]
        if p["dPs"] != 0.0: out.append(f"pshiftconstant {z} {a} {p['dPs']:.6f}")
        if p["fE1"] != 1.0: out.append(f"ftableadjust {z} {a} {p['fE1']:.6f} e1")
        if p["wE1"] != 1.0: out.append(f"wtableadjust {z} {a} {p['wE1']:.6f} e1")
        if p["fM1"] != 1.0: out.append(f"ftableadjust {z} {a} {p['fM1']:.6f} m1")
    if p["rsc"] != 1.0: out.append(f"rspincut {p['rsc']:.6f}")
    for key, kw in (("nv1", "v1adjust"), ("nrv", "rvadjust"), ("nav", "avadjust"), ("nd1", "d1adjust")):
        if p[key] != 1.0:
            out += [f"{kw} {part} {p[key]:.6f}" for part in omp_parts]
    if p["m2c"] != 1.0: out.append(f"m2constant {p['m2c']:.6f}")
    return txt + "\n" + "\n".join(out) + "\n"


# ---------------- 16-knob set: 4 extra knobs ----------------
# pv1  v1adjust p  (outgoing protons)                                   lin, half-range 0.03
# av1  v1adjust a  (outgoing alphas, Avrigeanu 2014 real depth)          lin, half-range 0.05
# fvc  vfiscoradjust Z A x (WKB fission-path height scale, Z >= 90)      lin, half-range 0.05
# fbc  betafiscoradjust Z A x (WKB fission-path width scale, Z >= 90)    lin, half-range 0.05
# fisbaradjust/fishwadjust were DROPPED: inert under the production fismodel 6.
EXTRA = {"pv1": ("lin", 0.03), "av1": ("lin", 0.05), "fvc": ("lin", 0.05), "fbc": ("lin", 0.05)}
KNOBS = {**KNOBS12, **EXTRA}
NAMES = list(KNOBS)
FISS = ("fvc", "fbc")
FISS_IDX = (14, 15)


def phys(u):
    """u (dict, or list with 12..16 entries in NAMES order; missing = 0) -> physical values of all 16 knobs"""
    if not isinstance(u, dict): u = dict(zip(NAMES, u))
    out = phys12({k: u.get(k, 0.0) for k in NAMES12})
    for k, (kind, h) in EXTRA.items():
        out[k] = 1.0 + h * float(u.get(k, 0.0))
    return out


def deck(base_text, u, omp_parts=OMP_PARTS):
    """base lines text + knob point u (12..16 entries) -> new lines text"""
    if not isinstance(u, dict): u = dict(zip(NAMES, u))
    txt = deck12(base_text, {k: u.get(k, 0.0) for k in NAMES12}, omp_parts)
    p = phys(u); out = []
    if p["pv1"] != 1.0: out.append(f"v1adjust p {p['pv1']:.6f}")
    if p["av1"] != 1.0: out.append(f"v1adjust a {p['av1']:.6f}")
    fz = [(z, a) for z, a in sorted(field_cells(base_text)) if z >= 90]
    for kw, k in (("vfiscoradjust", "fvc"), ("betafiscoradjust", "fbc")):
        if p[k] != 1.0: out += [f"{kw} {z} {a} {p[k]:.6f}" for z, a in fz]
    return txt + ("\n".join(out) + "\n" if out else "")


def names(nk):
    return NAMES12 if nk == 12 else NAMES[:nk]


def dlnx_du(nk=12):
    """d ln x / du per knob (lin: h, log: ln h); additive knobs (dC, dPs): dx/du = h"""
    import math
    out = []
    for k in names(nk):
        kind, h = KNOBS[k]
        out.append(h if kind == "lin" else (math.log(h) if kind == "log" else h))
    return out
