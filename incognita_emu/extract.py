"""Extract: every finished run under OUT/runs -> one compact npz (EXTRACT path) with rid, Z, A, u[12], e_ev[54],
capture[54], n2n[54], every other tot_*/ch_xs* channel, scalars (ing_* numeric), cpu_s, wall_s.
usage: python -m incognita_emu.extract OUT EXTRACT"""
import json, sys
from pathlib import Path
import numpy as np


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out = Path(argv[0]); rows = []
    keys_ch = None
    for d in sorted((out / "runs").iterdir()):
        f = list(d.glob("*.npz"))
        if not f: continue
        try:
            z = np.load(f[0], allow_pickle=True); th = json.load(open(d / "theta.json"))
        except Exception as ex:
            print("skip", d.name, ex); continue
        r = {"rid": d.name, "Z": th["Z"], "A": th["A"], "u": np.array(th["u"]), "e_ev": z["e_ev"]}
        for k in z.files:
            v = z[k]
            if (k.startswith(("tot_", "ch_xs")) or k == "capture_mb") and v.shape == z["e_ev"].shape: r["x:" + k] = v.astype(float)
            elif k.startswith("ing_") and v.shape == () and v.dtype.kind in "fiub": r["s:" + k] = float(v)
            elif k in ("cpu_s", "wall_s", "run_cpu_s"): r["s:" + k] = float(v)
        rows.append(r)
    allk = sorted({k for r in rows for k in r if k.startswith(("x:", "s:"))})
    n, ne = len(rows), len(rows[0]["e_ev"])
    D = {"rid": np.array([r["rid"] for r in rows]), "Z": np.array([r["Z"] for r in rows]), "A": np.array([r["A"] for r in rows]),
         "u": np.stack([r["u"] for r in rows]), "e_ev": rows[0]["e_ev"]}
    for k in allk:
        if k.startswith("x:"): D[k] = np.stack([r.get(k, np.zeros(ne)) for r in rows])
        else: D[k] = np.array([r.get(k, np.nan) for r in rows])
    np.savez_compressed(argv[1], **D); print(n, "runs", len(allk), "keys")


if __name__ == "__main__":
    main()
