"""Fast evaluation of the Incognita engine through per-nucleus GP emulators.
Models: 185 nuclei, 9 channels, grid54 (1 keV-20 MeV), 12 knobs u in [-1, 1] (incognita_emu.knobs), GPU if present.

    from incognita_emu import api
    E = api.load()                                 # $INCOGNITA_EMU_MODELS/models.pt (default <repo>/models/default)
    S = E.sigma("092_238", U)                      # U (n, nk) -> {channel: (n, 54) mb}, NaN off the modelled cells
    J = E.jacobian("092_238", u)                   # {channel: (nk, 54) d log10 sigma / du} by autograd at u
    T = E.tmc("092_238", n=10**6)                  # mean, sd, quantiles, relcov (54x54) per channel; uniform knob prior
    E.batch(keys, U)                               # many nuclei x many points; see bench()
    E.verify_items(["092_238"], n=10, path="items.txt")   # then: scripts/verify.sh items.txt OUTDIR (exact engine)
    E.compare(OUTDIR)                               # per (run, channel) |emulator - exact| in dex
Channels: total elastic nonel capture inel n2n np na fission. Outside |u| <= 1: extrapolation, not validated.
Cells with sigma0 <= 1e-3 mb are not modelled (NaN). A nucleus NOT in E.nuclei needs its own centre + axis + ~32 Sobol
engine runs (incognita_emu.design) before it can be emulated."""
import json, time
from pathlib import Path
import numpy as np
import torch

from . import gp as T
from . import config
from .knobs import NAMES12

KNOBS = tuple(NAMES12)
CH_NPZ = {"total": "tot_total", "elastic": "tot_elastic", "nonel": "tot_nonelastic", "capture": "capture_mb",
          "inel": "ch_xs100000", "n2n": "ch_xs200000", "np": "ch_xs010000", "na": "ch_xs000001", "fission": "tot_fission"}


def _dev(m):
    return {k: (v.to(T.DEV) if torch.is_tensor(v) else ([t.to(T.DEV) for t in v] if isinstance(v, list) else v)) for k, v in m.items()}


class Emu:
    def __init__(self, d=None):
        p = Path(config.models_dir(d)); p = p if p.suffix == ".pt" else p / "models.pt"
        M = torch.load(p, weights_only=False, map_location="cpu"); self.path = p
        self.e_ev = np.asarray(M["e_ev"]); self.channels = list(M["channels"]); self.models = M["models"]
        self.knobs = list(M.get("knobs") or KNOBS); self.nk = len(self.knobs)
        self.floor = M.get("floor", 1e-5); self.cut = M.get("cut", 1e-3)
        for md in self.models.values():
            md["m"] = {ch: _dev(m) for ch, m in md["m"].items()}
        self.nuclei = sorted(self.models)

    def logratio(self, key, U, channels=None, grad=False):
        md = self.models[key]
        Ut = U if torch.is_tensor(U) else torch.as_tensor(np.atleast_2d(np.asarray(U, float)), dtype=T.F64, device=T.DEV)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return {ch: T.pred_gp(md["m"][ch], Ut) for ch in (channels or self.channels) if ch in md["m"]}

    def sigma(self, key, U, channels=None):
        md = self.models[key]; out = {}
        for ch, Y in self.logratio(key, U, channels).items():
            mask = md["mask"][ch]; s = np.full((Y.shape[0], len(self.e_ev)), np.nan)
            s[:, mask] = md["centre"][ch][mask] * 10.0 ** Y.cpu().numpy(); out[ch] = s
        return out

    def jacobian(self, key, u, channels=None):
        md = self.models[key]; out = {}
        for ch in (channels or self.channels):
            if ch not in md["m"]: continue
            ut = torch.as_tensor(np.asarray(u, float)[None], dtype=T.F64, device=T.DEV).requires_grad_(True)
            y = self.logratio(key, ut, [ch], grad=True)[ch][0]
            Jc = torch.stack([torch.autograd.grad(y[c], ut, retain_graph=True)[0][0] for c in range(len(y))], 1)
            J = np.full((self.nk, len(self.e_ev)), np.nan); J[:, md["mask"][ch]] = Jc.detach().cpu().numpy(); out[ch] = J
        return out

    def tmc(self, key, n=100_000, seed=0, chunk=100_000, channels=None):
        """uniform knob prior: per channel mean, sd, quantiles (2.5/16/50/84/97.5 %), relative covariance (54 x 54, NaN rows
        off the modelled cells), log10 mean/sd; computed on the GPU in chunks (exact accumulation of first/second moments)"""
        g = torch.Generator(device=T.DEV).manual_seed(seed); md = self.models[key]; res = {}
        acc = {}
        for i in range(0, n, chunk):
            U = torch.rand(min(chunk, n - i), self.nk, generator=g, device=T.DEV, dtype=T.F64) * 2 - 1
            for ch, Y in self.logratio(key, U, channels).items():
                s0 = torch.as_tensor(md["centre"][ch][md["mask"][ch]], dtype=T.F64, device=T.DEV); S = s0 * 10.0 ** Y
                a = acc.setdefault(ch, {"n": 0, "s": 0, "ss": 0, "q": []})
                a["n"] += S.shape[0]; a["s"] = a["s"] + S.sum(0); a["ss"] = a["ss"] + S.T @ S
                a["q"].append(S[:20000])
        for ch, a in acc.items():
            mu = a["s"] / a["n"]; C = a["ss"] / a["n"] - torch.outer(mu, mu); C = C * a["n"] / (a["n"] - 1)
            R = C / torch.outer(mu, mu); Q = torch.quantile(torch.cat(a["q"]), torch.tensor([.025, .16, .5, .84, .975], dtype=T.F64, device=T.DEV), 0)
            mask = self.models[key]["mask"][ch]; nE = len(self.e_ev)
            full = lambda v: (lambda o: (o.__setitem__(mask, v), o)[1])(np.full(nE, np.nan))
            rc = np.full((nE, nE), np.nan); rc[np.ix_(mask, mask)] = R.cpu().numpy()
            res[ch] = {"mean": full(mu.cpu().numpy()), "sd": full(torch.sqrt(torch.diagonal(C)).cpu().numpy()),
                       "q": np.stack([full(q) for q in Q.cpu().numpy()]), "relcov": rc}
        return res

    def batch(self, keys, U, channels=None):
        """same U (n, 12) on many nuclei -> {key: {channel: (n, cells) log10 ratio torch}}"""
        Ut = torch.as_tensor(np.asarray(U, float), dtype=T.F64, device=T.DEV)
        return {k: self.logratio(k, Ut, channels) for k in keys}

    def bench(self, n_per_nucleus=10_000, keys=None):
        keys = keys or self.nuclei
        U = torch.rand(n_per_nucleus, self.nk, device=T.DEV, dtype=T.F64) * 2 - 1
        if T.DEV == "cuda": torch.cuda.synchronize()
        t0 = time.time(); tot = 0
        for k in keys:
            for ch, Y in self.logratio(k, U).items(): tot += Y.numel()
        if T.DEV == "cuda": torch.cuda.synchronize()
        w = time.time() - t0
        return {"points": n_per_nucleus * len(keys), "nuclei": len(keys), "cells": tot, "wall_s": w,
                "points_per_s": n_per_nucleus * len(keys) / w, "device": T.DEV}

    def verify_items(self, keys, n=10, seed=0, path="verify_items.txt", tag="ev"):
        rng = np.random.Generator(np.random.PCG64(seed)); rows = []
        for key in keys:
            for i, u in enumerate(rng.uniform(-1, 1, (n, self.nk))):
                rows.append(f"{tag}{i:05d}_{key} {int(key[:3])} {int(key[4:])} " + ",".join(f"{x:.6f}" for x in u))
        Path(path).write_text("\n".join(rows) + "\n"); return path

    def compare(self, outdir):
        rows = []
        CH = CH_NPZ
        for d in sorted(Path(outdir, "runs").iterdir()):
            f = list(d.glob("*.npz"))
            if not f: continue
            th = json.load(open(d / "theta.json")); key = f"{th['Z']:03d}_{th['A']:03d}"
            if key not in self.models: continue
            z = np.load(f[0], allow_pickle=True); S = self.sigma(key, np.array([th["u"]]))
            for ch, s in S.items():
                ex = z[CH[ch]].astype(float) if CH[ch] in z.files else np.zeros(len(self.e_ev))
                ok = np.isfinite(s[0]) & (ex > 1e-3); e = np.log10(s[0][ok]) - np.log10(ex[ok])
                rows.append({"rid": d.name, "nuc": key, "ch": ch, "cells": int(ok.sum()),
                             "med_dex": float(np.median(np.abs(e))) if ok.any() else np.nan,
                             "max_dex": float(np.max(np.abs(e))) if ok.any() else np.nan})
        return rows


def load(d=None):
    return Emu(d)
