#!/usr/bin/env python3
"""Persistent warm engine workers.

One shared `run_item()` core with C-path semantics (reset fitlib cards + chartrun target caches, new_cascade, ChainedFull,
engine_c.run, ingredient dump, same npz keys). Workers are spawned once (threads=1, pinned) and take items until recycled after
`recycle_items` items or RSS > `recycle_rss_gb`. `recycle_items=1` is the one-process-per-item mode (the A/B flag, --mode fresh): same code,
a new spawned interpreter per item.

CLI:  engine_pool.py --repo REPO --items items.json --cpus 9,9 [--mode warm|fresh] [--recycle 0] [--rss-gb 2.5] [--stats-out f.json]
items.json: [[out_dir, Z, A, lines_file, grid_json, extra_lines, tag, env_dict], ...]; env values None = unset.
Each npz also records cpu_s (item CPU + the worker start-up CPU on the worker's first item), run_cpu_s (item only), startup_cpu_s/wall_s,
worker_pid, worker_item_index, worker_rss_gb. Items whose npz exists are skipped; a failure writes err_ZZZ_AAA.txt."""
import argparse, json, os, sys, threading, time, traceback, queue
from multiprocessing import get_context
from pathlib import Path

PREFIX = os.environ.get("INCOGNITA_EMU_ENGINE_PREFIX", "INCOGNITA")


def ev(name):
    """engine environment switch `name` with the engine's prefix"""
    return f"{PREFIX}_{name}"


META = {"cpu_s", "wall_s", "run_cpu_s", "startup_cpu_s", "startup_wall_s", "worker_pid", "worker_item_index", "worker_rss_gb", "tag", "box"}


def _rss_gb():
    try:
        with open("/proc/self/statm") as fh: return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except OSError: return 0.0


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _setup(repo, cpus, engphys):
    """worker start-up: threads=1, pin, imports. Returns the namespace of modules and the baseline env."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
    os.environ[ev("TALYS_FIT")] = "nodata"; os.environ.pop(ev("TALYS_FIT_LINES"), None)
    for k in ("ENGPHYS", "DIRCAP", "SOUKHO_EFCLAMP", "DEF_EVENEVEN", "PESPIN_NORM"): os.environ.pop(ev(k), None)
    if engphys: os.environ[ev("ENGPHYS")] = engphys
    if cpus:
        try: os.sched_setaffinity(0, set(cpus))
        except Exception: pass
    sys.path.insert(0, repo); os.chdir(repo)
    import numpy as np, torch; torch.set_num_threads(1)
    from physics.hf import chartrun, warmrun
    from physics.hf.engine import ChainedFull
    from physics.hf import engine_c
    from physics.hf.ingredients import ingredients as ingredient_dump
    from physics.hf.input import fitlib
    engine_c.lib()
    return dict(np=np, torch=torch, chartrun=chartrun, warmrun=warmrun, ChainedFull=ChainedFull, engine_c=engine_c,
                ingredient_dump=ingredient_dump, fitlib=fitlib), dict(os.environ)


def run_item(M, base_env, it, repo, engphys, idx, startup):
    """one item, C-path semantics. Environment is restored to the worker baseline first, so a warm worker sees exactly the
    environment a fresh process would."""
    np, torch, fitlib, chartrun, warmrun = M["np"], M["torch"], M["fitlib"], M["chartrun"], M["warmrun"]
    out, Z, A, lf, gj, extra, tag = it[:7]; env = it[7] if len(it) > 7 and it[7] else {}
    p = Path(out) / f"{Z:03d}_{A:03d}.npz"
    if p.exists(): return "skip"
    os.environ.clear(); os.environ.update(base_env)
    for k, v in env.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = str(v)
    os.environ[fitlib.ENV_LINES_FILE] = lf
    if extra: os.environ[ev("TALYS_FIT_LINES")] = extra
    else: os.environ.pop(ev("TALYS_FIT_LINES"), None)
    grid = np.asarray(json.load(open(gj)), float); e_mev = tuple(float(np.float32(e)) for e in grid / 1e6)
    t0 = time.process_time(); w0 = time.time()
    try:
        fitlib.deactivate(); fitlib._cards.cache_clear(); chartrun.drop_target_caches()
        cas = warmrun.new_cascade(Z, A, e_mev)
        with torch.inference_mode():
            res = M["engine_c"].run(injection=M["ChainedFull"](Z=Z, A=A, declared_energies=e_mev, cascade=cas))
        c = fitlib.active()
        cards = "" if c is None else json.dumps({"options": list(c.options), "cells": [list(x[:4]) for x in c.cells],
                                                 "particles": [list(x) for x in c.particles], "globals": [list(x) for x in c.globals_],
                                                 "skipped": [list(x) for x in c.skipped], "source": list(c.source)})
        ex = {"ch_" + k: v.detach().cpu().numpy().astype(np.float64) for k, v in res.channels_mb.items()}
        tm = getattr(res, "totals_mb", None)
        if isinstance(tm, dict): ex.update({"tot_" + str(k): v.detach().cpu().numpy().astype(np.float64) for k, v in tm.items() if hasattr(v, "detach")})
        for pre, dd in (("lv_", getattr(res, "levels_mb", None)), ("", getattr(res, "residual_production_mb", None))):
            if isinstance(dd, dict):
                for k, v in dd.items():
                    if hasattr(v, "detach"):
                        a = v.detach().cpu().numpy().astype(np.float32)
                        if np.any(a > 0): ex[pre + str(k)] = a
        ex["path"] = "c"
        try: ex.update(M["ingredient_dump"](cas, res))
        except Exception as exc: open(Path(out) / f"ingerr_{Z:03d}_{A:03d}.txt", "w").write(f"{exc}\n{traceback.format_exc()}")
        run_cpu = time.process_time() - t0
        tmp = Path(out) / f".{Z:03d}_{A:03d}.tmp.npz"
        np.savez_compressed(tmp, capture_mb=res.channels_mb["xs000000"].detach().cpu().numpy().astype(float), e_ev=grid, arm="nodata",
                            engphys=engphys, extra_lines=extra, engine_repo=os.path.basename(os.path.normpath(repo)), lines_file=os.path.basename(lf), lines_sha256=_sha(lf), cards=cards, tag=tag,
                            cpu_s=run_cpu + (startup["cpu"] if idx == 0 else 0.0), run_cpu_s=run_cpu,
                            wall_s=time.time() - w0 + (startup["wall"] if idx == 0 else 0.0), startup_cpu_s=startup["cpu"],
                            startup_wall_s=startup["wall"], worker_pid=os.getpid(), worker_item_index=idx, worker_rss_gb=_rss_gb(), **ex)
        os.replace(tmp, p)
        return "ok"
    except Exception as exc:
        open(Path(out) / f"err_{Z:03d}_{A:03d}.txt", "w").write(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return "err"
    finally:
        cas = None; chartrun.drop_target_caches()


def _worker(conn, repo, cpus, engphys, recycle_items, rss_gb, t_spawn):
    w_entry, c_entry = time.time(), time.process_time()
    M, base_env = _setup(repo, cpus, engphys)
    startup = {"cpu": time.process_time() - c_entry + 0.0, "wall": time.time() - t_spawn}   # wall includes the spawn latency
    idx = 0
    conn.send(("ready", os.getpid()))
    while True:
        it = conn.recv()
        if it is None: return
        st = run_item(M, base_env, it, repo, engphys, idx, startup); idx += 1
        recycle = (recycle_items and idx >= recycle_items) or _rss_gb() > rss_gb
        conn.send((st, bool(recycle), _rss_gb()))
        if recycle: return


class EnginePool:
    """N slots; each slot owns one spawned worker process and respawns it when it recycles."""
    def __init__(self, repo, cpus, mode="warm", recycle_items=0, recycle_rss_gb=2.5, engphys="dircap,pespin"):
        self.repo = os.path.expanduser(repo); self.cpus = list(cpus); self.engphys = engphys
        self.recycle_items = 1 if mode == "fresh" else int(recycle_items); self.rss = float(recycle_rss_gb)
        self.stats = {"spawns": 0, "items": 0, "recycled_rss": 0}; self._lock = threading.Lock(); self.retries = {}

    def _slot(self, k, q):
        ctx = get_context("spawn"); proc = conn = None; ec = None; retries = self.retries
        try:
            while True:
                try: it = q.get_nowait()
                except queue.Empty: break
                if proc is None:
                    conn, child = ctx.Pipe(); t_spawn = time.time()
                    proc = ctx.Process(target=_worker, args=(child, self.repo, [self.cpus[k % len(self.cpus)]], self.engphys, self.recycle_items, self.rss, t_spawn))
                    proc.start(); child.close()
                    with self._lock: self.stats["spawns"] += 1
                    try: conn.recv()
                    except EOFError: raise RuntimeError("worker died at start-up")
                conn.send(tuple(it))
                try: st, recycle, rss = conn.recv()
                except EOFError:   # SIGKILLed etc.: next item gets a fresh worker; the item is retried (max 2)
                    st, recycle = "died", True; proc.join(5); ec = proc.exitcode
                with self._lock: self.stats["items"] += 1
                if recycle: proc.join(); proc = conn = None
                if st == "died":
                    n = retries.get(tuple(it[:3]), 0); retries[tuple(it[:3])] = n + 1
                    print("worker died on", it[:3], "exitcode", ec, "retry", n + 1, flush=True)
                    with self._lock: self.stats["died"] = self.stats.get("died", 0) + 1
                    if n < 2: q.put(it)
        finally:
            if proc is not None:
                try: conn.send(None)
                except Exception: pass
                proc.join(30)

    def run(self, items):
        q = queue.Queue()
        for it in items: q.put(it)
        ts = [threading.Thread(target=self._slot, args=(k, q)) for k in range(len(self.cpus))]
        [t.start() for t in ts]; [t.join() for t in ts]
        return self.stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--cpus", default="0"); ap.add_argument("--mode", default="warm", choices=["warm", "fresh"])
    ap.add_argument("--recycle", type=int, default=3, help="warm: recycle after K items (0 = never; default 3: warm workers slow down with age)"); ap.add_argument("--rss-gb", type=float, default=2.5)
    ap.add_argument("--stats-out"); ap.add_argument("--no-sort", action="store_true", help="keep the items.json order (default: heaviest A first)")
    a = ap.parse_args()
    cpus = [int(x) for x in a.cpus.split(",") if x]
    items = [x for x in json.load(open(a.items)) if not (Path(x[0]) / f"{x[1]:03d}_{x[2]:03d}.npz").exists()]
    for o in {x[0] for x in items}: Path(o).mkdir(parents=True, exist_ok=True)
    if not a.no_sort: items = sorted(items, key=lambda t: -t[2])
    t0 = time.time()
    pool = EnginePool(a.repo, cpus, a.mode, a.recycle, a.rss_gb)
    st = pool.run(items); st.update(wall_s=time.time() - t0, n_items=len(items), mode=a.mode, cpus=cpus, load=os.getloadavg()[0])
    if a.stats_out: json.dump(st, open(a.stats_out, "w"))
    print("done", time.strftime("%H:%M"), json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
