#!/usr/bin/env python3
"""Fetch the large files (models, run extracts, engine input decks) listed in MANIFEST.json into this repo.

Default source: the GitHub release (MANIFEST "release" tag) of the repository this checkout was cloned from (git remote
`origin`), or of INCOGNITA_EMU_RELEASE_URL (".../releases/download/<tag>/") if set; each file is checked against its sha256. `--from DIR` takes the files from a local
directory instead (same file names as the release, e.g. a copy of the release assets). Assets with "unpack": "tar.gz" are
unpacked to their dest directory after the check.
Groups: core (default models, legacy models, verification runs; enough for the API and the tests), previous (the 0.1.0
default models), train (12-knob training runs), knobs16 (runs for the 16-knob set), extras (sensitivities, MF33 demo), engine (input decks for exact runs).
usage: python scripts/fetch_assets.py [--group core|previous|train|knobs16|extras|engine|all] [--only NAME,...] [--from DIR] [--check]"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tarfile, tempfile, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def release_url(M):
    u = os.environ.get("INCOGNITA_EMU_RELEASE_URL")
    if u: return u.rstrip("/") + "/"
    r = subprocess.run(["git", "-C", str(REPO), "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
    m = re.match(r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+)/([^/]+?)(?:\.git)?/?$", r)
    if not m:
        raise RuntimeError("cannot derive the release URL from git remote 'origin'; set INCOGNITA_EMU_RELEASE_URL or use --from DIR")
    return f"https://github.com/{m.group(1)}/{m.group(2)}/releases/download/{M['release']}/"


def present(e):
    d = REPO / e["dest"]
    if e.get("unpack"): return d.is_dir() and any(d.iterdir())
    return d.is_file() and sha256(d) == e["sha256"]


def place(e, f):
    """f = verified downloaded/copied file -> dest (unpacked if needed)"""
    d = REPO / e["dest"]; d.parent.mkdir(parents=True, exist_ok=True)
    if d.is_symlink() or d.is_file(): d.unlink()
    if e.get("unpack") == "tar.gz":
        if d.is_dir(): shutil.rmtree(d)
        with tarfile.open(f) as t:
            top = {m.name.split("/")[0] for m in t.getmembers()}
            if top != {d.name} or any(m.name.startswith("/") or ".." in m.name.split("/") for m in t.getmembers()):
                raise RuntimeError(f"{e['file']}: unexpected archive layout {sorted(top)}")
            try: t.extractall(d.parent, filter="data")
            except TypeError: t.extractall(d.parent)
    else:
        shutil.move(str(f), d)


def local_sources(e):
    """private mirrors (MANIFEST.local.json, not part of the public repo): local paths or ssh host aliases"""
    p = REPO / "MANIFEST.local.json"
    if not p.exists(): return []
    L = json.load(open(p)); out = []
    for s in L.get("sources", {}).get(e["name"], []):
        loc = Path(os.path.expanduser(s["path"]))
        if loc.exists(): out.append(("local", loc))
        else:
            t = os.environ.get(f"INCOGNITA_EMU_HOST_{s['host'].upper()}", L.get("hosts", {}).get(s["host"]))
            if t: out.append(("ssh", f"{t}:{s['path']}"))
    return out


def fetch(e, src_dir=None, base=None):
    if present(e): return "present"
    with tempfile.TemporaryDirectory(dir=REPO) as td:
        tmp = Path(td) / e["file"]
        if src_dir:
            shutil.copy2(Path(src_dir) / e["file"], tmp)
        else:
            for kind, s in local_sources(e):
                d = REPO / e["dest"]
                if kind == "local" and s.is_dir():            # private mirror of an unpacked tree
                    if d.is_symlink() or d.is_file(): d.unlink()
                    elif d.is_dir(): shutil.rmtree(d)
                    shutil.copytree(s, d); return f"copied tree from {s}"
                if kind == "local": shutil.copy2(s, tmp)
                else: subprocess.run(["rsync", "-a", s, str(tmp)], capture_output=True)
                if tmp.exists() and sha256(tmp) == e["sha256"]:
                    place(e, tmp); return f"mirror {s}"
            url = base() + e["file"]
            print(f"  downloading {url} ({e['size'] / 1e6:.1f} MB)", flush=True)
            urllib.request.urlretrieve(url, tmp)
        h = sha256(tmp)
        if h != e["sha256"]: return f"FAILED: sha256 mismatch ({h[:12]}... != {e['sha256'][:12]}...)"
        place(e, tmp)
    return "fetched"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--group", default="core"); ap.add_argument("--only", default="")
    ap.add_argument("--from", dest="src", default=None); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); M = json.load(open(REPO / "MANIFEST.json")); bad = 0
    base = lambda: release_url(M)
    for e in M["assets"]:
        if a.only:
            if e["name"] not in a.only.split(","): continue
        elif a.group != "all" and e["group"] != a.group: continue
        try: st = ("ok" if present(e) else "absent") if a.check else fetch(e, a.src, base)
        except Exception as ex: st = f"FAILED: {ex}"
        bad += st.startswith(("FAILED", "absent")); print(f"{e['name']:16s} {st}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
