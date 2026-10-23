# incognita-emu

Fast emulators of the Incognita nuclear-reaction engine (a Python/C port of TALYS; repository `incognita` under the
same GitHub account as this one). For 185 target nuclei (Z = 13–96) they return the engine's neutron-induced cross sections from 1 keV to
20 MeV (total, elastic, nonelastic, capture, inelastic, (n,2n), (n,p), (n,α), fission) at any point of 12 global model
knobs, in milliseconds, with Jacobians and Monte Carlo sampling on a GPU or CPU.

They emulate the engine, not measurements: an error of 0.01 dex means 0.01 dex from what the engine would compute.

Code: Apache-2.0 · Data and models: CC BY 4.0 · Version 0.4.0 · Trained on engine release `engine-2026-10-19`

New in 0.4.0: the default models are retrained on engine release `engine-2026-10-19` (every nucleus re-run); the API is
unchanged. The 0.1.0 default models (engine `engine-2026-10`) stay available as `models/v0_1` (`--group previous`).

## How to use

### Install

Python ≥ 3.10 with numpy, scipy, torch, pandas and pyarrow. With [uv](https://docs.astral.sh/uv/) (or use
`python -m venv .venv` and `pip install -e ".[test]"`):

```bash
git clone <repository-url> incognita-emu && cd incognita-emu
uv venv && . .venv/bin/activate
uv pip install -e ".[test]"
python scripts/fetch_assets.py      # models + verification runs from this repository's GitHub release (~140 MB, sha256-checked)
pytest                              # checks the models against 20 exact engine runs they were not fitted on
```

`fetch_assets.py --group previous` (the 0.1.0 default models), `--group train` (training runs), `--group extras`
(sensitivities, MF33 demo), `--group engine` (engine input decks) or `--group all` download the rest of the release.

### Python

```python
import numpy as np
from incognita_emu import api

E = api.load()                                   # models/default/models.pt
print(len(E.nuclei), "nuclei;", E.channels)      # keys are "ZZZ_AAA", e.g. "026_056" = Fe-56
u = np.zeros(12)                                 # knob point, each knob in [-1, 1]; 0 = engine defaults
u[4] = 0.5                                       # e.g. E1 strength half-way to its +25 % prior edge
S = E.sigma("026_056", u[None, :])               # {channel: (1, 54) cross section in mb}, NaN where not modelled
i = np.searchsorted(E.e_ev, 1e6)
print(f"Fe-56 capture at {E.e_ev[i]/1e6:.1f} MeV: {S['capture'][0, i]:.3f} mb")
J = E.jacobian("026_056", u, ["capture"])        # {channel: (12, 54)} d log10(sigma) / du
print("most influential knob at 1 MeV:", E.knobs[int(np.nanargmax(np.abs(J["capture"][:, i])))])
T = E.tmc("026_056", n=100_000, channels=["capture"])   # uniform knob prior: mean, sd, quantiles, relcov
print("capture relative sd at 1 MeV:", round(float(T["capture"]["sd"][i] / T["capture"]["mean"][i]), 3))
```

`E.e_ev` holds the 54 energies (eV). `E.batch(keys, U)` evaluates many nuclei at many points. `E.verify_items(...)`
writes an item file of exact engine runs to check the emulator against (see below).

### Knobs

Each knob takes one value for every nucleus in a run; u = ±1 is the edge of its prior range.

| # | knob | engine keyword | range |
|---|---|---|---|
| 0 | sA | alphald, betald (×) | ±5 % |
| 1 | dC | ctable level-density shift (ln ρ) | ±0.15 |
| 2 | dPs | pshiftconstant (MeV) | ±0.30 |
| 3 | rsc | rspincut | ×1.15^±1 |
| 4 | fE1 | ftableadjust e1 | ×1.25^±1 |
| 5 | wE1 | wtableadjust e1 | ×1.10^±1 |
| 6 | fM1 | ftableadjust m1 | ×1.50^±1 |
| 7 | nv1 | v1adjust n, g | ±2 % |
| 8 | nrv | rvadjust n, g | ±1 % |
| 9 | nav | avadjust n, g | ±5 % |
| 10 | nd1 | d1adjust n, g | ±10 % |
| 11 | m2c | m2constant | ×1.25^±1 |

`incognita_emu.knobs` also defines four extra knobs (proton and α optical-potential depth, two fission-path
parameters); the release contains engine runs for them but no 16-knob model.

### Exact engine runs (optional)

Needed only to check the emulator, to add nuclei or to retrain. Install the engine at the release the models were
trained on (tag `engine-2026-10-19`; needs [uv](https://docs.astral.sh/uv/), and gcc for the optional C kernels), then
build the TALYS structure view (TALYS data are not redistributed; this downloads the `structure` tree of the TALYS
repository, several GB, plus one RIPL-4 file):

```bash
bash scripts/get_engine.sh ../incognita-engine             # clone at engine-2026-10-19, uv sync, build C kernels
bash scripts/make_structure.sh                             # -> cache/talys
python scripts/fetch_assets.py --group engine              # engine input decks -> data/lines_P
export INCOGNITA_EMU_ENGINE=$PWD/../incognita-engine INCOGNITA_EMU_TALYS_DIR=$PWD/cache/talys
pytest -m engine                                           # Fe-56 centre run, bit-identical to the training run
```

Run a list of knob points on one machine (four workers) and extract the results:

```bash
python -c "from incognita_emu import api; api.load().verify_items(['026_056'], n=4, path='items.txt')"
ITEMS=items.txt OUT=runs/check bash scripts/pool.sh 4
TAG=check ITEMS=items.txt OUT=runs/check DATA=data/ext bash scripts/finish.sh   # -> data/ext/ext_check_0.npz
python -c "from incognita_emu import api; print(api.load().compare('runs/check'))"
```

`pool.sh` can also share one queue between several machines over ssh (`QHOST=user@host`, see the script header).

### Configuration

| variable | meaning |
|---|---|
| `INCOGNITA_EMU_MODELS` | directory with models.pt (default `models/default`; older models: `models/v0_1`, `models/legacy`) |
| `INCOGNITA_EMU_DATA` | directory of run extracts (default `data`) |
| `INCOGNITA_EMU_ENGINE` | engine checkout with its `.venv` |
| `INCOGNITA_EMU_TALYS_DIR` | TALYS structure view from `scripts/make_structure.sh` |
| `INCOGNITA_EMU_PLINES` | engine input decks (default `data/lines_P`) |
| `INCOGNITA_EMU_CC_CACHE` | coupled-channels cache (default `cache/cc`); speed only, results are identical |
| `INCOGNITA_EMU_SLTABLE` | measured S-function table (default `incognita_emu/meta/sfunc_ripl.csv`; `off` to disable) |
| `INCOGNITA_EMU_PY` | python of the engine (default `$INCOGNITA_EMU_ENGINE/.venv/bin/python`) |

### New nuclei and retraining

A nucleus is emulated only after its own engine runs exist (one centre, 24 axis and ≥ 32 Sobol runs); the models do
not transfer between nuclei. Engine input decks exist for 251 nuclei (`data/lines_P`).

```bash
python -m incognita_emu.design nucleus --key 030_068 --nsob 32 --out runs/new   # -> runs/new/items_030_068.txt
ITEMS=runs/new/items_030_068.txt OUT=runs/new bash scripts/pool.sh 4
TAG=new ITEMS=runs/new/items_030_068.txt OUT=runs/new DATA=data/ext bash scripts/finish.sh
python scripts/fetch_assets.py --group train
python -m incognita_emu.emu final --ext data/ext/train_a.npz,data/ext/train_b.npz,data/ext/train_c.npz,data/ext/train_d.npz,data/ext/train_e.npz,data/ext/train_f.npz,data/ext/ext_new_0.npz --out models/new
```

Add the new key to the nucleus list first (`INCOGNITA_EMU_SPLIT` = a JSON file with keys ALL, HN, VAL, TRIO; the
shipped one is `incognita_emu/meta/design_meta_stage1.json`). The training runs in the release were made with engine
`engine-2026-10`: `final` without the new extract reproduces the 0.1.0 models (`models/v0_1`; GPU, a few minutes), and runs
added to them should come from that engine release with its input decks (`lines_P.tar.gz` of release v0.1.0). The runs
behind the 0.4.0 default models are not in the release.

## Limits

The models cover the 185 nuclei in `incognita_emu/meta/design_meta_stage1.json`, the 54 energies from 1 keV to 20 MeV
and knob points with |u| ≤ 1 (beyond that they extrapolate, unchecked). On engine runs not used for fitting, the error
is within 0.003 dex (median) and 0.015 dex (95th percentile) on every channel, near reaction thresholds included.
Cells below 1e-3 mb are not modelled (NaN). The older `models/legacy` set is worse for
heavy-nucleus (n,α) near threshold (up to 0.9 dex for Au-197). The MF33 demo propagates the knob prior only and is not a
calibrated covariance; it and the sensitivities were made with engine `engine-2026-10`.
New exact runs reproduce the training runs bit for bit except for Ni-61, Sm-148 and Sm-154, whose
training used measured S-function values that cannot be redistributed (their new runs differ by up to 0.04–0.26 dex).

## License and credits

Code: Apache-2.0 (`LICENSE`). Models and data: CC BY 4.0 (`LICENSE-DATA`, which also lists third-party terms).
The engine is a port of TALYS (A.J. Koning, S. Hilaire, S. Goriely et al., MIT licence); see `NOTICE`. Measured
strength functions in `incognita_emu/meta/sfunc_ripl.csv` come from RIPL-3 and RIPL-4 (IAEA).
