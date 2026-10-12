import os
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
# Optional tree of older job code for private equivalence tests (INCOGNITA_EMU_LEGACY); skipped when unset.
LEGACY = Path(os.path.expanduser(os.environ["INCOGNITA_EMU_LEGACY"])) if os.environ.get("INCOGNITA_EMU_LEGACY") else None


def models_path(which=None):
    """which=None: the configured default (models/default, floor 1e-3 mb); "legacy": the older floor-1e-5 models."""
    from incognita_emu import config
    p = (REPO / "models" / which if which else config.models_dir()) / "models.pt"
    return p if p.exists() else None


def legacy(sub):
    p = LEGACY / sub if LEGACY else None
    if p is None or not p.exists(): pytest.skip(f"legacy code not present: {p}")
    return p
