"""Production engine settings used by the emulator runners: the deck options, the engine switches and the measured
S-function input. The S-function table is incognita_emu/meta/sfunc_ripl.csv (RIPL-3/RIPL-4 values) unless
INCOGNITA_EMU_SLTABLE names another csv (Z,A,S0,S1); INCOGNITA_EMU_SLTABLE=off switches the input off."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_pool as EP  # noqa: E402,F401
import hrule as RP  # noqa: E402,F401
ev = EP.ev
BASE = "preequilibrium 1.0;multipreeq 1.0;knockfix2 y;leeshell nm;ctwindow2 y;peunits y;fists2 y;sigavcont y"
FLAGS = {ev(k): v for k, v in (("TRITFID", "1"), ("RESISO", "1"), ("CCSOCB", "1"), ("MPEFIX", "1"), ("CPINC2", "1"),
                               ("SPECOUT", "1"), ("SPECOUT_DA", "0"), ("LEVJ", "1"), ("C6FIX", "1"), ("TLOGIT", "y"))}
PEON = BASE.replace("preequilibrium 1.0;multipreeq 1.0", "preequilibrium y;multipreeq y")
TABLE = os.environ.get("INCOGNITA_EMU_SLTABLE") or str(Path(__file__).resolve().parents[2] / "meta" / "sfunc_ripl.csv")
TABLE = "" if TABLE.lower() in ("off", "none", "0") else os.path.expanduser(TABLE)
FENV = {ev("SFUNC"): "u1", ev("SFUNC_TABLE"): TABLE, ev("SFUNC_SCOPE"): "first"} if TABLE else {}
DECK = {"P": PEON, "A": PEON}
