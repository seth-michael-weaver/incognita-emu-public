#!/bin/bash
# Build the TALYS structure directory the emulator's engine runs use (INCOGNITA_EMU_TALYS_DIR). TALYS structure data are
# not redistributed here: this fetches the `structure` tree of the public TALYS repository (sparse checkout, pinned commit)
# and builds a symlink view of it without the fitted level-density parameter files (*.ld_*), the local GDR parameters
# (gamma/gdr) and the local neutron optical potentials (optical/neutron), so the engine uses its global systematics.
# It also downloads RIPL-4 levels/levels-param.data (IAEA, pinned commit, sha256-checked) to OUT/ripl4/.
# usage: scripts/make_structure.sh [TALYS_SRC=~/opt/talys-src] [OUT=<repo>/cache/talys]   then export INCOGNITA_EMU_TALYS_DIR=OUT
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-$HOME/opt/talys-src}; OUT=${2:-$REPO/cache/talys}
TALYS_COMMIT=1f634cbcdc7003e38a6bb91d3d69e6b38dfbe234
if [ ! -d "$SRC/structure" ]; then
  git clone --filter=blob:none --sparse --no-checkout https://github.com/arjankoning1/talys "$SRC"
  git -C "$SRC" sparse-checkout set structure
  git -C "$SRC" checkout -q $TALYS_COMMIT
fi
R=$(cd "$SRC/structure" && pwd); O=$OUT/structure
rm -rf "$O"; mkdir -p "$O"
for d in $(ls "$R"); do case $d in density|gamma|optical|resonances) ;; *) ln -s "$R/$d" "$O/$d";; esac; done
mkdir -p "$O/density/ground"
for d in fission ph phjp; do ln -s "$R/density/$d" "$O/density/$d"; done
for d in $(ls "$R/density/ground"); do mkdir -p "$O/density/ground/$d"
  for f in $(ls "$R/density/ground/$d"); do case $f in *.ld_*) ;; *) ln -s "$R/density/ground/$d/$f" "$O/density/ground/$d/$f";; esac; done; done
mkdir -p "$O/gamma"; for d in $(ls "$R/gamma"); do [ $d = gdr ] || ln -s "$R/gamma/$d" "$O/gamma/$d"; done
mkdir -p "$O/optical"; for d in $(ls "$R/optical"); do [ $d = neutron ] || ln -s "$R/optical/$d" "$O/optical/$d"; done
RIPL4_COMMIT=db6a8c1db52493f2ccfa644c7861eb661d117835
LP_SHA=588d9ae956bc73e39a813536532f12fc962dd4cf773083d6039db3ddb14e0f21
sh256() { [ -f "$1" ] || return 0; if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
mkdir -p "$OUT/ripl4"; LP=$OUT/ripl4/levels-param.data
if [ "$(sh256 "$LP")" != $LP_SHA ]; then
  curl -fsSL -o "$LP" https://raw.githubusercontent.com/IAEA-NDS/RIPL-4/$RIPL4_COMMIT/levels/levels-param.data
  [ "$(sh256 "$LP")" = $LP_SHA ] || { echo "levels-param.data: sha256 mismatch" >&2; exit 1; }
fi
echo "TALYS structure view: $OUT   (export INCOGNITA_EMU_TALYS_DIR=$OUT)"
