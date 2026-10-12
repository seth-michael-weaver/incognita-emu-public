#!/bin/bash
# Clone and build the Incognita engine at the release the models were trained on (tag engine-2026-10).
# The engine repository is `incognita` under the same GitHub account as this repository (derived from git remote origin),
# or INCOGNITA_EMU_ENGINE_URL. Needs git, uv (https://docs.astral.sh/uv/) and, for the optional C kernels, gcc/cc.
# usage: bash scripts/get_engine.sh [DEST=../incognita-engine]      then: export INCOGNITA_EMU_ENGINE=DEST
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd); DEST=${1:-$REPO/../incognita-engine}; TAG=engine-2026-10
URL=${INCOGNITA_EMU_ENGINE_URL:-$(git -C "$REPO" remote get-url origin | sed -E 's#^git@github\.com:#https://github.com/#; s#/[^/]+$#/incognita.git#')}
[ -d "$DEST/.git" ] || git clone --branch $TAG --depth 1 "$URL" "$DEST"
cd "$DEST"; uv sync
failed=""; for s in scripts/build_*.sh; do bash "$s" > /dev/null 2>&1 || failed="$failed $s"; done
[ -z "$failed" ] && echo "native kernels built" || echo "native kernels NOT built:$failed (the engine falls back to pure Python, much slower)"
echo "engine ready: export INCOGNITA_EMU_ENGINE=$(pwd)"
