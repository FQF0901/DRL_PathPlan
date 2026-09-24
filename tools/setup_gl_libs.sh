#!/usr/bin/env bash
# venv-local GL runtime libs: panda3d (MetaDrive) needs libGL.so.1; this headless box has
# only the NVIDIA EGL driver, so we unpack the glvnd runtime debs into .venv/gl (NO system change).
# Re-run after a fresh .venv re-creation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VGL="$ROOT/.venv/gl"
LIBDIR="$VGL/usr/lib/x86_64-linux-gnu"

if [ -e "$LIBDIR/libGL.so.1" ] && [ -e "$LIBDIR/libEGL.so.1" ]; then
  echo "glvnd libs already present: $LIBDIR"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
echo "downloading glvnd runtime debs (libgl1 libegl1 libglvnd0 libglx0)..."
apt-get download libgl1 libegl1 libglvnd0 libglx0
mkdir -p "$VGL"
for d in ./*.deb; do dpkg -x "$d" "$VGL"; done
echo "unpacked -> $VGL"
ls -1 "$LIBDIR" | head
