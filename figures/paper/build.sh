#!/usr/bin/env bash
# Builds the figures of the Reverb paper into this directory: the TikZ sources in src/ into PDF with LuaLaTeX (the
# figures use DejaVu Sans, the font of the data panels, through fontspec) and a PNG preview, then the data panels
# (make_paper_figures.py).  The PDFs carry no date, so two builds give the same bytes.
#   bash figures/paper/build.sh [NAME ...]   (no name: every figure; texlive/2024 on Perlmutter; ghostscript)
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH=/global/common/software/nersc9/texlive/2024/bin/x86_64-linux:$PATH
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
if [ $# -gt 0 ]; then sources=$(for n in "$@"; do echo "$here/src/$n.tex"; done); else sources=$(ls "$here"/src/*.tex); fi
for source in $sources; do
  name=$(basename "$source" .tex)
  [[ "$name" == reverb_style ]] && continue
  (cd "$here/src" && lualatex -interaction=nonstopmode -halt-on-error -output-directory "$build" "$name.tex" \
     >"$build/$name.out" 2>&1) || { grep -m5 '^!' "$build/$name.log" >&2; exit 1; }
  cp "$build/$name.pdf" "$here/$name.pdf"
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r220 -sOutputFile="$here/$name.png" "$here/$name.pdf"
done
[ $# -gt 0 ] && exit 0
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3}
RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 "$PY" "$here/make_paper_figures.py"
# the Evaluation panels: 3.5 GHz slots, GPU runs at the 11.0 ms deadline (the scheduler variants exist only there)
RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 "$PY" "$here/make_eval_figures.py"
