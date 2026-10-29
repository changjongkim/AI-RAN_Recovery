#!/usr/bin/env bash
# Builds the figures of the Reverb paper into this directory: the TikZ sources in src/ into PDF (and a PNG preview),
# then the data panels (make_paper_figures.py).  The PDFs carry no date, so two builds give the same bytes.
#   bash figures/paper/build.sh        (texlive/2024 on Perlmutter; ghostscript for the previews)
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
type module >/dev/null 2>&1 && module load texlive/2024 >/dev/null 2>&1
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
for source in "$here"/src/*.tex; do
  name=$(basename "$source" .tex)
  [[ "$name" == reverb_style ]] && continue
  (cd "$here/src" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$build" "$name.tex" \
     >"$build/$name.out" 2>&1) || { grep -m5 '^!' "$build/$name.log" >&2; exit 1; }
  cp "$build/$name.pdf" "$here/$name.pdf"
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r220 -sOutputFile="$here/$name.png" "$here/$name.pdf"
done
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3}
"$PY" "$here/make_paper_figures.py"
