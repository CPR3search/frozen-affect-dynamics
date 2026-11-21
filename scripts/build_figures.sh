#!/usr/bin/env bash
# Compile each figure body written by src/affectdyn/reporting/make_tables.py (out/figures/*_fig.tex) to a standalone PDF.
# Usage: scripts/build_figures.sh [output-dir]
set -euo pipefail
D="${1:-out}"
command -v pdflatex >/dev/null || { echo "pdflatex not found; install a TeX distribution (TeX Live or MiKTeX)" >&2; exit 1; }
if [ ! -d "$D/figures" ] || ! ls "$D"/figures/*_fig.tex >/dev/null 2>&1; then
  echo "no figure bodies in $D/figures; run python -m affectdyn.reporting.make_tables first" >&2; exit 1
fi
cd "$D/figures"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
for body in *_fig.tex; do
  stem="${body%.tex}"
  # full-width figures use the two-column page width; column figures the single-column text width
  if [ -f "../tables/$stem.tex" ] && grep -q 'begin{figure\*}' "../tables/$stem.tex"; then W=522pt; F=252pt; else W=345pt; F=345pt; fi
  # widen only the crop box for figures marked CROP-WIDE
  VW=$W; grep -q '% CROP-WIDE' "$body" && VW=372pt
  cat > "$TMP/$stem.tex" <<TEX
\documentclass[10pt,border=1pt,varwidth=$VW]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usepackage{fontawesome5}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\pgfplotsset{every axis/.append style={font=\footnotesize, label style={font=\footnotesize},
  tick label style={font=\footnotesize}, legend style={font=\footnotesize, draw=none, fill=none},
  title style={font=\small}, every axis title/.append style={font=\small}}}
\usepackage{xcolor}
\definecolor{cImage}{HTML}{0072B2}\definecolor{cVideo}{HTML}{D55E00}\definecolor{cMark}{HTML}{CC79A7}
\definecolor{cAccent}{HTML}{009E73}\definecolor{cBandF}{HTML}{DCE6EE}
\newlength{\figw}\setlength{\figw}{$F}\setlength{\textwidth}{$W}
\graphicspath{{$PWD/}}
\begin{document}
\input{$PWD/$body}
\end{document}
TEX
  (cd "$TMP" && pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" > "$stem.out" 2>&1) \
    || { echo "FAILED $stem" >&2; tail -20 "$TMP/$stem.log" 2>/dev/null || tail -20 "$TMP/$stem.out"; exit 1; }
  cp "$TMP/$stem.pdf" "$stem.pdf"
  echo "  $D/figures/$stem.pdf"
done
