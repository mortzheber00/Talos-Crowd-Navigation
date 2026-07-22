#!/usr/bin/env bash
# Host-side: rebuild every figure, ensemble, and the slide deck from the current
# sweep results. Run after a full run_sweep pass. The sim runs in the container;
# everything here (matplotlib, scipy, convert, Pillow) runs on the host.
set -e
cd "$(dirname "$0")/.."   # repo root (worktree)
FIGS=experiments/figures
WEBP=$FIGS/webp
mkdir -p "$WEBP"

echo "== 1. figures =="
python3 experiments/analyze.py

echo "== 2. png -> webp =="
for png in "$FIGS"/*.png; do
  [ -e "$png" ] || continue
  base=$(basename "$png" .png)
  convert "$png" -quality 82 "$WEBP/$base.webp"
done
echo "   $(ls "$WEBP"/*.webp 2>/dev/null | wc -l) webp"

echo "== 3. ensembles (from deck_scenarios only) =="
python3 experiments/make_ensemble.py --dirs deck_scenarios --sensor gt --poster \
  --out "$FIGS/gifs/ensemble_all.gif"
python3 experiments/make_ensemble.py --dirs deck_scenarios --sensor perception --poster \
  --out "$FIGS/gifs/ensemble_perception.gif"

echo "== 4. nav gifs =="
python3 experiments/make_gifs.py

echo "== 5. deck =="
python3 experiments/build_deck.py

echo "== done =="
