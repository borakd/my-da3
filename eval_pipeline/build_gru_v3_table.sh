#!/bin/bash
# Rebuild the 6-column GRU-v3 table with the display names the user specified
# on 2026-08-12. The names lived only in an ad-hoc command line before this;
# they are pinned here so the two still-training arms get the same treatment
# when they land.
#
# Reads the CACHED summary/averages_table.csv. Does NOT re-aggregate: every
# label below already scored 4292/4292 and re-reducing ~39k files per label
# costs ~85s each for numbers that cannot have changed.
#
# NAMING NOTES (all deliberate, do not "fix" without asking):
#   * "G2" is the user's choice for arms that are configured G3
#     (pose_gru_bptt AND pose_gru_e2e both True). Asked for explicitly.
#   * Rows carry an "R8" marker only where the user put one. Note this makes
#     the R lever unreadable from the name in the last five rows: f0np/f1np are
#     iters=1 while the ResNet/DINOv2/corr arms are iters=8, and none of them
#     say so.
#   * The corr arm (gru_a4g3f1cr8) had no name in the user's list -- it was the
#     12th row of a 12-row table against 11 names given. "Corr" below is my
#     fill-in, built to match the ResNet-18/DINOv2-S rows.
#   * The two pending rows are labelled with their ACTUAL concatenated feature
#     widths: ResNet-18 is 2x512 = 1024-D and DINOv2-S is 2x384 = 768-D
#     (config comments say so in as many words). The user's message said
#     "2048D" for both; 2048 is the pooled-CUT3R-token arm (2x1024), i.e. the
#     f1np row. Flip these back to 2048D here if that was intended after all.
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval}
OUT_TEX=${OUT_TEX:-$OUT_ROOT/tables/gru_v3_table.tex}
cd "$(dirname "$0")/.."

ROWS=(
  augfull_lr1e5="Regular CUT3R"
  gtray_lr1e5="Current GT CUT3R"
  prevpred_lr1e5="Previous Predicted CUT3R"
  gru_a4g3="Captain Gru A4 G2"
  gru_a4g3f1="Captain Gru A4 G2 F1"
  gru_a4g3r8="Captain Gru A4 G2 R8"
  gru_a4g3f1r8="Captain Gru A4 G2 F1 R8"
  gru_a4g3f0np="Captain Gru A4 G2 F0 (noproj, 1024D)"
  gru_a4g3f1np="Captain Gru A4 G2 F1 (noproj, 2048D)"
  gru_a4g3f1rr8="Captain Gru A4 G2 F1 ResNet-18"
  gru_a4g3f1dr8="Captain Gru A4 G2 F1 DINOv2-S"
  gru_a4g3f1cr8="Captain Gru A4 G2 F1 Corr"
)

# Appended automatically once each arm has a 4292-scene row in the cache.
#
# gru_a4g3oraclegt is NOT AN ARM: it is the A4 G3 checkpoint re-scored with
# ORACLE=gt, which feeds the GRU the CURRENT view's GT pose at its input.
# run_captain_ray_eval_node.sh:64-68 says such a run must never be aggregated
# beside honest arms; it is included here only because the user asked for it
# explicitly, and it is passed to --no_rank so it cannot take a top-3 colour
# away from a real arm.
PENDING=(
  gru_a4g3f1rr8np="Captain Gru A4 G2 F1 ResNet-18 (noproj, 1024D)"
  gru_a4g3f1dr8np="Captain Gru A4 G2 F1 DINOv2-S (noproj, 768D)"
  gru_a4g3oraclegt="Captain Gru A4 G2 Oracle GT -- diagnostic, not an arm"
)
NORANK=(gru_a4g3oraclegt)
for spec in "${PENDING[@]}"; do
  lb=${spec%%=*}
  if grep -q "^$lb," "$OUT_ROOT/summary/averages_table.csv"; then
    ROWS+=("$spec")
  else
    echo "note: $lb not aggregated yet -- row omitted" >&2
  fi
done

ARGS=()
for spec in "${ROWS[@]}"; do ARGS+=(--row "$spec"); done
for lb in "${NORANK[@]}"; do
  # only pass --no_rank for rows that actually made it into the table
  for spec in "${ROWS[@]}"; do
    [ "${spec%%=*}" = "$lb" ] && { ARGS+=(--no_rank "$lb"); break; }
  done
done

python eval_pipeline/build_lr1e5_table.py \
  --out_root "$OUT_ROOT" \
  "${ARGS[@]}" \
  --out_tex "$OUT_TEX" \
  --compile
