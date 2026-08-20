#!/bin/bash
# Build summary_table_final: the campaign-closing paired GRU-pose / CUT3R-pose
# table over the full 4292-scene harness.
#
# Output: $OUT_ROOT/summary/summary_table_final.tex (+ .pdf/.png)
#
# Contents, in order: the four conditioning baselines (decreasing conditioning
# quality: none, own-frame GT, previous-frame GT, previous-frame prediction),
# the GRU focus-set arms (with their GRU-pose companion blocks), the 2026-08
# confidence-gate campaign arms (inference-time and gate-trained), and —
# via the PENDING mechanism — the external training-free baselines TTT3R and
# RayMap3R, which auto-append the day their 4292 rows land in
# summary/averages_table.csv (they are NOT on MN5 as of 2026-08-20; their
# evals live off-cluster and must be imported or re-run here first).
#
# Focus-table policy inherited: NO --no_rank — every row, including GT-fed
# diagnostics, competes for the top-3 cell shading, so the table shows how far
# honest arms sit from the GT-fed ceiling. Reads the CACHED averages_table.csv.
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval}
OUT_TEX=${OUT_TEX:-$OUT_ROOT/summary/summary_table_final.tex}
cd "$(dirname "$0")/.."

ROWS=(
  augfull_lr1e5="Regular CUT3R"
  gtray_lr1e5="Current GT CUT3R"
  prevgt_lr1e5="Previous GT CUT3R"
  prevpred_lr1e5="Previous Predicted CUT3R"
  gru_a4g3_rpt="Captain Gru A4 G2"
  gru_a4g3f1np_rpt="Captain Gru A4 G2 F1 (noproj, 2048D)"
  gru_a4g3f0np_rpt="Captain Gru A4 G2 F0 (noproj, 1024D)"
  gru_a4g3f1_rpt="Captain Gru A4 G2 F1"
  gru_a4g1f1np="Captain Gru A4 G1 F1 (noproj, 2048D)"
  gru_a4g1f1r8np="Captain Gru A4 G1 F1 R8 (noproj, 2048D)"
  gru_a4g3oraclegtfed="Captain Gru A4 G2 Oracle Trained (GT fed)"
  gru_a4g1oraclegtfed="Captain Gru A4 G1 Oracle Trained (GT fed)"
  augfull_cg_g7ema="Conf-Gate CUT3R (inference, g7ema)"
  augfull_cg_g7ema85="Conf-Gate CUT3R (inference, g7ema EMA .85)"
  cgtrain_g7ema="Gate-Trained CUT3R (gate-on eval)"
  cgtrain_plain="Gate-Trained CUT3R (plain eval)"
  augfull_cg_g7ema_bwd="Conf-Gate CUT3R backward pass"
  augfull_cg_fuse_g7="Conf-Gate CUT3R fwd+bwd fusion"
)

# External training-free baselines on the SAME regular-finetune checkpoint.
# Their 4292 evals are not on this cluster yet; rows appear automatically
# once aggregated into averages_table.csv under these labels.
PENDING=(
  ttt3r_augfull="TTT3R (training-free, regular ckpt)"
  raymap3r_augfull="RayMap3R (training-free, regular ckpt)"
)

AGG=$OUT_ROOT/summary/averages_table.csv
for spec in "${PENDING[@]}"; do
  lb=${spec%%=*}
  if grep -q "^$lb," "$AGG"; then
    ROWS+=("$spec")
  else
    echo "note: $lb not aggregated yet -- row omitted" >&2
  fi
done

ARGS=()
for spec in "${ROWS[@]}"; do ARGS+=(--row "$spec"); done

python eval_pipeline/build_gru_pose_table.py \
  --out_root "$OUT_ROOT" \
  --out_tex "$OUT_TEX" \
  --tight --compile \
  "${ARGS[@]}"
echo "wrote $OUT_TEX (+ .pdf/.png)"
