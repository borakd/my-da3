#!/bin/bash
# Aggregate ONLY the labels named on the command line, then merge their rows
# into the existing summary/averages_table.csv instead of replacing it.
#
#   bash eval_pipeline/aggregate_and_merge.sh <label> [<label> ...]
#
# WHY THIS EXISTS: aggregate_results.py REWRITES averages_table.csv from
# scratch containing only the --labels it was given (aggregate_results.py:110).
# Run it for two new arms and the other twelve rows vanish, and the table build
# then dies with "label 'augfull_lr1e5' not in ...". The naive fix -- pass all
# fourteen labels -- re-reads ~39k small files per label off GPFS at ~85 s each,
# ~20 minutes to re-derive numbers that cannot have changed. So: aggregate the
# new labels only, and merge.
#
# per_scene_<label>.csv is the expensive artifact and is written per label, so
# it survives untouched. Only the merged table is rebuilt.
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval}
SUM=$OUT_ROOT/summary
TABLE=$SUM/averages_table.csv
[ $# -ge 1 ] || { echo "usage: $0 <label> [<label> ...]" >&2; exit 1; }
cd "$(dirname "$0")/.."

BAK=$TABLE.bak_$(date +%Y%m%d_%H%M%S)
cp "$TABLE" "$BAK"
echo "backed up $TABLE -> $BAK"

python eval_pipeline/aggregate_results.py \
  --out_root "$OUT_ROOT" \
  --scene_list "$OUT_ROOT/scene_list.txt" \
  --labels "$@"

BAK="$BAK" TABLE="$TABLE" python - "$@" <<'PY'
import csv, os, sys
bak, table = os.environ["BAK"], os.environ["TABLE"]
new = {r["setup"]: r for r in csv.DictReader(open(table))}
old = list(csv.DictReader(open(bak)))
fields = list(csv.DictReader(open(bak)).fieldnames)
seen, rows = set(), []
for r in old:                       # keep prior rows in their original order,
    rows.append(new.get(r["setup"], r))   # refreshed if just recomputed
    seen.add(r["setup"])
for lb in sys.argv[1:]:             # then append genuinely new labels
    if lb not in seen:
        if lb not in new:
            sys.exit(f"ERROR: {lb} produced no row -- aggregation failed")
        rows.append(new[lb])
with open(table, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
print(f"merged: {len(rows)} rows in {table}")
for r in rows:
    print(f"  {r['setup']:<20} n_scenes={r['n_scenes']}")
PY
