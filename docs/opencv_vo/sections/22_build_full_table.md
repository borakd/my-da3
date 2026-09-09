## `eval_pipeline/build_opencv_full_table.py`, lines 1-121: the full-4292 LaTeX table builder

This file is the last, purely presentational stage of the OpenCV-VO control-arm pipeline. By the time it runs, every arm has already been scored: `opencv_vo.py` wrote per-scene `camera/*.npz` poses (with depth symlinked from the paired CUT3R run, decision 0.4), and `opencv_vo_eval.py` ran `eval_depth_poses.py` per scene so that each `<out_root>/<label>/eval/<scene>/eval_depth_pose_metrics.csv` exists. This script reads the ALL/MEAN row of those CSVs for five fixed arms over the full 4292-scene DROID wrist harness (decision 0.3: "then 430, then 4292 once frozen"), takes the `nanmean` over scenes as `aggregate_results.py` does, ranks each metric column (best / 2nd / 3rd, ties share a rank), emits a standalone booktabs LaTeX document with three footnotes (harness convention, a one-paragraph description of the backbone including the accepted retro-fill caveat, and a colour legend), compiles it with `pdflatex` and rasterises it with Ghostscript. Its output is the table quoted in the design doc's "FULL HARNESS" section: `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/tables/opencv_backbone_full4292.{tex,pdf,png}`. Nothing here estimates geometry; every geometric convention (Sim(3) alignment, RMSE reduction, median depth scaling) was fixed upstream in `eval_depth_poses.py`, and this file only has to report it faithfully.

### Lines 1-5: shebang and the docstring's purpose/style paragraph

```
1: #!/usr/bin/env python
2: """Full-harness (all 4292 DROID test scenes) LaTeX table: CUT3R zero-shot / finetuned, with and without the
3: OpenCV-VO pose backbone. Format/style follows
4: maks_plots_augfull_lr1e5/eval/.../maks_windows100/tables/windows100_breakdown.tex (standalone, booktabs,
5: best/2nd/3rd cell colours per column, footnotes in a minipage).
```

- **What it does.** Declares the script as a `python` executable via `env` lookup (whatever `python` is first on `PATH`; only NumPy plus the sibling `aggregate_results.py` are needed) and opens the module docstring. Lines 2-5 state the scope (all 4292 DROID test scenes), the row set (CUT3R zero-shot and finetuned, each with and without the OpenCV pose backbone) and the visual template: a *standalone* document class, `booktabs` rules, per-column best/2nd/3rd cell colours, footnotes in a `minipage`. The reference is a pre-existing table from the augfull_lr1e5 windowed-eval study (it exists on disk at `.../cut3r_eval/maks_windows100/tables/windows100_breakdown.tex` and uses the same `standalone` class, the same three `\cellcolor` specs and the same "(ties share)" legend line), so this table looks like that one when placed side by side.
- **Alternatives considered.** Emitting Markdown/CSV only (what `aggregate_results.py` already does with its `averages_table` outputs); a `\begin{table}` fragment to be `\input` into a paper; matplotlib-rendered table images. None of these are recorded as design decisions; they are standard reporting options.
- **Why this choice.** Consistency with the existing windows100 breakdown table is the stated reason ("Format/style follows ..."). The 4292-scene scope is decision 0.3 and, per the honesty audit, audit item 7 ("all thresholds tuned on the 12 reported (test) scenes") "is addressed by reporting on all 4292 scenes with the configuration frozen" — this table is that report.

### Lines 6-12: the docstring's value convention, usage line and close

```
6: 
7: Values: per-scene ALL/MEAN row of eval_depth_pose_metrics.csv (eval_depth_poses.py default args: depth
8: median-scale-aligned, pose Sim(3)-aligned, RMSE-reduced), nanmean over scenes -- identical to
9: eval_pipeline/aggregate_results.py.
10: 
11: Usage: build_opencv_full_table.py [--out_root ...] [--scene_list ...] [--tex ...]
12: """
```

- **What it does.** Line 6 is a blank separator inside the docstring. Lines 7-9 pin the *unit of value* for every cell: the per-scene summary row (`camera_id == "ALL"`, `local_timestep == "MEAN"`) of `eval_depth_pose_metrics.csv`, produced by `eval_depth_poses.py` with default arguments — depth aligned by a per-frame median scale, poses aligned by one Sim(3) per scene, and the pose error reduced by RMSE over frames — and then a `nanmean` across scenes. Because a Sim(3) is fitted per scene, one global scale per scene is free while scale drift within a scene is penalised (design doc, "Dataset facts": "A global scale is free; scale DRIFT is not"). Line 10 is a blank separator; line 11 gives the CLI; line 12 closes the docstring.
- **Alternatives considered.** Median instead of mean over scenes (the 12-scene sweeps in the design doc report both, e.g. ATE mean 111.9 / median 97.7 for v0 final); per-frame pooling over all frames instead of per-scene averaging; SE(3) alignment rather than Sim(3).
- **Why this choice.** The docstring says it in one word: "identical" to `aggregate_results.py`, which is the harness convention used by every CUT3R row in `CONF_GATE_CAMPAIGN.md`. The design doc's full-harness section repeats it: "nanmean over scenes = aggregate_results.py convention, pose RMSE-reduced per scene". Sim(3) alignment is a harness property (decision 3.3 keeps map scale "as given" precisely because "Sim(3) scoring absorbs one global scale"). One precision on "identical": `aggregate_results.py` line 76 averages `vals[np.isfinite(vals)]`, which drops NaN *and* ±inf, whereas `np.nanmean` (line 45 here) drops NaN only; the two agree whenever every per-scene value is finite, which is the case for the published table.

### Lines 13-18: standard-library and NumPy imports

```
13: import argparse
14: import os
15: import subprocess
16: import sys
17: 
18: import numpy as np
```

- **What it does.** `argparse` for the three CLI flags; `os` for path joining and `makedirs`; `subprocess` to shell out to `pdflatex` and `gs`; `sys` to patch the import path; `numpy` only for `np.nanmean` and `np.isfinite`. Line 17 is the PEP 8 blank between stdlib and third-party imports.
- **Alternatives considered.** `pandas` for CSV reading and grouping; `pathlib` instead of `os.path`. Not design decisions.
- **Why this choice.** Minimal dependencies: the CSV parsing is delegated to `aggregate_results.read_scene_summary` (line 22), so nothing beyond NumPy is needed.

### Lines 19-22: import the harness's own summary reader

```
19: 
20: HERE = os.path.dirname(os.path.abspath(__file__))
21: sys.path.insert(0, HERE)
22: from aggregate_results import METRICS, read_scene_summary  # noqa: E402
```

- **What it does.** Line 19 is blank. Lines 20-21 put `eval_pipeline/` itself at the front of `sys.path` so the sibling module imports regardless of the caller's working directory (agent/Slurm invocations reset cwd). Line 22 imports `METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]` (`aggregate_results.py` line 17) and `read_scene_summary(csv_path)` (lines 20-47 there), which returns a `dict metric -> float` for the `ALL`/`MEAN` row (falling back to any single-camera `MEAN` row), `float("nan")` for a metric that will not parse, and `None` if the file is missing/unreadable or has no `MEAN` row. `# noqa: E402` silences the "import not at top" lint that the path patch necessitates.
- **Alternatives considered.** Reimplementing the CSV read here; making `eval_pipeline` a package with relative imports; parsing `summary.json` diagnostics from `opencv_vo.py` instead of the harness CSV.
- **Why this choice.** Reusing the reader is what makes the "identical to aggregate_results.py" claim on lines 8-9 hold at the row-selection and NaN-policy level — the same row is chosen and the same unparsable-to-NaN rule applies to every cell. Decision 0.4 fixes that the *harness* CSV scores only ATE / RPE-trans / RPE-rot for the OpenCV arm, with diagnostics (failed frames, median inliers) kept in a side CSV; this table deliberately reads the harness CSV and therefore shows no failure counts.

### Lines 23-24: the default output root

```
23: 
24: OUT_DEFAULT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
```

- **What it does.** Line 23 is blank. Line 24 hard-codes the MN5 scratch root under which every arm lives as `<OUT_DEFAULT>/<label>/eval/<scene>/…`. It also seeds the defaults for `--scene_list` and `--tex` (lines 51-52).
- **Alternatives considered.** Reading the root from `eval_pipeline/mn5_paths.sh` (where `SCENES_ROOT` comes from), or requiring `--out_root` explicitly.
- **Why this choice.** Matches the `$OUT` used in the design doc's "v0 FINAL configuration" invocation and the published table path (`.../cut3r_eval/tables/opencv_backbone_full4292.{tex,pdf,png}`). Note this is the *outputs* tree, not a checkpoint tree, so the CLAUDE.md checkpoint-relocation rule does not apply to it.

### Lines 25-28: the row list, part 1 — two model-only baselines and the first OpenCV pairing

```
25: ROWS = [
26:     ("CUT3R zero-shot (pretrained ckpt)", "cut3r_zeroshot"),
27:     ("Regular CUT3R (finetuned, augfull\\_lr1e5)", "augfull_lr1e5"),
28:     ("CUT3R zero-shot + OpenCV pose backbone", "vo_full_zs"),
```

- **What it does.** Each entry is `(LaTeX display name, run label)`; the label is the directory name under `out_root`. Line 26 is the pretrained `cut3r_512_dpt_4_64.pth` checkpoint run fresh on the harness (design doc, sweep 2: "cut3r_zeroshot = pretrained … run fresh"; full-harness inference Slurm 45527499-502). Line 27 is the finetuned `augfull_lr1e5` checkpoint — the campaign's "Regular CUT3R" row (the `\\_` is a Python-escaped `\_` so LaTeX does not treat the underscore as a subscript). Line 28 is the first OpenCV row: the classical backbone driven by the zero-shot checkpoint's per-frame focal, with the zero-shot depth symlinked in.
- **Alternatives considered.** Adding the champion `augfull_cg_fuse_g7` (fwd × bwd fusion) as a row; adding the strictly model-free `--focal 203` row (within noise of v0 final: 109.1/7.04/0.869 vs 111.9/6.93/0.871 on the 12 scenes); adding the trivial-trajectory floors (constant pose, constant velocity) from the metric-floor analysis.
- **Why this choice.** The row set is the paired with/without design of the design doc's FULL HARNESS section ("CUT3R with and without the OpenCV pose backbone"): each checkpoint with and without the backbone, so the backbone's effect is read off within a pair. The design doc records no explicit reason for omitting the champion; audit item 9 (causal finetuned row vs fwd × bwd champion) is a related open concern, still pending the user's ruling, not a recorded exclusion decision. The zero-shot pairing is what shows the backbone's clearest win: RPE-rot 1.713 → 1.301 deg at equal ATE / RPE-t (full-harness table).

### Lines 29-31: the row list, part 2 — the finetuned pairing and the privileged diagnostic

```
29:     ("CUT3R Finetuned + OpenCV pose backbone", "vo_full_ft"),
30:     ("CUT3R Finetuned + OpenCV pose backbone, GT intrinsics", "vo_full_ftgt"),
31: ]
```

- **What it does.** Line 29: the backbone driven by the finetuned checkpoint's inference-time per-frame focal (`--focal perframe:<preds_root>`, decision 0.1b as revised 2026-09-06), depth inherited from `augfull_lr1e5`. Line 30: the same run with `--focal gt`: one scalar calibrated focal per scene (`fx` read from `cam/000000.npz`, rescaled by the 320×192 cover factor `max(W/W1, H/H1)`; pp = image centre) replacing the per-frame predicted focal — in `opencv_vo.py` this is `model_K` lines 98-102 and 113, and line 279 replicates that single `K` for all `n` frames, so unlike the model-focal rows it is *not* per-frame and it is not the full calibrated K matrix. This is decision 0.1b's "calibrated-K copy kept as a one-off diagnostic"; audit item 6 lists this code path, and the footnote (line 100) flags the row as privileged. Line 31 closes the list. Row order matters later: line 94 splits `lines[:2]` / `lines[2:]`, so the two model-only rows sit above a `\midrule` and the three OpenCV rows below it.
- **Alternatives considered.** Per decision 0.1b the intrinsics options were calibrated K, one nominal dataset K, model-estimated focal (per-scene median, later per-frame), and self-calibration. Focal-sensitivity data on the 12 scenes: nominal 203 px 109.1 ATE, finetuned-model focal 111.9, calibrated 116.0, 150 px 109.8, 305 px 117.9, 406 px 119.2, OpenCV self-calibration 116.6.
- **Why this choice.** Decision 0.1b: the closed-loop rule makes the model's own focal the reported configuration ("model output is not privileged"), and the 2026-09-06 revision makes it per-frame so frame *t* only uses what CUT3R produced up to *t* (user constraint). The GT-intrinsics row is kept because the design doc says the focal source is "immaterial" on the 12 scenes and the full harness confirms it: "GT intrinsics are worth ~2 mm ATE" (0.103 vs 0.104 m). Its inclusion is a user-visible honesty device — it bounds how much the backbone could gain from perfect calibration.

### Lines 32-35: metric direction and the colour ladder

```
32: HIGHER_BETTER = {"a1"}
33: COLORS = [r"\cellcolor{red!30}\textbf{", r"\cellcolor{orange!30}", r"\cellcolor{yellow!40}"]
34: 
35: 
```

- **What it does.** Line 32: of the five `METRICS`, only `a1` (the δ<1.25 depth-inlier fraction) is higher-is-better; `absrel`, `ate`, `rpe_trans`, `rpe_rot` are all errors. Line 33: the three cell prefixes for rank 0/1/2. Rank 0 opens a `\textbf{` group that line 76 must close with `}`; ranks 1-2 are bare `\cellcolor` commands (colortbl syntax from `xcolor[table]`) and need no closing brace. Lines 34-35 are the two blank lines PEP 8 puts before a top-level `def`.
- **Alternatives considered.** Bold/underline/italic for 1st/2nd/3rd (the common paper convention); colouring only the best; grey-scale for print.
- **Why this choice.** Copies the reference `windows100_breakdown.tex` scheme (lines 4-5 of the docstring; the reference file's cells use exactly `red!30`+`\textbf`, `orange!30`, `yellow!40`) so the two tables share a legend; the same three colours are echoed in the footnote legend on line 101. No design-doc decision covers presentation colours.

### Lines 36-41: `load()` — open the per-scene CSVs, skip the missing ones

```
36: def load(out_root, label, scenes):
37:     vals = {m: [] for m in METRICS}; n = 0
38:     for s in scenes:
39:         r = read_scene_summary(os.path.join(out_root, label, "eval", s, "eval_depth_pose_metrics.csv"))
40:         if r is None:
41:             continue
```

- **What it does.** For one run label, walks the scene list in order, builds the harness path `<out_root>/<label>/eval/<scene>/eval_depth_pose_metrics.csv`, and asks `read_scene_summary` for its ALL/MEAN row. A `None` (file absent or unparsable) silently skips the scene — the count `n` (line 42) records how many scenes contributed, so partial rows are visible later as `(n=…)` (line 78). `vals` accumulates one Python list of floats per metric; the two statements on line 37 are joined with `;` (style choice, not semantics).
- **Alternatives considered.** Failing hard on a missing scene; inserting NaN placeholders so every row has exactly `len(scenes)` entries; reading the pre-aggregated `per_scene_<label>.csv` that `aggregate_results.py` writes.
- **Why this choice.** Skip-and-count is what lets the builder be run while rows are still being scored (the "(pending)" mechanism, line 69) and matches `aggregate_setup` in `aggregate_results.py` (lines 52-58 there), which also `continue`s past scenes whose CSV is missing or whose summary is `None`. On the published table all five rows reached the full 4292, so no `(n=…)` suffix appears.

### Lines 42-47: `load()` — accumulate and reduce with `nanmean`

```
42:         n += 1
43:         for m in METRICS:
44:             vals[m].append(r[m])
45:     return n, {m: float(np.nanmean(vals[m])) if vals[m] else float("nan") for m in METRICS}
46: 
47: 
```

- **What it does.** Increments the scene count only for readable CSVs, appends each of the five metrics (a metric that failed to parse arrives as `nan` from the reader), and returns `(n, {metric: mean})`. `np.nanmean` ignores per-scene NaNs; an empty list (no scene readable at all) yields `nan` explicitly rather than raising. If every entry is NaN, NumPy returns `nan` with a `RuntimeWarning` — not an error. Units are whatever `eval_depth_poses.py` wrote: AbsRel dimensionless, `a1` a fraction in [0,1], ATE and RPE-trans in metres (the design doc's full-harness table is headed "ATE (m)", "RPE-t (m)"), RPE-rot in degrees. Lines 46-47 are blank.
- **Alternatives considered.** Plain `np.mean` (a single NaN scene would poison the column); median over scenes; weighting scenes by frame count (sequence length ranges 57 to ~1700 frames, so a frame-weighted mean would be dominated by the long scenes).
- **Why this choice.** `nanmean` over scenes is, again, the `aggregate_results.py` convention (docstring lines 7-9; see the ±inf caveat under lines 6-12) and the convention every CUT3R number in the campaign was produced with, so the OpenCV rows are directly comparable to the "Regular CUT3R" numbers elsewhere in the repo.

### Lines 48-53: `main()` — CLI flags

```
48: def main():
49:     ap = argparse.ArgumentParser()
50:     ap.add_argument("--out_root", default=OUT_DEFAULT)
51:     ap.add_argument("--scene_list", default=os.path.join(OUT_DEFAULT, "scene_list.txt"))
52:     ap.add_argument("--tex", default=os.path.join(OUT_DEFAULT, "tables", "opencv_backbone_full4292.tex"))
53:     args = ap.parse_args()
```

- **What it does.** Three optional flags. `--out_root` is the tree holding `<label>/eval/`; `--scene_list` defaults to the harness's `scene_list.txt` (4292 lines on disk, one scene id per line); `--tex` is the output `.tex` path whose directory also receives the `.pdf`, `.png` and `.build.log`. Note the defaults are computed from `OUT_DEFAULT`, not from the value of `--out_root`, so passing a different `--out_root` alone still reads the default scene list and writes to the default table directory.
- **Alternatives considered.** A `--labels`/`--rows` flag to choose arms at run time (as `pose_sim3_both.py --labels` does); a `--subset` flag for the 430 list.
- **Why this choice.** The row set is intentionally frozen in `ROWS` because the table's meaning depends on the paired layout and the `lines[:2]` split (line 94); the smoke-scale table is a different builder (`build_opencv_vo_table.py`, referenced in the design doc's inference-time-focal section). Decision 0.3 sets the scope to the full 4292 once the configuration is frozen (frozen 2026-09-05).

### Lines 54-57: read the scene list and load every row

```
54:     scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
55: 
56:     data = [(name, lbl) + load(args.out_root, lbl, scenes) for name, lbl in ROWS]
57:     present = [d for d in data if d[2] > 0]
```

- **What it does.** Line 54 reads scene ids, stripping whitespace and dropping blank lines. Line 56 builds one 4-tuple per row: `(display name, label, n, {metric: mean})` — tuple concatenation of the `(name, lbl)` pair with `load`'s `(n, dict)` return. Line 57 keeps the rows that had at least one readable scene; only those take part in ranking, so a still-pending row cannot occupy a colour slot. Line 55 is blank.
- **Alternatives considered.** Ranking all rows and letting NaN sort last; excluding rows with `n < len(scenes)` from ranking (stricter).
- **Why this choice.** Rows with partial coverage are still ranked (their `(n=…)` tag on line 78 discloses it); only rows with nothing at all are excluded. This is the pragmatic middle ground used while the full-harness runs (Slurm 45527508 / 45529336) were landing.

### Lines 58-62: rank per column on the displayed 3-decimal value

```
58:     # rank per column (ties share a rank), 3-decimal display values are what get compared, as in the reference
59:     disp = {(d[1], m): round(d[3][m], 3) for d in present for m in METRICS}
60:     rank = {}
61:     for m in METRICS:
62:         vs = sorted({disp[(d[1], m)] for d in present if np.isfinite(disp[(d[1], m)])}, reverse=(m in HIGHER_BETTER))
```

- **What it does.** `disp` maps `(label, metric)` to the value rounded to three decimals — the *same* precision the cell will print with (`:.3f`, line 73), so what is compared is what the reader sees. Line 62 builds, per metric, the sorted **set** of distinct finite display values: ascending for errors, descending for `a1`. Because it is a set, two rows with equal display values collapse to one entry, and (line 65) both receive that entry's index — *dense* ranking, so after a tie the next distinct value is the next rank, not rank+2. Rounding first is what makes ties happen at all: in the published table the finetuned RPE-trans values 0.0079 m and 0.0084 m (design doc: "ties on RPE (… 0.0084 vs 0.0079)") both display as `0.008` and are ranked equal-best.
- **Alternatives considered.** Ranking on the unrounded float (would break every visual tie and colour cells that read identically differently); competition ranking (1, 1, 3); ranking with a tolerance.
- **Why this choice.** The comment cites the reference table's convention ("as in the reference"). It is also the honest choice for this harness: the metric-floor analysis shows RPE-trans is "SATURATED at the no-motion floor (7.9 mm = the RMS step itself) for every arm except the champion (7.2)", so a sub-millimetre RPE-trans difference is not a real ordering and should not be coloured as one.

### Lines 63-66: assign each row its column rank

```
63:         for d in present:
64:             v = disp[(d[1], m)]
65:             rank[(d[1], m)] = vs.index(v) if np.isfinite(v) and v in vs else 99
66:     lines = []
```

- **What it does.** For each present row, `rank[(label, metric)]` is the position of its display value in the distinct sorted list — 0 = best. A NaN value gets the sentinel `99`, which is `>= 3` and therefore never coloured (line 75). `rank` is keyed by label, so two `ROWS` entries sharing a label would overwrite each other (not the case here). Line 66 starts the list of LaTeX row strings.
- **Alternatives considered.** `scipy.stats.rankdata(method="dense")`; storing ranks on the tuple.
- **Why this choice.** Ten lines of explicit code with no extra dependency; the `99` sentinel makes "NaN is never highlighted" a one-line invariant.

### Lines 67-71: emit a row — the "(pending)" placeholder

```
67:     for name, lbl, n, v in data:
68:         if n == 0:
69:             lines.append(f"{name} & \\multicolumn{{5}}{{c}}{{(pending)}} \\\\")
70:             continue
71:         cells = []
```

- **What it does.** Iterates over **all** rows in `ROWS` order (not just `present`), so the table layout never changes as rows arrive. A row with no readable scene becomes `name & \multicolumn{5}{c}{(pending)} \\` — one centred cell spanning the five metric columns. Doubled braces are f-string escapes producing single braces; `\\\\` in the Python source produces the LaTeX row terminator `\\`. Line 71 starts the five metric cells for a real row.
- **Alternatives considered.** Omitting unfinished rows; printing `--` per cell; failing the build.
- **Why this choice.** Same policy as the campaign's `summary_table_final` commit (git log: "ttt3r/raymap3r as PENDING rows"): the table can be circulated before every arm is scored and its shape is stable. In the delivered 4292 table no row is pending.

### Lines 72-77: emit a row — format and colour each metric cell

```
72:         for m in METRICS:
73:             txt = f"{v[m]:.3f}"
74:             rk = rank[(lbl, m)]
75:             if rk < 3:
76:                 txt = COLORS[rk] + txt + ("}" if rk == 0 else "")
77:             cells.append(txt)
```

- **What it does.** Each value prints with three decimals (`nan` prints as `nan`). If the rank is 0, 1 or 2 the cell is wrapped: rank 0 → `\cellcolor{red!30}\textbf{0.076}` (closing brace appended), rank 1 → `\cellcolor{orange!30}0.103`, rank 2 → `\cellcolor{yellow!40}0.104`. Rank 3+ (including the 99 sentinel) prints plain. Because ranking is dense over distinct values, a column with only two distinct values — the depth columns, where the three finetuned rows are identical by construction and the two zero-shot rows are identical by construction — shows red on three cells, orange on two, and no yellow at all, exactly as in the rendered `.tex`.
- **Alternatives considered.** Four significant figures (would separate 0.0079 from 0.0084); millimetres for ATE/RPE-t as in the 12-scene sweeps; scientific notation.
- **Why this choice.** Three decimals in metres is the reference table's convention and matches the design doc's full-harness table, which quotes 0.120 / 0.012 / 1.713. The finer resolution survives only in the stdout summary at line 117, which prints four decimals.

### Lines 78-81: emit a row — the `(n=…)` disclosure, and the footnote's scene count

```
78:         nn = f" (n={n})" if n != len(scenes) else ""
79:         lines.append(f"{name}{nn} & " + " & ".join(cells) + r" \\")
80:     n_full = max((d[2] for d in present), default=0)
81: 
```

- **What it does.** If a row covers fewer scenes than the list, its name is suffixed with ` (n=<count>)`; a complete row gets no suffix. Line 79 joins name and cells with `&` and terminates the LaTeX row (raw string, so ` \\` is literal). Line 80 sets the number quoted in the first footnote ("All … DROID wrist test scenes") to the *largest* scene count among present rows. Line 81 is blank.
- **Alternatives considered.** Always printing `n`; using `len(scenes)` for the footnote; refusing to build unless all rows are complete.
- **Why this choice.** Minimal-noise disclosure: the suffix appears only when there is something to disclose. Caveat: the footnote uses the max, so if one row were partial the footnote would still say "All 4292" while that row's `(n=…)` tag would contradict it — acceptable because in the delivered table all rows are complete (n = 4292 = `len(scenes)`).

### Lines 82-87: LaTeX preamble

```
82:     tex = r"""\documentclass[11pt,border=8pt]{standalone}
83: \usepackage{booktabs}
84: \usepackage[table]{xcolor}
85: \newsavebox{\tblbox}
86: \begin{document}
87: \sbox{\tblbox}{%
```

- **What it does.** Opens a raw triple-quoted string (so every backslash is literal). `standalone` with an 8 pt border crops the page to the content, which is what makes the Ghostscript PNG on line 111 a tight image. `booktabs` supplies `\toprule/\midrule/\bottomrule`; `xcolor` with the `table` option loads `colortbl` and enables `\cellcolor`. Lines 85-87 define a save box and start filling it with the tabular: the box's width `\wd\tblbox` is later used (line 97) to make the footnote minipage exactly as wide as the table so footnotes wrap under it rather than across the page.
- **Alternatives considered.** `article` class with a `table` float; `threeparttable` for table notes; `tabularx`.
- **Why this choice.** Verbatim the reference table's scaffold (docstring lines 4-5; `windows100_breakdown.tex` line 1 is the same `\documentclass[11pt,border=8pt]{standalone}`). The `standalone` class is also what forces the TeX Live tree choice on lines 108-109: the system `/usr/bin/pdflatex` has no `standalone.cls`.

### Lines 88-93: column spec and two-level header

```
88: \begin{tabular}{@{}l r r | r r r@{}}
89: \toprule
90:  & \multicolumn{2}{c|}{Depth} & \multicolumn{3}{c}{Pose metrics} \\
91: Method & AbsRel$\downarrow$ & $\delta{<}1.25\uparrow$
92:  & ATE$\downarrow$ & RPE\textsubscript{trans}$\downarrow$ & RPE\textsubscript{rot}$\downarrow$ \\
93: \midrule
```

- **What it does.** One left-aligned name column, two right-aligned depth columns, a vertical rule, three right-aligned pose columns; `@{}` strips the outer padding. The column order is exactly `METRICS` order (`absrel, a1, ate, rpe_trans, rpe_rot`), which is what makes the `" & ".join(cells)` on line 79 line up. Row 90 is the group header; rows 91-92 name the metrics with direction arrows — `↓` on the four errors, `↑` on `δ<1.25`, mirroring `HIGHER_BETTER` (line 32). `\textsubscript` is available in modern LaTeX kernels without `fixltx2e`.
- **Alternatives considered.** Splitting RPE into separate translation/rotation super-columns; showing failed-frame % and reboots as extra columns (they exist per scene in `summary.json`); dropping depth columns for the OpenCV rows.
- **Why this choice.** Decision 0.4: the harness scores "ATE / RPE-trans / RPE-rot only" and the diagnostics go to "a side CSV (not the harness CSV)", so they are absent here by design. The depth columns are kept — and clearly labelled as the paired model's in the footnote — because decision 0.4 says AbsRel/δ₁ "are INHERITED from augfull_lr1e5 depth via symlink; mark as inherited in tables".

### Line 94: splice the rows around the group `\midrule`

```
94: """ + "\n".join(lines[:2]) + "\n\\midrule\n" + "\n".join(lines[2:]) + r"""
```

- **What it does.** Closes the raw preamble string, inserts the first two rows (the two model-only baselines), a `\midrule`, then the remaining three rows (the OpenCV arms), then re-opens a raw string for the tail. The split index `2` is hard-coded to `ROWS`' layout; reordering `ROWS` without changing this line would move the rule.
- **Alternatives considered.** A per-row "group" field; no separator; one rule between every pair.
- **Why this choice.** The visual grouping *with / without backbone* is the reading the design doc draws from the table ("CUT3R with and without the OpenCV pose backbone"); a single rule communicates it without extra structure.

### Lines 95-99: close the table, open the note block, footnote 1 (harness convention)

```
95: \bottomrule
96: \end{tabular}}%
97: \begin{minipage}{\wd\tblbox}
98: \usebox{\tblbox}\par\vspace{2pt}
99: {\footnotesize All """ + str(n_full) + r""" DROID wrist test scenes. Values = per-scene averages (eval\_depth\_poses.py default args), averaged over scenes. Depth = mean over frames (per-frame median-scaled); pose = Sim3-aligned RMSE over frames; RPE\textsubscript{rot} in degrees.\par}
```

- **What it does.** Ends the tabular and the save box (the trailing `%` suppresses a stray space). Line 97 opens a minipage as wide as the table; line 98 places the boxed table and a 2 pt gap. Line 99 is the first footnote: it splices in `n_full` (4292 on the published table) and restates, for the reader, the value convention from docstring lines 7-9 — per-scene averages from `eval_depth_poses.py` defaults, then averaged over scenes; depth is a per-frame median-scaled mean over frames; pose is a Sim(3)-aligned RMSE over frames; RPE-rot is in degrees. `\_` escapes the underscore in the script name.
- **Alternatives considered.** Putting the convention in the caption of an enclosing float; omitting it and pointing to the README.
- **Why this choice.** The table is meant to travel as a single PNG; the convention has to be *in* the image. The wording is the `aggregate_results.py` convention (decision 0.4 and docstring), and "Sim3-aligned" is what makes ATE-in-metres comparable across arms of arbitrary map scale (decision 3.3).

### Line 100: footnote 2 — what the backbone is, what it may see, and the accepted retro-fill caveat

```
100: {\footnotesize OpenCV pose backbone: Shi--Tomasi + LK tracks, parallax-gated essential-matrix bootstrap, PnP against a triangulated map, keyframes every 5\,px of parallax, outlier culling, re-bootstrap after 3 failed frames with scale hand-off, static-track lever on. Inputs at frame $t$: the images and CUT3R's predicted focal for frame $t$ (closed loop, inference-time); depth columns are the paired CUT3R model's own depth. The last row replaces the predicted focal with the calibrated intrinsics (privileged, diagnostic). One non-causal step is retained: frames before each (re-)bootstrap succeeds are posed retroactively by PnP against the map built at the bootstrap frame (typically 2--16 frames per scene, plus a few after each re-bootstrap); all other frames use only past information.\par}
```

- **What it does.** A one-paragraph, decision-by-decision summary of the frozen v0 pipeline so the table is self-describing. Sentence 1 enumerates the design: Shi-Tomasi corners + LK tracks (decisions 1.3, 1.5), parallax-gated essential-matrix bootstrap (2.9, 2.10), PnP against a triangulated map (0.1, 0.2, 2.1, 2.4), keyframes every 5 px of median parallax (2.12; `5\,px` is a thin space), outlier culling (2.14), re-bootstrap after 3 consecutive failed frames with scale hand-off (3.2b, 2.15), static-track lever on (1.2). Sentence 2 states the *information boundary*: at frame *t* the backbone sees the images and CUT3R's predicted focal for frame *t* only (0.1b as revised: inference-time, closed loop), and the depth columns are the paired model's own (0.4; audit item 10). Sentence 3 marks the GT-intrinsics row as privileged. Sentence 4 is the retro-fill clause required by the user's ruling on audit item 1: frames before a (re-)bootstrap succeeds are posed *retroactively* by PnP against the map built at the bootstrap frame — a non-causal step — with the size estimate "typically 2--16 frames per scene, plus a few after each re-bootstrap" (this figure appears only here, not in the design doc; see the limitations); every other frame uses only past information.
- **Alternatives considered.** For the causality question specifically: `--no_retro_fill` (strictly causal; pre-bootstrap frames keep the held anchor pose and stay flagged failed). Measured on the 12 scenes: with retro-fill 110.4 / 7.5 / 0.858, without 110.6 / 7.7 / 0.881 (ATE mm / RPE-t mm / RPE-r deg) — "+0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r. It does not change any ordering." For the summary sentence: a bare "OpenCV VO" label with a pointer to the design doc.
- **Why this choice.** Every parameter named is the frozen v0 value from the design doc's "Frozen v0 parameters" table (keyframe/bootstrap trigger 5 px; lever ON since 2026-09-05 by user decision; reboot after 3 fails from sweep 2: failed frames 18 → 14 %, RPE-r 1.11 → 1.01; scale hand-off from sweep 3: ATE 116.8 → 111.9, median 117 → 98; culling from sweep 1: PnP failures 26 % → 5.6 %). The retro-fill sentence exists because the honesty audit's item 1 was "ACCEPTED 2026-09-07 with a footnote clause in the full-4292 table (eval_pipeline/build_opencv_full_table.py)" — this line *is* that clause. The per-frame-focal wording reflects the user's 2026-09-06 constraint that retroactive per-scene medians "are not an inference-time method".

### Lines 101-104: footnote 3 (colour legend) and document close

```
101: {\footnotesize Highlighting: \colorbox{red!30}{\textbf{best}}, \colorbox{orange!30}{2nd}, \colorbox{yellow!40}{3rd} per column (ties share)\par}
102: \end{minipage}
103: \end{document}
104: """
```

- **What it does.** The legend renders the three swatches with the same `xcolor` specs as `COLORS` (line 33) so the legend and the cells are guaranteed to match, and states the tie rule ("ties share") that lines 59-65 implement. Lines 102-104 close the minipage, the document, and the Python raw string.
- **Alternatives considered.** Omitting the legend (relying on convention); a caption-level legend.
- **Why this choice.** Same legend line as the reference table (`windows100_breakdown.tex` line 25 is character-for-character this string); the explicit "(ties share)" is needed because dense tie-sharing produces visibly "missing" colours that would otherwise look like a bug: no yellow in the depth columns (red ×3, orange ×2), and only two colours in the RPE-trans column, where the three finetuned rows all display `0.008` and share red and the two zero-shot rows share `0.012` and orange, so no yellow appears (rendered `.tex`: rpe_trans cells are red/red/red/orange/orange). In the ATE column, by contrast, all three colours appear (red 0.076, orange 0.103, yellow 0.104) and the two `0.120` cells are plain simply because they are dense rank 3.

### Lines 105-107: write the `.tex` and derive the build names

```
105:     os.makedirs(os.path.dirname(args.tex), exist_ok=True)
106:     open(args.tex, "w").write(tex)
107:     d, base = os.path.dirname(args.tex), os.path.splitext(os.path.basename(args.tex))[0]
```

- **What it does.** Creates the `tables/` directory if needed, writes the LaTeX source (overwriting any previous build), and splits the path into directory `d` and stem `base` (`opencv_backbone_full4292`) for the compile command. The file handle is not explicitly closed; CPython closes it when the temporary is collected, and the subsequent `pdflatex` reads the file only after this statement completes.
- **Alternatives considered.** `with open(...) as f:` (which is what `build_lr1e5_table.py` lines 201-202 do); writing to a temp file and renaming atomically.
- **Why this choice.** Script-grade simplicity; there is no concurrency on this path.

### Lines 108-111: the MN5 `pdflatex` + Ghostscript command

```
108:     cmd = ("export TEXMFROOT=/apps/GPP/LATEX/20240430 && export TEXMFCNF=$TEXMFROOT:$TEXMFROOT/texmf-dist/web2c && "
109:            "export PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:$PATH; "
110:            f"cd {d} && pdflatex -interaction=nonstopmode {base}.tex >{base}.build.log 2>&1 && "
111:            f"gs -sDEVICE=png16m -r300 -o {base}.png -dBATCH -dNOPAUSE {base}.pdf >/dev/null")
```

- **What it does.** Builds one `bash` command string. Lines 108-109 point TeX at the site's TeX Live 2024 tree under `/apps/GPP/LATEX/20240430` by setting `TEXMFROOT`, `TEXMFCNF` and prepending that tree's binary directory to `PATH` (so its `pdflatex` shadows the system `/usr/bin/pdflatex`, which is on the default `PATH` but lacks `standalone.cls`, needed by line 82). Line 110 `cd`s into the table directory so the `.aux/.log/.pdf` land beside the `.tex`, runs `pdflatex` non-interactively (errors do not block on a prompt) and captures all its output in `<base>.build.log`. Line 111 rasterises the PDF with Ghostscript (the system `/usr/bin/gs`; MN5 has no Ghostscript module, per the sibling builder's comment): `png16m` = 24-bit colour PNG, `-r300` = 300 dpi, `-o` = output file plus implicit batch/no-pause, and the extra `-dBATCH -dNOPAUSE` are redundant with `-o` but harmless. The `&&` chain stops at the first failure.
- **Alternatives considered.** `module load latex/20240430` (the way `build_gru_overfit_table.py` reaches the same tree, per the comment in `build_lr1e5_table.py`); `latexmk`; `pdftoppm` for rasterisation; rendering the table with matplotlib and skipping TeX entirely.
- **Why this choice.** The system `/usr/bin/pdflatex` lacks `standalone.cls` (needed by line 82); the full TeX Live 2024 tree lives under `/apps/GPP/LATEX/20240430` and would normally be reached via `module load latex/20240430`, but lmod is not initialised in non-interactive login-node shells, so the script sets `TEXMFROOT`/`TEXMFCNF`/`PATH` itself — the same workaround as `eval_pipeline/build_lr1e5_table.py` lines 207-218, which documents the reason in its comment ("/usr/bin/pdflatex has no standalone.cls; the TeXLive tree under /apps/GPP/LATEX does … lmod is not initialised on the login nodes (/etc/profile.d/lmod.sh is absent)") and which produced `lr1e5_3arm`/`lr1e5_5arm` with the same `pdflatex` + `gs` pair. The only differences are that this script also exports `TEXMFROOT`/`TEXMFCNF` and keeps the `pdflatex` output in `<base>.build.log` instead of discarding it to `/dev/null`. No design-doc decision covers it.

### Lines 112-115: run the compile and fail loudly

```
112:     r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
113:     if r.returncode != 0:
114:         raise SystemExit(f"compile failed: see {d}/{base}.build.log")
115:     print("OK:", args.tex, "+ .pdf + .png")
```

- **What it does.** Executes the command through `bash -c` (needed for the `export`/`&&` syntax), capturing stdout/stderr as text so the terminal stays quiet. A non-zero exit — from `pdflatex` (a LaTeX error under `nonstopmode` still yields exit 1) or from `gs` — aborts with a one-line pointer to the build log; success prints the three output paths.
- **Alternatives considered.** `check=True` on `subprocess.run` (would raise `CalledProcessError` with a Python traceback instead of the tidy pointer); ignoring the return code and letting a missing PNG be discovered later.
- **Why this choice.** The `.build.log` pointer is the useful failure message on a cluster where the login node kills any process at 300 s of CPU (CLAUDE.md); a compile that dies from that cap would surface here rather than as a silently stale PNG.

### Lines 116-118: stdout summary at four decimals

```
116:     for name, lbl, n, v in data:
117:         print(f"  {name:55s} n={n:4d} " + " ".join(f"{m}={v[m]:.4f}" for m in METRICS))
118: 
```

- **What it does.** After a successful build, prints one line per row (name padded to 55 characters, `n` to 4 digits) with all five metrics at **four** decimals — one more than the table. This is the only place in this builder where the 0.0079 vs 0.0084 m RPE-trans distinction the table rounds away is printed (the design doc quotes those values without saying where they were read from; its metric-floor table, computed on 144 sampled scenes, gives 0.0081 / 0.0083 for the same pair, so the two four-decimal sources are not the same computation). Line 118 is blank.
- **Alternatives considered.** Writing a CSV alongside the `.tex`; logging via `logging`.
- **Why this choice.** A cheap audit trail: the 3-decimal table and the 4-decimal stdout are produced from the same `data`, so a reviewer can confirm no tie in the table hides a real ordering — and, per the metric-floor analysis, that the RPE-trans "ordering" it does hide is below the 7.9 mm no-motion floor anyway.

### Lines 119-121: entry point

```
119: 
120: if __name__ == "__main__":
121:     main()
```

- **What it does.** Line 119 is the second PEP 8 blank line; lines 120-121 run `main()` only when the file is executed as a script, so `ROWS`, `load`, and the constants can be imported by other tooling without triggering a compile.
- **Alternatives considered.** None meaningful; standard Python idiom.
- **Why this choice.** Standard idiom.

### Known limitations / honesty notes for this block

- **Retro-fill is non-causal and is retained** (honesty-audit item 1, accepted by the user 2026-09-07 with the footnote clause on line 100). Ablation on the 12 smoke scenes: with vs without retro-fill 110.4/7.5/0.858 vs 110.6/7.7/0.881 (ATE mm / RPE-t mm / RPE-r deg); no ordering changes. The clause's "typically 2--16 frames per scene" is stated only in the footnote itself (line 100); the design doc records no per-scene count for retro-filled frames, and audit item 12 notes the failed-frame count excludes them, so the figure is not audit-traceable from the doc.
- **Depth columns are inherited, not produced by the backbone** (decision 0.4; audit item 10). The three finetuned rows and the two zero-shot rows are identical on AbsRel / δ<1.25 by construction (symlinked depth), and the colouring on those columns rewards the OpenCV rows for depth they did not compute. The footnote on line 100 discloses this; the cells themselves do not.
- **Three-decimal rounding creates the ties.** RPE-trans 0.0079 m vs 0.0084 m both print as `0.008` and share "best". Per the metric-floor analysis this is defensible — RPE-trans is saturated at the 7.9 mm no-motion floor for every arm except the champion — but it means the table cannot show a 37 % ATE loss (0.104 vs 0.076) next to *any* RPE-trans separation. ATE and RPE-rot are the informative columns.
- **No failure diagnostics are shown.** Failed-frame %, re-bootstrap counts and median inliers live in `<label>/preds/<scene>/{diag.csv,summary.json}` (decision 0.4), not in the harness CSV this builder reads; and audit item 12 notes the failed-frame count excludes retro-filled frames anyway.
- **The GT-intrinsics row is privileged** (footnote sentence 3; decision 0.1b's "one-off diagnostic"; audit item 6). It is one scalar calibrated focal per scene from `cam/000000.npz` with pp at the image centre, not per-frame calibrated K. Its purpose is to bound the calibration headroom: ~2 mm ATE on the full harness.
- **The champion (`augfull_cg_fuse_g7`) is absent**, so this table does not show the campaign's best pose numbers. The row set is the paired with/without design of the FULL HARNESS section; the design doc records no explicit reason for omitting the champion. Audit item 9 (causal finetuned row vs fwd × bwd champion) is a related open concern, still pending the user's ruling, not a recorded exclusion decision.
- **"Identical to aggregate_results.py" holds on finite data only.** `aggregate_results.py` line 76 averages `vals[np.isfinite(vals)]` (drops NaN and ±inf); line 45 here uses `np.nanmean` (drops NaN only). No ±inf appears in the published rows, so the numbers agree, but the two reducers are not the same function.
- **Hard-coded layout couplings.** The `lines[:2]` split on line 94 assumes `ROWS` starts with exactly two model-only rows; `rank`/`disp` are keyed by label, so duplicate labels would collide; `--scene_list`/`--tex` defaults derive from `OUT_DEFAULT`, not from `--out_root`; the footnote's scene count is the *max* over rows, so a partial row would be disclosed only by its `(n=…)` suffix. None of these bite on the delivered table (all five rows at n = 4292).
- **Zero-shot got no configuration search** (audit item 11) and **all thresholds were tuned on 12 of the reported test scenes** (item 7, addressed by freezing before the 4292 run) — both inherited by every OpenCV row here.
