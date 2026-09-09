## `eval_pipeline/build_opencv_vo_table.py`, lines 1-125: the 12-scene smoke-set LaTeX table

This script is the last, presentation-only step of the OpenCV VO control-arm pipeline on the 12-scene smoke set (decision 0.3). Upstream, `opencv_vo.py` writes per-scene poses (`camera/%06d.npz`) plus a diagnostics side file (`preds/<scene>/summary.json`, decision 0.4), and `pose_sim3_both.py` scores each label into `<OUT>/<label>/sim3_pose_both.csv` (Sim(3)-aligned ATE / RPE, per-scene). This builder reads those two artefacts for seven labels (three CUT3R model arms, four OpenCV arms that differ only in their focal source, decision 0.1b), reduces them to one number per column per label, bolds the column minima, emits a `standalone`/`booktabs`/`newtx` LaTeX file, compiles it with the MareNostrum 5 LaTeX tree, and rasterises the PDF to PNG with Ghostscript. It contains no geometry and makes no decisions; its job is to reproduce, from disk, the "Inference-time (causal) focal (2026-09-06; Slurm 45485470)" table of the design doc. The full-4292 counterpart is `build_opencv_full_table.py` (different scorer and aggregation, see the honesty notes).

### Lines 1-3: shebang and docstring opening

```
1: #!/usr/bin/env python
2: """LaTeX table (+ PDF + PNG) of the OpenCV VO control arm vs the CUT3R arms on the 12-scene smoke set.
3: 
```

**What it does.** Declares the file as a directly runnable Python script (resolved through `env`, so whichever `python` is on `PATH`) and opens the module docstring with a one-line summary: the output is a `.tex` plus its `.pdf` and `.png` renderings. Line 3 is a blank line inside the docstring.

**Alternatives considered.** No design-doc decision covers this; the standard alternatives are a `python3`-pinned shebang or no shebang (invoked as `python build_opencv_vo_table.py`). The docstring's own `Usage:` line (line 9) shows the direct script invocation, which is what the shebang enables.

**Why this choice.** Convention shared with the sibling table builders in `eval_pipeline/` (the docstring at line 7 names `build_gru_vs_prevpred_teal.py` as the recipe source). Not a benchmarked choice.

### Lines 4-8: docstring body — inputs, omitted columns, recipe

```
4: Reads <OUT>/<label>/sim3_pose_both.csv (pose_sim3_both.py output; Sim(3)-aligned, per-scene means) and
5: <OUT>/<label>/preds/<scene>/summary.json (OpenCV diagnostics). Depth columns are not shown: the OpenCV arm
6: predicts poses only (AbsRel / delta1 would be inherited from augfull_lr1e5). Same standalone/booktabs/newtx
7: recipe as build_gru_vs_prevpred_teal.py; compiled with the MN5 latex module + /usr/bin/gs.
8: 
```

**What it does.** Documents the two input artefacts and the one deliberate omission. `sim3_pose_both.csv` is written by `pose_sim3_both.py`, which Umeyama-aligns (with scale) the predicted c2w trajectory to GT, computes per-frame ATE (metres) and consecutive-pair RPE (translation in metres, rotation in degrees), and aggregates each scene under both a `_mean` (nanmean over frames) and a `_rmse` reduction. `summary.json` is written by `opencv_vo.py` per scene and carries `frames`, `failed_frames`, `reboots`, `keyframes`, `median_inliers`, `map_points`, `seconds`, and the focal statistics. Depth columns (AbsRel, δ1) are declared absent because the OpenCV arm produces no depth. Line 8 is a blank line inside the docstring.

**Alternatives considered.** (a) Show depth columns inherited via the `--depth_link` symlink, as the full-4292 table does. (b) Use the harness's own `eval_depth_poses.py` per-scene CSVs (RMSE-reduced) instead of `pose_sim3_both.py` mean-reduced values. (c) Report only the harness CSV and skip diagnostics.

**Why this choice.** Decision 0.4: "harness scores ATE / RPE-trans / RPE-rot only (AbsRel, d1 are INHERITED from augfull_lr1e5 depth via symlink; mark as inherited in tables). Plus diagnostics in a side CSV (not the harness CSV)". For the smoke table the inherited columns are simply dropped rather than marked, because every OpenCV row would repeat the paired model's depth numbers verbatim (honesty-audit item 10). `pose_sim3_both.py` is the scorer named in the v0 FINAL recipe ("Then: python eval_pipeline/pose_sim3_both.py ...") and the scorer every sweep table in the design doc (sweeps 1-3, focal checks) was produced with, so the smoke table stays on the same footing as those sweeps.

### Lines 9-10: usage line and docstring close

```
9: Usage: build_opencv_vo_table.py [--out_root ...] [--scene_list ...] [--tex ...]
10: """
```

**What it does.** Names the three optional CLI flags (all have defaults, lines 51-53) and closes the docstring.

**Alternatives considered.** A `--labels` flag to choose rows at run time (as `pose_sim3_both.py` has); a `--no_compile` flag to emit only the `.tex`.

**Why this choice.** The row set is the design-doc's frozen comparison, not a user-facing parameter; hard-coding it (lines 23-31) keeps the table reproducible from the script alone. Not benchmarked; a plumbing choice.

### Lines 11-16: standard-library imports

```
11: import argparse
12: import csv
13: import glob
14: import json
15: import os
16: import subprocess
```

**What it does.** `argparse` for the three flags; `csv.DictReader` to parse the scorer CSV by column name; `glob` to enumerate `preds/*/summary.json`; `json` to parse them; `os` for path joins and `makedirs`; `subprocess` to run the `pdflatex` + `gs` shell pipeline.

**Alternatives considered.** `pandas.read_csv` + `DataFrame.to_latex`; `pathlib` instead of `os.path`/`glob`.

**Why this choice.** Zero non-numpy dependencies keeps the script light; the design doc notes the login node has a 300 s per-process CPU cap (OPENCV_VO_DESIGN.md line 467), and the LaTeX tree is driven directly via environment variables (lines 112-113) so no module system is needed. Not a design-doc decision.

### Lines 17-20: numpy and the output root

```
17: 
18: import numpy as np
19: 
20: OUT_DEFAULT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
```

**What it does.** Imports numpy for `mean`/`median` over per-scene arrays (line 39). `OUT_DEFAULT` is the eval output root under which every label directory lives (`<OUT>/<label>/sim3_pose_both.csv`, `<OUT>/<label>/preds/<scene>/...`, and `<OUT>/tables/`). Lines 17 and 19 are separators.

**Alternatives considered.** Read the root from `eval_pipeline/mn5_paths.sh` (which defines `SCENES_ROOT`), or require `--out_root` with no default.

**Why this choice.** This is the same `$OUT` the v0 FINAL recipe in the design doc uses (`--out_root $OUT`, `--preds_root $OUT/augfull_lr1e5/preds`) and the root the full-4292 table is also written under (`.../outputs/cut3r_eval/tables/opencv_backbone_full4292.*`). Hard-coding matches the sibling builders; it is overridable via `--out_root` (line 51).

### Lines 21-23: the row list, opened

```
21: 
22: # (display name, label, group). Groups are separated by \midrule.
23: ROWS = [
```

**What it does.** Begins `ROWS`, a list of `(display name, label, group)` triples. `label` is the on-disk directory name under `OUT`; `group` is a string compared between consecutive rows to decide where a `\midrule` separator is inserted (lines 62-64). Line 21 is a separator.

**Alternatives considered.** Discover labels by globbing `OUT/*/sim3_pose_both.csv`; pass labels on the CLI.

**Why this choice.** The table is a fixed comparison: exactly the arms the design doc's 2026-09-06 inference-time table reports, in that order. A glob would also pick up the retired retroactive per-scene-median rows still on disk (honesty-audit item 13), which the doc explicitly says are "no longer the reported configuration" (decision 0.1b, revised 2026-09-06).

### Lines 24-26: the three CUT3R model rows

```
24:     (r"CUT3R zero-shot (pretrained ckpt)", "cut3r_zeroshot", "model"),
25:     (r"CUT3R finetuned (\texttt{augfull\_lr1e5})", "augfull_lr1e5", "model"),
26:     (r"Champion: gated fwd$\times$bwd fusion (\texttt{augfull\_cg\_fuse\_g7})", "augfull_cg_fuse_g7", "model"),
```

**What it does.** Three rows in group `"model"`. Display names are raw strings so LaTeX backslashes survive (`\texttt{}`, escaped underscores `\_`, math `$\times$`). `cut3r_zeroshot` is the pretrained `cut3r_512_dpt_4_64.pth` checkpoint run fresh on the 12 smoke scenes (design doc, sweep 2; Slurm 45453660). `augfull_lr1e5` is the finetuned checkpoint that the DROID harness calls "Regular CUT3R". `augfull_cg_fuse_g7` is the campaign grand champion: confidence-gated forward × backward passes fused per scene. These directories contain no `summary.json`, so their diagnostics cell renders as `--` (line 69).

**Alternatives considered.** Only the finetuned row (the arm the OpenCV pipeline borrows its focal from); or the finetuned row plus the zero-shot row without the champion.

**Why this choice.** Decision 0.3 requires the smoke set to be one on which "augfull_lr1e5 and augfull_cg_fuse_g7 both have preds+eval on all 12, so the finetuned checkpoint and the OpenCV arm are compared on identical scenes". The zero-shot row was added in sweep 2 as the reference the classical arm actually reaches on ATE ("ATE is stuck at ~117-125 mm for every variant = parity with zero-shot"). The champion row is the ceiling the whole control arm is a control *for* (design doc purpose statement). Numbers these rows reproduce from the doc: zero-shot 118.0 / 10.12 / 1.506, finetuned 72.3 / 6.54 / 0.960, champion 62.8 / 6.02 / 0.860 (ATE mm / RPE-t mm / RPE-r deg), with ATE medians 118.4 / 69.8 / 55.1. Honesty-audit item 9 applies: the finetuned row is causal, the champion row uses a backward pass and is therefore not.

### Lines 27-31: the four OpenCV rows and the list close

```
27:     (r"OpenCV VO v0 + zero-shot CUT3R focal at frame $t$", "vo6_zs_perframe", "opencv"),
28:     (r"OpenCV VO v0 + finetuned CUT3R focal at frame $t$", "vo6_ft_perframe", "opencv"),
29:     (r"OpenCV VO v0 + finetuned CUT3R focal, running median to $t$", "vo6_ft_causal", "opencv"),
30:     (r"OpenCV VO v0, fixed focal 203\,px (model-free)", "vo4_fnominal", "opencv"),
31: ]
```

**What it does.** Four rows in group `"opencv"`, all the same frozen v0 pipeline (`--cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff`, lever ON) and differing only in the focal fed to every per-frame camera matrix K (pp = image centre per decision 0.1b; distortion coefficients passed as `None`, `opencv_vo.py` line 186): frame t's focal from the zero-shot checkpoint's `camera/*.npz` (`--focal perframe:<zs preds>`); frame t's focal from the finetuned checkpoint (`--focal perframe:<ft preds>`, the reported configuration); the running median of the finetuned per-frame focals up to t; and a constant 203 px (`--focal 203`). `\,` is a LaTeX thin space between number and unit. The `vo6_` rows reproduce the numbers of the 2026-09-06 inference-time table (Slurm 45485470) and `vo4_fnominal` those of the 2026-09-05 fixed-203-px row (Slurm 45453975); the label names themselves are not recorded in the design doc, so this mapping is inferred from the on-disk numbers matching those rows.

**Alternatives considered.** Decision 0.1b lists the focal sources: calibrated K from `cam/*.npz`; one nominal dataset K; model-estimated focal (per-scene median or per-frame); self-calibration (`--focal selfcal`). The doc also measured a retroactive per-scene-median row (111.9 / 6.93 / 0.871) and a calibrated-GT-focal row (116.0 / 6.91 / 0.885).

**Why this choice.** Decision 0.1b as revised 2026-09-06: "OpenCV at frame t may only use what CUT3R has produced up to t; retroactive per-scene medians are not an inference-time method", so the per-scene-median row is dropped and the two per-frame rows plus the running-median row replace it. The doc's readings, which these rows reproduce: finetuned per-frame 110.4 / 7.53 / 0.858 (ATE median 91.5, 19.1 % failed, 51 reboots); running median 113.8 / 6.85 / 0.844 (98.9, 19.6 %, 51); zero-shot per-frame 124.8 / 10.33 / 1.059 (109.4, 14.8 %, 63); fixed 203 px 109.1 / 7.04 / 0.869 (92.7, 16.4 %, 54). The 203 px row is kept because the doc concludes "`--focal 203` gives a strictly model-free arm at no cost" and "focal source is immaterial"; honesty-audit item 4 notes that 203 px is the calibrated value in the 320x192 frame, so "model-free" does not mean "calibration-free". The calibrated-GT-focal row is not shown because decision 0.1b keeps it as "a one-off diagnostic"; self-calibration is not shown because the doc rules it "not recommended" (per-scene focals scattered 132-510 px against a true ~203; 116.6 / 7.86 / 0.935).

### Lines 32-34: `load_label` signature

```
32: 
33: 
34: def load_label(out_root, label, scenes):
```

**What it does.** Defines the per-label loader. `scenes` is the ordered list of scene ids from the scene list; the function returns a flat `dict` of scalars for one table row. Lines 32-33 are PEP 8 separators.

**Alternatives considered.** A single pass that loads all labels into one DataFrame keyed by (label, scene).

**Why this choice.** One label per call mirrors the on-disk layout (`<OUT>/<label>/...`) and lets a missing label fail loudly at its own `open()`. Not a design-doc decision.

### Lines 35-36: read the scorer CSV and restrict to the scene list

```
35:     rows = {r["scene"]: r for r in csv.DictReader(open(os.path.join(out_root, label, "sim3_pose_both.csv")))}
36:     rows = [rows[s] for s in scenes if s in rows]
```

**What it does.** Parses `sim3_pose_both.csv` (header `scene, ate_mean, rpe_trans_mean, rpe_rot_mean, ate_rmse, rpe_trans_rmse, rpe_rot_rmse`; one row per scene) into a dict keyed by scene id, then re-lists it in scene-list order, silently skipping scenes the label has no row for. The file handle is left to CPython's refcount to close.

**Alternatives considered.** Take every row in the CSV regardless of the scene list; or raise on a missing scene.

**Why this choice.** Decision 0.3 pins the comparison to `cg_smoke_scenes_12.txt`; filtering by the list guarantees every row is scored on the same scene set even if a label's CSV contains more scenes (e.g. a 430-subset scoring run). The silent skip is a known limitation: a label missing a scene would be averaged over fewer scenes without warning, and only the first row's count is printed in the title (line 72). On the current disk state all seven labels have all 12 scenes.

### Lines 37-39: per-column arrays and unit conversion

```
37:     ate = np.array([float(r["ate_mean"]) for r in rows]); rt = np.array([float(r["rpe_trans_mean"]) for r in rows])
38:     rr = np.array([float(r["rpe_rot_mean"]) for r in rows])
39:     vals = dict(n=len(rows), ate=1000 * ate.mean(), ate_med=1000 * np.median(ate), rpe_t=1000 * rt.mean(), rpe_r=rr.mean())
```

**What it does.** Builds three length-`n` float arrays from the `_mean` columns (each already a per-scene mean over frames after Sim(3) alignment on that scene's trajectory centres). `ate_mean` and `rpe_trans_mean` are in metres and are multiplied by 1000 to millimetres; `rpe_rot_mean` is in degrees and is left as is. Four scalars per label: mean ATE, median ATE (median over the 12 per-scene means, not over frames), mean RPE-trans, mean RPE-rot. `n` records how many scenes contributed.

**Alternatives considered.** Use the `_rmse` columns (the current `eval_depth_poses.py` default and what the full-4292 table uses: "pose RMSE-reduced per scene"); report medians for every column; keep SI metres.

**Why this choice.** Every sweep table in the design doc (sweeps 1-3, focal-source, focal-sensitivity, inference-time focal) is stated as "Means over 12 scenes (ATE mm / RPE-trans mm / RPE-rot deg)" from `pose_sim3_both.py`, so `_mean` and mm/deg are the units those numbers were decided in. An ATE-median column first appears in the doc's sweep-3 table, where the scale hand-off (decision 2.15) moved the median more than the mean ("ATE 116.8 -> 111.9 mean, 117 -> 98 median"); the median exposes short-scene behaviour while the mean is pulled by the 400-700-frame scenes at 130-195 mm. The choice of mean-over-scenes (unweighted by frame count) matches `aggregate_results.py`'s nanmean-over-scenes convention cited in the full-harness section.

### Lines 40-41: load the OpenCV diagnostics

```
40:     summ = [json.load(open(f)) for f in glob.glob(os.path.join(out_root, label, "preds", "*", "summary.json"))]
41:     summ = [s for s in summ if s["scene"] in set(scenes)]
```

**What it does.** Globs every `preds/<scene>/summary.json` under the label and keeps those whose `"scene"` field is in the scene list. For the three model labels the glob matches nothing (their `preds/<scene>/` holds `camera/`, `depth/` etc. but no `summary.json`), so `summ` is empty and the diagnostics keys are never set. `set(scenes)` is rebuilt per iteration; harmless at 12 scenes.

**Alternatives considered.** Read the per-frame `diag.csv` and recompute; require a diagnostics file for every row.

**Why this choice.** Decision 0.4 puts diagnostics "in a side CSV (not the harness CSV)" precisely so the harness CSV stays comparable across arms while the classical arm carries extra state; `summary.json` is the per-scene aggregate of that side CSV (`failed_frames, reboots, keyframes, median_inliers`, per the v0 FINAL recipe text). Reading the aggregate rather than `diag.csv` avoids re-deriving the failed-flag rule here.

### Lines 42-46: pool the diagnostics and return

```
42:     if summ:
43:         fr = sum(s["frames"] for s in summ)
44:         vals["failed"] = 100.0 * sum(s["failed_frames"] for s in summ) / fr
45:         vals["reboots"] = sum(s["reboots"] for s in summ)
46:     return vals
```

**What it does.** When any diagnostics were found: `failed` = failed frames as a percentage of all frames pooled across the 12 scenes (frame-weighted, so long scenes dominate, unlike the frame-count-agnostic pose means above); `reboots` = total number of re-bootstraps summed over the 12 scenes. A "failed frame" is one where the live PnP returned fewer than 30 inliers (decision 3.1) and the pose was held (decision 3.2); a "reboot" is a fresh essential-matrix bootstrap (decision 2.9/2.10) triggered by 3 consecutive PnP failures (3.2b) or map starvation. `keyframes`, `median_inliers`, `map_points`, `seconds`, and the focal statistics in the JSON are not surfaced.

**Alternatives considered.** Decision 0.4 lists: "ATE/RPE only; + failed-frame count; + failed-frame count and median inliers". The design-doc sweep tables additionally show keyframes per 100 frames, median PnP inliers and seconds per scene.

**Why this choice.** Decision 0.4 picked failed-frame count plus median inliers, reasoning that it "separates lost-tracking from drift; median inliers moves before ATE does". The table shows the failed percentage and the reboot count (the two that the sweep readings use to explain ATE: "it is set by the 30-58 re-bootstraps per 12 scenes, each starting a new arbitrary-scale segment that Sim(3) cannot repair"), and omits median inliers, which was a sweep-time tuning signal rather than a headline. Pooling is frame-weighted so the percentage means "share of all 4243 smoke frames", which is how the doc's failed% values were produced: frame-weighted pooling of the on-disk `summary.json` files reproduces the inference-time table's 19.1 / 19.6 / 14.8 / 16.4 exactly, whereas an unweighted mean over scenes would not (it gives 21.2 / 21.0 / 18.0 / 19.4). Honesty-audit item 12: retro-filled pre-bootstrap frames are not counted as failed (they were accepted at >= 4 inliers, item 2), so this percentage undercounts frames whose pose did not come from live PnP.

### Lines 47-52: `main`, argument parser, first two flags

```
47: 
48: 
49: def main():
50:     ap = argparse.ArgumentParser()
51:     ap.add_argument("--out_root", default=OUT_DEFAULT)
52:     ap.add_argument("--scene_list", default=os.path.join(os.path.dirname(__file__), "cg_smoke_scenes_12.txt"))
```

**What it does.** Opens `main()` and defines `--out_root` (defaults to `OUT_DEFAULT`) and `--scene_list`, defaulting to `cg_smoke_scenes_12.txt` next to the script (resolved relative to `__file__`, so it works from any cwd). Lines 47-48 are separators.

**Alternatives considered.** The other eval scopes in decision 0.3: the 430 subset and the full 4292.

**Why this choice.** Decision 0.3: "smoke on eval_pipeline/cg_smoke_scenes_12.txt (all 12 are inside the 430 subset ...)". Decision 0.3 chose this list because `augfull_lr1e5` and `augfull_cg_fuse_g7` already had preds+eval on all 12; the zero-shot row was run fresh on the same 12 scenes (Slurm 45453660, design doc sweep 2).

### Lines 53-55: output path and scene list

```
53:     ap.add_argument("--tex", default=os.path.join(OUT_DEFAULT, "tables", "opencv_vo_smoke12.tex"))
54:     args = ap.parse_args()
55:     scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
```

**What it does.** `--tex` names the output `.tex`; the `.pdf`, `.png`, `.build.log` and pdflatex's own `.log`/`.aux` land beside it under the same basename. Note the default is anchored to `OUT_DEFAULT`, not to `--out_root`, so overriding `--out_root` alone still writes the table under the default root. The scene list is read as one id per line, blank lines dropped.

**Alternatives considered.** Derive `--tex` from `--out_root`; write the `.tex` into the repo.

**Why this choice.** Tables live on scratch next to the data they summarise (the full-4292 table is at the same `tables/` location per the design doc). Plumbing; the `OUT_DEFAULT` anchoring is a known wart, not a decision.

### Lines 56-59: load every row and find column minima

```
56: 
57:     data = [(name, load_label(args.out_root, lbl, scenes), grp) for name, lbl, grp in ROWS]
58:     cols = ["ate", "ate_med", "rpe_t", "rpe_r"]
59:     best = {c: min(v[c] for _, v, _ in data) for c in cols}
```

**What it does.** Calls `load_label` once per `ROWS` entry, keeping the display name and group. `cols` fixes the four numeric columns in display order. `best` is the minimum of each column over all seven rows, model rows included; lower is better for every column (arrows in the header, line 90). Line 56 is a separator.

**Alternatives considered.** Bold the best within each group separately; bold with a significance test; no bolding.

**Why this choice.** A single global minimum makes the table answer the control-arm question directly: does the classical arm ever beat the models? On the compiled table the champion holds ATE, ATE median and RPE-trans, and the running-median OpenCV row holds RPE-rot (0.84 vs champion 0.86). The design doc states the reading this bolding produces: "The OpenCV arm beats each checkpoint on RPE-rot, ties/loses on ATE, and never beats the finetuned model on ATE." No significance test is applied (see honesty notes).

### Lines 60-64: row loop and group separators

```
60:     lines, prev = [], None
61:     for name, v, grp in data:
62:         if prev is not None and grp != prev:
63:             lines.append(r"\midrule")
64:         prev = grp
```

**What it does.** Iterates rows in `ROWS` order, inserting a booktabs `\midrule` whenever the group string changes (once, between the third model row and the first OpenCV row). `prev is not None` avoids a rule before the first row.

**Alternatives considered.** `\addlinespace`, a blank row, or multi-row group labels.

**Why this choice.** booktabs convention inherited from the teal recipe (line 7); the doc's own sweep tables separate model reference rows from classical rows the same way.

### Lines 65-68: format the four numeric cells

```
65:         cells = []
66:         for c in cols:
67:             txt = f"{v[c]:.1f}" if c != "rpe_r" else f"{v[c]:.2f}"
68:             cells.append(r"\textbf{" + txt + "}" if abs(v[c] - best[c]) < 1e-12 else txt)
```

**What it does.** Millimetre columns (ATE, ATE median, RPE-trans) print with one decimal; RPE-rot in degrees prints with two. A cell is wrapped in `\textbf{}` when it equals the column minimum to within 1e-12, i.e. exact floating-point equality in practice (a true tie would bold both cells).

**Alternatives considered.** The design doc reports RPE-trans with two decimals (6.93, 7.53) and RPE-rot with three (0.858, 0.871); three-decimal RPE-rot is what the console summary prints (line 121).

**Why this choice.** Display precision was chosen to match what the doc calls "within noise": the focal-source check found the OpenCV rows differ by fractions of a millimetre RPE-trans and hundredths of a degree RPE-rot ("`--focal 203` ... is within noise of the v0 final"), so extra decimals would suggest resolution the 12-scene means do not have. Consequence visible in the compiled table: finetuned-per-frame 0.858 and champion 0.860 both display as 0.86, while the bold still goes to the exact minimum 0.844 (displayed 0.84). Not a benchmarked choice.

### Lines 69-70: diagnostics cell and row assembly

```
69:         diag = f"{v['failed']:.0f}\\,\\% / {v['reboots']}" if "failed" in v else "--"
70:         lines.append(f"{name} & " + " & ".join(cells) + f" & {diag} \\\\")
```

**What it does.** For OpenCV rows renders `<failed %, no decimals>\,\% / <reboots>` (the doubled backslashes in the non-raw f-string become single ones, so LaTeX sees `19\,\% / 51`: a thin space and an escaped percent sign). Model rows get `--` (an en-dash in LaTeX). Line 70 joins name, four cells and the diagnostics cell with `&` and ends the row with `\\` (four backslashes in source, two in output).

**Alternatives considered.** Two separate columns; a blank cell instead of `--` for the model rows; one decimal on the percentage.

**Why this choice.** One combined cell (failed % / reboots) is a layout choice; the doc's readings use both numbers together to explain ATE. Zero decimals on the failed percentage: the doc's sweep readings quote it at one decimal (19.1 %, 14.8 %) but the per-row differences the doc draws conclusions from are several points (e.g. min-inliers 10 -> 9.5 %, 30 -> 42.7 % in sweep 1), so integer precision carries the message. `--` rather than blank so the reader sees that model arms have no notion of PnP failure or re-bootstrap, not that the value is missing.

### Lines 71-72: scene count for the title

```
71: 
72:     n = data[0][1]["n"]
```

**What it does.** Takes the number of scenes that contributed to the first row (`cut3r_zeroshot`) and uses it in the title (line 82). Line 71 is a separator.

**Alternatives considered.** `len(scenes)`; the minimum over rows; asserting all rows have equal `n`.

**Why this choice.** Plumbing. Known limitation: if a later label were missing a scene (line 36 skips silently), the title would still say 12. On the current disk state every label has 12.

### Lines 73-78: LaTeX preamble — class and packages

```
73:     tex = r"""\documentclass[border=8pt,varwidth=25cm]{standalone}
74: \usepackage{booktabs}
75: \usepackage{amssymb}
76: \usepackage[T1]{fontenc}
77: \usepackage[table]{xcolor}
78: \usepackage{newtxtext,newtxmath}
```

**What it does.** Starts a raw triple-quoted string holding the document. `standalone` with `border=8pt` crops the page to the content plus an 8 pt margin (so the PDF/PNG is a tight table, not an A4 page) and `varwidth=25cm` wraps content in a `varwidth` box of at most 25 cm, which is what allows the `\linewidth`-relative footnote minipage (line 96) to have a defined width. `booktabs` supplies `\toprule/\midrule/\cmidrule/\bottomrule`. `amssymb` and `xcolor[table]` are loaded but nothing in this table body uses them (no AMS symbols beyond kernel `\downarrow`/`\times`; no cell colours). `fontenc` T1 gives proper 8-bit font encoding. `newtxtext,newtxmath` set a Times-like text and maths font.

**Alternatives considered.** `article` class with `\pagestyle{empty}` and `pdfcrop`; Computer Modern (default fonts); `siunitx` `S` columns for decimal alignment; `\usepackage{colortbl}` directly.

**Why this choice.** Line 7 states it: the "same standalone/booktabs/newtx recipe as build_gru_vs_prevpred_teal.py". That builder (its lines 91-96) loads the identical `standalone[border=8pt,varwidth=25cm]` / `booktabs` / `amssymb` / `fontenc[T1]` / `xcolor[table]` / `newtxtext,newtxmath` preamble and uses `\cellcolor` (its line 75), which is why `amssymb` and `xcolor[table]` are carried over here rather than pruned; harmless. Note that `build_opencv_full_table.py` (lines 82-84) uses `standalone[11pt,border=8pt]` + `booktabs` + `xcolor[table]` but not `newtx`, so the two OpenCV tables are not font-identical. `standalone` is also why the system `/usr/bin/pdflatex` cannot be used (line 111: it "lacks standalone.cls").

### Lines 79-83: document start, title, body font

```
79: 
80: \begin{document}
81: \begin{center}
82: {\normalsize\bfseries OpenCV visual-odometry control arm vs.\ CUT3R on the DROID wrist smoke set (""" + str(n) + r""" scenes)}\par\vspace{8pt}
83: \footnotesize
```

**What it does.** Opens the document and a centred block. Line 82 ends the raw string, splices in `str(n)` (the scene count from line 72) with ordinary string concatenation, and reopens a raw string: the title reads "... smoke set (12 scenes)" in bold `\normalsize`, followed by an 8 pt vertical gap. `vs.\ ` uses a control space so LaTeX does not treat the period as a sentence end. `\footnotesize` sets the table body font. Line 79 is a blank line inside the LaTeX source.

**Alternatives considered.** `\caption` inside a `table` float (not available in `standalone` without extra packages); no title.

**Why this choice.** The title states the three facts the reader needs to place the table: it is the control arm, it is compared to CUT3R, and it is the smoke set (decision 0.3), not the 430 or 4292 scopes. Plumbing otherwise.

### Lines 84-86: spacing and column spec

```
84: \setlength{\tabcolsep}{7pt}
85: \renewcommand{\arraystretch}{1.15}
86: \begin{tabular}{lccccc}
```

**What it does.** 7 pt horizontal padding on each side of every cell, 1.15x row height, and a six-column layout: one left-aligned method-name column and five centred numeric/diagnostic columns.

**Alternatives considered.** `siunitx` `S` columns aligning on the decimal point; right-aligned `r` numeric columns; `tabularx` to fill a fixed width.

**Why this choice.** Recipe inherited from the teal builder (line 7); fixed decimal counts per column (line 67) make centred alignment read cleanly without `siunitx`. Not a design-doc decision.

### Lines 87-90: header rows

```
87: \toprule
88:  & \multicolumn{4}{c}{Pose (Sim(3)-aligned, per-scene mean, then mean over scenes)} & Diagnostics \\
89: \cmidrule(lr){2-5}\cmidrule(lr){6-6}
90: Method & ATE (mm) $\downarrow$ & ATE median (mm) $\downarrow$ & RPE$_{\mathrm{trans}}$ (mm) $\downarrow$ & RPE$_{\mathrm{rot}}$ ($^{\circ}$) $\downarrow$ & failed frames / re-bootstraps \\
```

**What it does.** A two-level header. Line 88 spans columns 2-5 with a group label that spells out the aggregation ("per-scene mean, then mean over scenes", i.e. the `_mean` columns of line 37-39 then `np.mean`/`np.median` across the 12) and labels column 6 "Diagnostics". Line 89 draws trimmed partial rules under each group (`(lr)` trims both ends so the two rules do not touch). Line 90 names the columns with units and a down-arrow meaning lower is better; the ATE-median column's label makes clear it is a median over scenes, not a different per-scene reduction.

**Alternatives considered.** Report RMSE-reduced columns ("per-scene RMSE") as the current harness default and the full-4292 table do; put units in a separate row.

**Why this choice.** Stating the aggregation in the header is the only place the reader is told these are `_mean` numbers, which matters because the same scenes reduce to different values under `_rmse` (the CSV carries both reductions side by side). Sim(3) alignment is what the scoring section of the design doc prescribes ("Scoring: Sim(3)-aligned ATE / RPE ... A global scale is free; scale DRIFT is not"), and it is why the OpenCV arm's arbitrary map scale (decision 3.3, "as given by the map") can be scored at all.

### Lines 91-95: rows and table close

```
91: \midrule
92: """ + "\n".join(lines) + r"""
93: \bottomrule
94: \end{tabular}
95: \par\vspace{6pt}
```

**What it does.** A rule under the header, then the raw string is closed, the seven formatted rows (plus the one inter-group `\midrule`, line 63) are joined with newlines and spliced in, and the raw string reopens for `\bottomrule` and the end of the tabular. A 6 pt gap precedes the footnote.

**Alternatives considered.** Building the whole document with a templating engine or `str.format`; both would have to escape every LaTeX brace.

**Why this choice.** The raw-string-splice pattern is the sibling builders' convention; it keeps LaTeX readable in the source at the cost of the two visible `"""` seams (here and line 82). Plumbing.

### Lines 96-100: footnote, first half — the pipeline in one sentence

```
96: \begin{minipage}{0.97\linewidth}\scriptsize
97: OpenCV VO v0: Shi--Tomasi + LK tracks, parallax-gated essential-matrix bootstrap (0.5\,px), PnP against a
98: triangulated map, keyframes every 5\,px of parallax, outlier culling, re-bootstrap after 3 failed frames with
99: scale hand-off, static-track lever on; images only plus the stated focal, which at frame $t$ uses only what CUT3R
100: has produced up to $t$ (inference-time). ``Failed frames'':
```

**What it does.** Opens a `\scriptsize` minipage at 97 % of the enclosing width (defined thanks to `varwidth`) and summarises the frozen v0 configuration so the table is self-describing: Shi-Tomasi corners tracked by Lucas-Kanade (decisions 1.3, 1.5); bootstrap by essential matrix with 0.5 px RANSAC threshold, gated on parallax (2.9, 2.10); PnP against a triangulated monocular map (0.1, 0.2, 2.1); a keyframe whenever median track displacement since the last keyframe reaches 5 px (2.12); culling of map points after 3 consecutive PnP-outlier hits (2.14); re-bootstrap after 3 consecutive failed frames (3.2b) with scale hand-off across the segment boundary (2.15); the static-track lever ON (1.2); and the causal focal rule (0.1b as revised). The sentence ends mid-line, continuing into the next group.

**Alternatives considered.** No footnote (rely on the design doc); a full parameter table (the "Frozen v0 parameters" table in the doc).

**Why this choice.** Every number in the footnote is a frozen v0 parameter from the design doc's table: bootstrap E threshold 0.5 px, keyframe/bootstrap trigger 5 px, lever ON by default since 2026-09-05 (a user decision). The footnote names them so a reader of the PNG alone knows the OpenCV rows are one fixed configuration and not four tuned ones. "Images only plus the stated focal" is the closed-loop rule from decision 0.1 ("plain image inputs only, no privileged info") and 0.1b ("model output is not privileged").

### Lines 101-106: footnote, second half, and document end

```
101: pose held because PnP found $<30$ inliers; each re-bootstrap starts a new arbitrary-scale segment. Model rows
102: use the same scorer on the same scenes. Depth metrics not shown (the OpenCV arm predicts poses only).
103: \end{minipage}
104: \end{center}
105: \end{document}
106: """
```

**What it does.** Finishes the failed-frame definition (fewer than 30 PnP inliers, pose held: decisions 3.1 and 3.2), explains what a re-bootstrap costs (a new segment with its own arbitrary scale, which Sim(3)'s single global scale cannot undo: decision 3.3 and the sweep-2 reading), states that the model rows were scored identically, and repeats the depth-column omission. Closes minipage, centre block, document, and the Python raw string.

**Alternatives considered.** Add the retro-fill clause the full-4292 table carries ("footnote clause", honesty-audit item 1 as accepted 2026-09-07); mark the champion row as non-causal (item 9).

**Why this choice.** The 30-inlier floor is decision 3.1 (sweeps 1-2: "30 vs 20: 0.861 vs 0.923 RPE-r, 7.07 vs 7.34 RPE-t, ATE 116.8 vs 120.0"), and "hold last pose + flag" is 3.2 (constant velocity was mixed: "121.9/8.47/1.375 vs hold 122.9/9.65/1.129"; rotation worse). The retro-fill and champion-causality caveats are absent from this footnote: the retro-fill ruling came on 2026-09-07 with the clause placed in `build_opencv_full_table.py`; this smoke footnote predates that ruling. See the honesty notes.

### Lines 107-109: write the `.tex`, derive directory and basename

```
107:     os.makedirs(os.path.dirname(args.tex), exist_ok=True)
108:     open(args.tex, "w").write(tex)
109:     d, base = os.path.dirname(args.tex), os.path.splitext(os.path.basename(args.tex))[0]
```

**What it does.** Creates `tables/` if needed, overwrites the `.tex`, and splits the path into the directory to `cd` into and the extension-less basename that pdflatex and gs will use for `.pdf`, `.png` and `.build.log`.

**Alternatives considered.** `tempfile` build directory with the artefacts copied out; `latexmk`.

**Why this choice.** Building in place leaves `pdflatex`'s `.log`/`.aux` beside the `.tex`, which is what the sibling builders do and what makes a failed build inspectable. Plumbing.

### Lines 110-113: the MN5 LaTeX environment

```
110:     # Prefer the MN5 latex module when lmod is present (compute/older login nodes); fall back to /usr/bin/pdflatex.
111:     # The MN5 LaTeX tree is used directly (the system /usr/bin/pdflatex lacks standalone.cls); no lmod needed.
112:     modload = ("export TEXMFROOT=/apps/GPP/LATEX/20240430 && export TEXMFCNF=$TEXMFROOT:$TEXMFROOT/texmf-dist/web2c && "
113:                "export PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:$PATH; ")
```

**What it does.** Builds a shell prefix that points a TeX Live 2024 installation at `/apps/GPP/LATEX/20240430` without `module load`: `TEXMFROOT` names the tree, `TEXMFCNF` tells kpathsea where to find `texmf.cnf` (the root and its `web2c` directory), and the tree's `bin/x86_64-linux` is prepended to `PATH` so `pdflatex` resolves there rather than to `/usr/bin/pdflatex`. The two comments contradict each other: line 110 describes a module-or-fallback strategy, line 111 describes what the code actually does (direct tree, no lmod, no fallback). Line 111 is current; line 110 is stale.

**Alternatives considered.** `module load latex` through lmod (requires lmod to be initialised in the non-interactive `bash -c` shell, which is not guaranteed); the system `/usr/bin/pdflatex` (present, but without `standalone.cls`); a container.

**Why this choice.** Per the comment at line 111, the system `/usr/bin/pdflatex` lacks `standalone.cls`, and driving the tree through environment variables works in a bare `bash -c` without lmod. Not a design-doc decision; the stale comment is a known wart.

### Lines 114-115: compile and rasterise

```
114:     cmd = (modload + f"cd {d} && pdflatex -interaction=nonstopmode {base}.tex >{base}.build.log 2>&1 && "
115:            f"gs -sDEVICE=png16m -r300 -o {base}.png -dBATCH -dNOPAUSE {base}.pdf >/dev/null")
```

**What it does.** One `&&`-chained shell command: change into the table directory; run `pdflatex` in `nonstopmode` (never waits for keyboard input on an error, exits non-zero on a fatal error) with all its console output captured to `<base>.build.log`; then run Ghostscript with `-sDEVICE=png16m` (24-bit RGB PNG), `-r300` (300 dpi), `-o <base>.png` (output file; `-o` also implies `-dBATCH -dNOPAUSE`, so the two explicit flags are redundant), input `<base>.pdf`, Ghostscript's stdout discarded. `standalone` produces a single page, so one PNG results.

**Alternatives considered.** `pdftoppm -png -r 300`; ImageMagick `convert` (needs a policy that allows PDF); two passes of pdflatex (unnecessary here: no cross-references, no `\ref`, so one pass converges).

**Why this choice.** The docstring (line 7) names `/usr/bin/gs`, the system Ghostscript, as the rasteriser, and 300 dpi yields a PNG legible in a report or chat. Plumbing.

### Lines 116-119: run, check, report

```
116:     r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
117:     if r.returncode != 0:
118:         raise SystemExit(f"compile failed: {r.stderr[-1500:]}\n{r.stdout[-800:]}")
119:     print("OK:", args.tex, "+ .pdf + .png")
```

**What it does.** Executes the chain in a fresh `bash -c` (so the `export`s take effect and `cd` is local to that shell), capturing stdout/stderr as text. A non-zero exit from any link aborts with the last 1500 characters of stderr and 800 of stdout; because pdflatex's output was redirected into `.build.log`, what surfaces here is mostly Ghostscript's or the shell's, and the pdflatex diagnostics have to be read from `<base>.build.log`. On success prints the `.tex` path.

**Alternatives considered.** `check=True` (raises `CalledProcessError` with the full output); tailing `.build.log` into the error message.

**Why this choice.** Consistent with the sibling builders; the tail-truncation keeps a failed run's console readable. Known limitation: on a pdflatex error the printed excerpt is uninformative and one must open `<base>.build.log`.

### Lines 120-121: console summary

```
120:     for name, v, _ in data:
121:         print(f"  {name:70s} ATE {v['ate']:6.1f} med {v['ate_med']:6.1f} RPEt {v['rpe_t']:5.2f} RPEr {v['rpe_r']:.3f}")
```

**What it does.** Prints one line per row with the LaTeX display name left-aligned and padded on the right to 70 characters (`{name:70s}`) and the four numbers at higher precision than the table: ATE and median at one decimal, RPE-trans at two decimals, RPE-rot at three (the precision the design doc's tables use, e.g. 110.4 / 7.53 / 0.858). The diagnostics are not printed.

**Alternatives considered.** Print nothing; print the diagnostics too.

**Why this choice.** Gives the three-decimal values needed to transcribe into the design doc, where sweeps are recorded at that precision, without cluttering the rendered table. Plumbing.

### Lines 122-125: entry point

```
122: 
123: 
124: if __name__ == "__main__":
125:     main()
```

**What it does.** Standard guard so the module can be imported (e.g. to reuse `load_label` or `ROWS`) without running the build. Lines 122-123 are separators.

**Alternatives considered.** Run `main()` unconditionally.

**Why this choice.** Convention. Not a design-doc decision.

### Known limitations / honesty notes for this block

- **Aggregation differs from the full-4292 table.** This table uses `pose_sim3_both.py`'s `_mean` per-scene reduction; the full-harness table (`build_opencv_full_table.py`) uses `eval_depth_poses.py` "pose RMSE-reduced per scene". The smoke rows (e.g. finetuned 72.3 mm) and the full-4292 rows (e.g. Regular CUT3R 0.076 m) are therefore not the same statistic and should not be read as the same scenes at two scales.
- **No significance test behind the bold.** Line 59 bolds the exact column minimum. The RPE-rot bold falls on the running-median OpenCV row (0.844, shown 0.84) over the champion (0.860) and finetuned-per-frame OpenCV (0.858); the design doc treats focal-source differences of this size as "within noise of the v0 final". The bold reports ordering only.
- **RPE-trans cannot separate arms.** The metric-floor analysis (144 sampled scenes, full-harness convention) found RPE-trans "SATURATED at the no-motion floor (7.9 mm = the RMS step itself) for every arm except the champion (7.2)", and ATE of the zero-shot and OpenCV rows "sits at the constant-velocity floor (0.12)". The table bolds RPE-trans regardless; "RPE-rot and ATE are the informative columns".
- **Retro-fill is in the numbers but not in the footnote.** The OpenCV rows include the non-causal retroactive PnP fill of pre-bootstrap frames (honesty-audit item 1, accepted 2026-09-07 with a footnote clause in the full-4292 table only). Ablation on these scenes: with retro-fill 110.4 / 7.5 / 0.858, strictly causal 110.6 / 7.7 / 0.881 (+0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r; no ordering changes). This smoke footnote (lines 97-102) predates the ruling and does not mention it.
- **Failed-frame percentage undercounts.** Retro-filled frames are accepted at >= 4 inliers versus 30 live and are marked not-failed (audit items 2 and 12), so the "failed frames" cell excludes them.
- **Causality of the model rows is mixed.** The finetuned row is causal; the champion row is a forward x backward fusion (audit item 9). The footnote does not say so.
- **"Model-free" focal is calibration-derived.** The 203 px row's focal is the calibrated value in the 320x192 frame (audit item 4); the doc's verified claim is that it is model-free, not knowledge-free.
- **Every threshold was tuned on these 12 reported scenes** (audit item 7), which is why the doc addresses this by freezing the configuration and reporting on all 4292; this smoke table is the tuning-set table, not the held-out one.
- **Silent scene drop.** Line 36 skips scenes missing from a label's CSV without warning, and line 72 reports only the first row's scene count. Currently every label has all 12.
- **Stale comment at line 110** describes a module/fallback strategy the code does not implement; line 111 is accurate.
- **`--tex` default is anchored to `OUT_DEFAULT`, not `--out_root`** (line 53); overriding only the root still writes the table under the default root.
- **Unused packages.** `amssymb` and `xcolor[table]` (lines 75, 77) are inherited from the teal recipe and unused by this table body.
