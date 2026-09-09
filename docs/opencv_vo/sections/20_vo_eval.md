## `eval_pipeline/opencv_vo_eval.py`, lines 1-55: harness scoring of the OpenCV predictions

This is the whole file. It takes the per-scene predictions written by `eval_pipeline/opencv_vo.py`
(`<out_root>/<label>/preds/<scene>/camera/%06d.npz`, one c2w pose + K per frame, plus a `depth`
symlink into the paired CUT3R run) and scores every scene with the same evaluator every model arm
is scored with, `eval_bundle/bin/eval_depth_poses.py` at its default arguments, launched as one
subprocess per scene from a `multiprocessing.Pool`. Each subprocess writes
`<out_root>/<label>/eval/<scene>/eval_depth_pose_metrics.csv`; the table builder
(`eval_pipeline/build_opencv_full_table.py`) later reads each scene's `ALL/MEAN` row and takes the
`nanmean` over scenes, which the design doc calls the `aggregate_results.py` convention with "pose
RMSE-reduced per scene" (design doc, FULL HARNESS section). In the pipeline it sits between the VO
run and the table: `opencv_vo.py` (poses) -> **this script** (per-scene CSVs) ->
`build_opencv_full_table.py` (the 4292-scene table). It is the scoring path behind the full-harness
rows (Slurm 45527508 / 45529336, "48 cores, ~16 min per row + scoring"); the earlier 12-scene sweeps
in the design doc were scored with `eval_pipeline/pose_sim3_both.py` instead, so sweep numbers
(means in mm) and full-harness numbers (metres, harness evaluator) are not the same scorer. The
script has no geometry of its own: it is bookkeeping for `skip` / `missing` / `fail` / `ok`.

### Lines 1-3: shebang and the one-sentence contract

```
1: #!/usr/bin/env python
2: """Score OpenCV-VO preds with the SAME harness eval as every model arm (eval_depth_poses.py, default args),
3: one subprocess per scene, in parallel. Mirrors cg_fuse_fwd_bwd.py's eval step.
```

**What it does.** Standard `env python` shebang (the file is run with the interpreter of the active
environment; line 27 below re-uses that same interpreter for the children). The docstring states the
two invariants the rest of the file enforces: (a) *default args* of `eval_depth_poses.py`, i.e.
`--align sim3`, `--pose_reduce rmse`, `--depth_scale_align median`, `--num_cameras 1`,
`--pred_pose_type c2w`, `--gt_pose_type c2w` (its `parse_args`), and (b) one child process per scene.
"Mirrors cg_fuse_fwd_bwd.py's eval step" refers to the champion fusion script, whose eval step builds
the identical command line (`--pred_root`, `--gt_root`, `--output_csv`), applies the identical
"CSV exists and is non-empty" skip rule and the identical `returncode != 0 or empty CSV` failure rule,
but runs scenes sequentially inside a Slurm shard rather than through a `Pool`.

**Alternatives considered.** (i) Score with `eval_pipeline/pose_sim3_both.py`, the scorer used for
sweeps 1-3 and the "v0 FINAL" recipe in the design doc; (ii) go through the standard model-arm entry
point `eval_pipeline/arm_eval_for_run.sh`, which couples scoring to inference; (iii) import
`eval_depth_poses` in-process instead of spawning it.

**Why this choice.** Decision 0.3 fixed the eval scope to end at "full 4292 once frozen" and
decision 0.4 fixed that "harness scores ATE / RPE-trans / RPE-rot only" with AbsRel and d1
"INHERITED from augfull_lr1e5 depth via symlink"; both require the *harness* evaluator, not the
sweep scorer, so that the OpenCV rows and the model rows in the 4292 table are produced by the same
code path (the design doc's FULL HARNESS section names exactly this file as the scorer).
`eval_depth_poses.py` exposes only a CLI `main()` that parses `sys.argv` and writes files, which is
why it is a subprocess here and in `cg_fuse_fwd_bwd.py`. The design doc's runtime figure for the model
arm, "~70 ms/frame CUT3R on one H100 (incl. eval subprocess)", shows the model side also pays for
this subprocess.

### Lines 4-9: expected layout and usage

```
4: 
5: Expects <out_root>/<label>/preds/<scene>/{camera/*.npz, depth -> symlinked model depth}. Writes
6: <out_root>/<label>/eval/<scene>/eval_depth_pose_metrics.csv (skips scenes already scored).
7: 
8: Usage: opencv_vo_eval.py --out_root ... --label ... --scenes_root ... --scene_list ... [--workers 48]
9: """
```

**What it does.** Lines 4 and 7 are blank lines inside the docstring. Line 5 states the input
contract, which is exactly what `opencv_vo.py` produces: `camera/%06d.npz` with keys `pose` (c2w,
4x4, float32) and `intrinsics` for *every* frame, and `depth` as a symlink to the paired model run's
`preds/<scene>/depth` (created by `opencv_vo.py --depth_link`). A real scene directory from the
finetuned full run shows this: `camera/`, `depth -> .../augfull_lr1e5/preds/<scene>/depth`,
`diag.csv`, `summary.json`. Line 6 states the output path and the resume rule. Line 8 is the
command line; the four required flags map onto the two roots the evaluator needs (`--pred_root`
derived from `out_root/label`, `--gt_root` derived from `scenes_root`).

**Alternatives considered.** Writing the CSV next to the predictions (the evaluator's own default,
`<pred_root>/eval_depth_pose_metrics.csv`) versus a parallel `eval/` tree; copying depth versus
symlinking it; scoring poses only (`camera/` without any `depth/`).

**Why this choice.** The `preds/` + `eval/` split is the harness layout every model arm already uses
(the table builder reads `<out_root>/<label>/eval/<scene>/eval_depth_pose_metrics.csv`). The depth
symlink is decision 0.4: depth columns are inherited and must be "mark[ed] as inherited in tables";
the design doc's metric-floor section confirms "Depth columns identical by construction
(symlinked)". Scoring poses only is not an option because the evaluator raises
`FileNotFoundError` when it finds neither depth nor camera files, and because the table needs
AbsRel / d<1.25 columns (0.179 / 0.786 on the finetuned rows, 0.479 / 0.558 on the zero-shot rows,
by construction equal to the paired model's).

### Lines 10-14: imports

```
10: import argparse
11: import os
12: import subprocess
13: import sys
14: from multiprocessing import Pool
```

**What it does.** Only the standard library. `subprocess` runs the evaluator, `sys` supplies
`sys.executable`, `multiprocessing.Pool` provides process-level parallelism (on Linux the default
start method is `fork`, so each worker is a forked copy of this tiny parent; the heavy work happens in
the grandchildren spawned by `subprocess.run`). No `numpy`, `cv2` or `evo` is imported here: the
parent never touches a pose.

**Alternatives considered.** `concurrent.futures.ProcessPoolExecutor` or a thread pool (the work is
process-bound in the child anyway, so threads would also do); Slurm job arrays with one task per
scene; an in-process evaluator with `numpy` loaded in the parent.

**Why this choice.** Keeping the parent free of numeric imports means the 48 forked workers are
cheap and there is nothing for `fork` to corrupt (no BLAS thread pools in the parent). The design
doc's plumbing note "Run as a CPU job (login node has a 300 s per-process CPU cap)" is why this is a
Slurm CPU job with an in-job pool rather than something run interactively. No benchmark settled the
pool flavour; it is a plain engineering choice.

### Lines 15-18: locating the evaluator

```
15: 
16: HERE = os.path.dirname(os.path.abspath(__file__))
17: EVAL_SCRIPT = os.path.join(os.path.dirname(HERE), "eval_bundle", "bin", "eval_depth_poses.py")
18: 
```

**What it does.** Lines 15 and 18 are blank. `HERE` is the absolute directory of this file
(`.../my-da3/eval_pipeline`); `EVAL_SCRIPT` is its sibling tree `.../my-da3/eval_bundle/bin/eval_depth_poses.py`,
resolved relative to the repository rather than to the current working directory, so the script
works from any cwd (Slurm jobs start in the submission directory, which may not be the repo).

**Alternatives considered.** Passing the evaluator path as a flag (`cg_fuse_fwd_bwd.py` has
`--eval_script`); importing it as a module.

**Why this choice.** There is exactly one evaluator the harness recognises, and the point of the file
is to make it impossible to score the OpenCV rows with a different one; hard-wiring the path removes
a free parameter. This is a plain engineering choice with no design-doc ruling; the doc's "as few
free parameters as possible" purpose concerns the VO arm's algorithm, not this script. No benchmark
involved.

### Lines 19-21: the per-scene task

```
19: 
20: def one(task):
21:     scene, pred_dir, gt_root, eval_csv = task
```

**What it does.** Line 19 is blank. `one` is the function executed in the pool workers; it takes a
single 4-tuple (built at lines 41-42) so it can be handed to `imap_unordered`: the scene name (used
only for reporting), `pred_dir` = `<out_root>/<label>/preds/<scene>`, `gt_root` =
`<scenes_root>/<scene>/dense` (the directory holding `rgb/ cam/ depth/ outlier_mask/ sky_mask/`,
design doc "Dataset facts"), and the target CSV path. It returns a `(scene, status, message)` triple
in every branch, with `status` one of `"skip"`, `"missing"`, `"fail"`, `"ok"`.

**Alternatives considered.** A keyword-argument signature with `functools.partial`; raising
exceptions from the worker and catching them in the parent (the `cg_fuse_fwd_bwd.py` style).

**Why this choice.** Returning a status triple instead of raising keeps the pool alive when a scene
fails: an exception propagating out of `imap_unordered` would abort the whole 4292-scene run at the
first bad scene. Status strings are what decision 0.4's "failure reporting" needs at the scene
level (which scenes were scored, which were not), although note that decision 0.4's per-frame
failure diagnostics live in `opencv_vo.py`'s `diag.csv` / `summary.json`, not here.

### Lines 22-23: resume rule

```
22:     if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
23:         return scene, "skip", ""
```

**What it does.** If the per-scene CSV already exists and is non-empty, the scene is reported as
`skip` and no subprocess is started. This is what makes the script idempotent: re-launching after a
Slurm time-out only scores the scenes that are still missing.

**Alternatives considered.** No resume (always rescore); a `--force` flag; validating the CSV
content (e.g. that an `ALL/MEAN` row exists) rather than its size.

**Why this choice.** The rule is copied verbatim from `cg_fuse_fwd_bwd.py` (same
`isfile and getsize > 0` test, `cg_fuse_fwd_bwd.py` lines 328-329), so the two scoring paths resume
identically. The evaluator writes the CSV in a single `open("w")` after all metrics are computed
(`eval_depth_poses.py` line 899); only the `eval_metrics_readable.txt` sidecar (lines 905-906) and
summary prints follow, so a non-empty file is in practice a complete one. There is no `--force`: to rescore, delete the `eval/`
tree. Not benchmarked; engineering choice.

### Lines 24-25: missing-input guard

```
24:     if not os.path.isdir(os.path.join(pred_dir, "camera")) or not os.path.isdir(os.path.join(pred_dir, "depth")):
25:         return scene, "missing", "no camera/ or depth/ under preds"
```

**What it does.** Both `camera/` and `depth/` must exist under the scene's preds directory, else
the scene is `missing` and nothing is launched. `os.path.isdir` follows symlinks, so a `depth`
symlink whose target has been deleted counts as missing (this is the practical meaning of the
CLAUDE.md rule never to purge a run's `preds` that others symlink into). `camera/` is missing when
`opencv_vo.py` never finished the scene (it writes `summary.json` last and prints `ERROR <scene>` on
exceptions), and `depth/` is missing when `opencv_vo.py` was run without `--depth_link`.

**Alternatives considered.** Letting the evaluator fail on its own (it raises
`FileNotFoundError` only when *both* `depth/` and `camera/` are absent, and silently produces
NaN depth columns when only `depth/` is absent); scoring pose-only scenes and back-filling depth
later.

**Why this choice.** The evaluator's own guard is too weak for the inherited-depth design of
decision 0.4: a scene with `camera/` but no `depth/` would score "successfully" with NaN AbsRel /
d1, and `build_opencv_full_table.py`'s `nanmean` over scenes would then silently average fewer
scenes in the depth columns than in the pose columns, breaking the "identical by construction"
property the metric-floor analysis relies on. Requiring both directories up front turns that into
a counted, printed `missing`. The distinction between `missing` (inputs absent, this script's
fault or the VO run's) and `fail` (evaluator crashed) is the scene-level analogue of decision
0.4's "separates lost-tracking from drift".

### Lines 26-28: launch the evaluator

```
26:     os.makedirs(os.path.dirname(eval_csv), exist_ok=True)
27:     cmd = [sys.executable, EVAL_SCRIPT, "--pred_root", pred_dir, "--gt_root", gt_root, "--output_csv", eval_csv]
28:     r = subprocess.run(cmd, capture_output=True, text=True)
```

**What it does.** Creates `<out_root>/<label>/eval/<scene>/` (the evaluator would also do this, but
creating it here means a crash still leaves a directory to inspect). Builds the argument vector with
the *same interpreter as the parent* (`sys.executable`, so the child sees the same conda
environment and the `numpy` it needs; `evo`, `torch`, `cv2` and `scipy` are imported only inside the
`--eval_like_cut3r` code path, which is not used here, per the evaluator's `parse_args` help at
lines 106-120) and only three flags; everything else is the evaluator's
default. Meaning of each argument to `eval_depth_poses.py`:

- `--pred_root pred_dir`: the evaluator globs `pred_root/depth/*.npy` and `pred_root/camera/*.npz`,
  keeps files whose stem is all digits, and then **renumbers them by position** (`_split_stream_by_camera`
  assigns `local_idx` by counting in sorted order), which is the trap recorded in the design doc:
  "eval_depth_poses.py numbers frames by position, so every frame must get a pose". The predicted
  pose is read as c2w (`--pred_pose_type c2w` default), matching what `opencv_vo.py` writes after
  inverting `solvePnP`'s world-to-camera `(rvec, tvec)` (design doc plumbing note).
- `--gt_root gt_root`: the evaluator resolves the GT camera folder as `camera/` if present else
  `cam/` (`_resolve_cam_folder`); DROID scenes use `cam/*.npz` with key `pose` (c2w) and GT depth
  `depth/*.npy` with 0 = invalid (design doc "Dataset facts").
- `--output_csv eval_csv`: overrides the evaluator's default of writing next to the predictions.

With default arguments the evaluator then, per camera stream (one stream, `--num_cameras 1`):
intersects predicted and GT timesteps, aligns predicted camera centres to GT centres with a
Umeyama Sim(3) (`--align sim3`; rotation applied to the pose, `s*(R t)+t` to the centre), reports
per-frame `ate` = Euclidean distance of aligned centres (metres), `rpe_trans` / `rpe_rot` from
`inv(d_gt) @ d_pred` over consecutive common frames (metres / degrees), depth `absrel` and `a1`
after per-frame median scale alignment (`--depth_scale_align median`), and writes a `MEAN` row per
camera plus an `ALL/MEAN` row where the pose columns are RMSE-reduced (`--pose_reduce rmse`) and
the depth columns nanmean-reduced. `capture_output=True, text=True` buffers the child's stdout and
stderr as strings so 48 children do not interleave on the job log, and so line 30 can quote stderr.

**Alternatives considered.** `--eval_like_cut3r` (the evaluator's evo-based CUT3R-identical mode);
`--align se3` or `none`; `--pose_reduce mean`; `--num_cameras > 1` (interleaved streams).

**Why this choice.** "Default args" is the contract at line 2 and in `cg_fuse_fwd_bwd.py`: the
finetuned ("Regular CUT3R"), zero-shot and champion rows were all produced with the defaults, and
the OpenCV rows must be comparable cell-for-cell. Sim(3) alignment is also what the design was built
around from the start: "Scoring: Sim(3)-aligned ATE / RPE ... A global scale is free; scale DRIFT is
not" (Dataset facts), which is why decision 3.3 leaves scale "as given by the map" and decision 2.15
bothers with a scale hand-off across re-bootstraps. The design doc's metric-floor section is
computed by pushing trivial trajectories "through the SAME Sim(3) scorer": constant pose scores
0.171 m ATE / 0.0079 m RPE-t / 1.21 deg RPE-r, and every arm except the champion (0.0072) sits at that
RPE-trans floor, so with this scorer "RPE-rot and ATE are the informative columns".

### Lines 29-31: verdict for the scene

```
29:     if r.returncode != 0 or not (os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0):
30:         return scene, "fail", r.stderr[-400:]
31:     return scene, "ok", ""
```

**What it does.** A scene is `fail` if the evaluator exited non-zero **or** left no non-empty CSV
(guards against a zero exit with nothing written); the message is the last 400 characters of the
child's stderr, which for an uncaught Python exception is the tail of the traceback. Otherwise
`ok`. The evaluator's own soft failures (its `[eval_like_cut3r] ATE failed:` prints) are not
reachable with default args; with defaults the pose block either succeeds or raises.

**Alternatives considered.** Raise-and-catch per scene as `cg_fuse_fwd_bwd.py` does (the same
two-part test raises `RuntimeError` at lines 353-354, the per-scene `except Exception` at lines
361-364 counts it, appends `scene\trepr(e)` to `_failures_shard<id>.txt`, prints `FAIL <scene>`
inline and continues, and the shard exits non-zero at the end if any scene failed; this is the
"cg_fuse style" already named in the Lines 19-21 alternatives above); keeping full stderr;
retrying once.

**Why this choice.** Same two-part test as `cg_fuse_fwd_bwd.py` (`returncode != 0 or empty CSV`,
its lines 351-352). The stderr tail kept is 400 characters here versus 500 in `cg_fuse_fwd_bwd.py`
(line 354); no reason is recorded for the different constant. Decision 2.5 rules out silent retries at the frame level ("a failed PnP is
logged as a failed frame ... not retried"); the same spirit holds here: a failed scene is counted
and printed, never retried or hidden. Note the asymmetry with `skip`: if the evaluator wrote the
CSV and then died (non-zero exit after the write), this run reports `fail` but the next launch
reports `skip`.

### Lines 32-35: entry point and parser

```
32: 
33: 
34: def main():
35:     ap = argparse.ArgumentParser()
```

**What it does.** Two blank lines (32-33), then `main` starts with a bare `ArgumentParser` (no
`description`, so `--help` lists only the five flags; the usage line at line 8 lives in the module
docstring and is not surfaced by argparse).

**Alternatives considered.** A shared parser with `opencv_vo.py` so the two scripts cannot disagree
on `--out_root` / `--label`.

**Why this choice.** Convenience; the four shared flag names are identical by convention, not by
code. Not a benchmarked decision.

### Lines 36-39: arguments

```
36:     ap.add_argument("--out_root", required=True); ap.add_argument("--label", required=True)
37:     ap.add_argument("--scenes_root", required=True); ap.add_argument("--scene_list", required=True)
38:     ap.add_argument("--workers", type=int, default=48)
39:     a = ap.parse_args()
```

**What it does.** `--out_root` / `--label` locate `preds/` and `eval/` exactly as in `opencv_vo.py`
(`writes <out_root>/<label>/preds/<scene>/camera/`); `--scenes_root` is `$SCENES_ROOT` from
`eval_pipeline/mn5_paths.sh` (design doc "Dataset facts"); `--scene_list` is a text file with one
scene name per line (for the smoke scope, `eval_pipeline/cg_smoke_scenes_12.txt`). `--workers`
sets the pool size, default 48.

**Alternatives considered.** Deriving `--workers` from `os.cpu_count()` or
`$SLURM_CPUS_PER_TASK`; sharding by `--shard_id/--num_shards` as `opencv_vo.py` and
`cg_fuse_fwd_bwd.py` do.

**Why this choice.** 48 matches the CPU allocation the design doc records for the OpenCV full-harness
rows ("Slurm 45527508 / 45529336 (48 cores, ~16 min per row + scoring)"); it is a hard-coded default,
not a measured optimum, and must be overridden on a smaller allocation. Scene lists follow
decision 0.3: the 12 smoke scenes (all inside the 430 subset, shared with `augfull_lr1e5` and
`augfull_cg_fuse_g7`), then 430, then the full 4292.

### Lines 40-42: scene list and task tuples

```
40:     scenes = [l.strip() for l in open(a.scene_list) if l.strip()]
41:     tasks = [(s, os.path.join(a.out_root, a.label, "preds", s), os.path.join(a.scenes_root, s, "dense"),
42:               os.path.join(a.out_root, a.label, "eval", s, "eval_depth_pose_metrics.csv")) for s in scenes]
```

**What it does.** Reads the scene list, dropping blank lines (same idiom as `opencv_vo.py` line
`scenes = [l.strip() for l in open(args.scene_list) if l.strip()]`), and builds the 4-tuples that
`one` unpacks at line 21. Note the GT root is `<scenes_root>/<scene>/dense`, i.e. the directory that
contains `rgb/ cam/ depth/ outlier_mask/ sky_mask/`; the evaluator only reads `cam/` (or `camera/`)
and `depth/` from it. Building all tuples up front costs nothing (4292 short tuples) and lets
`len(tasks)` serve as the progress denominator at line 50.

**Alternatives considered.** Streaming scenes into the pool with a generator; discovering scenes by
listing `preds/` instead of taking a list.

**Why this choice.** An explicit list is the harness convention (every arm is scored on the same
enumerated set), and it is what makes decision 0.3's "compared on identical scenes" auditable.
Engineering choice, not benchmarked.

### Lines 43-46: the pool

```
43:     counts = {}
44:     with Pool(a.workers) as p:
45:         for i, (scene, st, msg) in enumerate(p.imap_unordered(one, tasks, chunksize=4)):
46:             counts[st] = counts.get(st, 0) + 1
```

**What it does.** `counts` accumulates the four status strings. `Pool(a.workers)` forks 48 workers;
`imap_unordered(one, tasks, chunksize=4)` hands tasks to workers in chunks of 4 and yields results
**in completion order**, not list order, so `i` is the number of finished scenes, not an index into
`tasks`. Chunking amortises the pickling round-trip (4292 is divisible by 4, so there is no partial chunk). The
`with` block joins the pool on exit. Each worker's `one` call
blocks in `subprocess.run`, so at most 48 evaluator processes run at once.

**Alternatives considered.** `p.map` (ordered, but blocks until all are done and holds every result);
`imap` (ordered streaming, but head-of-line blocking on a slow scene); `chunksize=1`.

**Why this choice.** Unordered streaming gives live progress and lets short scenes finish while a
1700-frame scene (design doc: "Sequence length 57 to ~1700 frames") is still being scored.
`chunksize=4` is a plain engineering constant with no benchmark behind it.

### Lines 47-50: progress reporting

```
47:             if st in ("fail", "missing") and counts[st] <= 5:
48:                 print(f"{st}: {scene}: {msg}", flush=True)
49:             if (i + 1) % 500 == 0:
50:                 print(f"  {i+1}/{len(tasks)} {counts}", flush=True)
```

**What it does.** Only the first five `fail` and the first five `missing` scenes are printed with
their message (the stderr tail from line 30, or the fixed string from line 25); later ones are
counted but silent. Every 500 completions a running tally is printed; 4292 is not a multiple of
500, so the last partial block of fewer than 500 scenes is reported only by the final tally at
line 51. `flush=True` makes the lines appear in the Slurm log immediately.

**Alternatives considered.** A per-scene failure log file (as `cg_fuse_fwd_bwd.py` writes); printing
every failure; `tqdm`.

**Why this choice.** Keeps the Slurm log short on a run where "missing" can legitimately be large
(e.g. a scene list scored before the VO run finished). Not a benchmarked decision; the per-scene
diagnostics that decision 0.4 asks for live in `opencv_vo.py`'s side CSV, not in this log.

### Lines 51-55: final tally and guard

```
51:     print(f"{a.label}: {counts}", flush=True)
52: 
53: 
54: if __name__ == "__main__":
55:     main()
```

**What it does.** Prints one summary line, e.g. `vo_full_ft: {'ok': ..., 'skip': ..., ...}`, which
is the only place the complete counts appear. Lines 52-53 are blank; 54-55 are the standard entry
guard; it is the standard protection for any module that uses `multiprocessing` (under the
`spawn` start method workers re-import the module and must not re-run `main`; under Linux's
default `fork` it is merely conventional). The process always exits
with status 0, regardless of how many scenes failed.

**Alternatives considered.** `sys.exit(1)` when `fail` or `missing` is non-zero, so a Slurm
dependency chain (`afterok`) could gate the table build on a clean score.

**Why this choice.** This is a divergence from `cg_fuse_fwd_bwd.py`, not a match: that script
prints its `DONE done=... skipped=... failed=...` tally and then `sys.exit(1)` if `failed` is
non-zero (lines 365-368); this script omits that guard and always exits 0, so the human must read
the tally. No reason is recorded for dropping the guard. A known limitation (below).

### Known limitations / honesty notes for this block

- **Exit status is always 0.** A run in which every scene is `fail` or `missing` still exits
  cleanly; the only signal is the printed tally. This diverges from `cg_fuse_fwd_bwd.py`, which
  exits 1 when any scene failed (lines 365-368), so a Slurm `afterok` dependency on this script
  cannot detect failed scenes. Downstream, `build_opencv_full_table.py` takes the `nanmean` over
  whichever scene CSVs exist and appends `(n=N)` to a row only when fewer scene CSVs than the scene
  list were found (line 78), so a partially scored row would be averaged over fewer scenes without
  an error from this script.
- **`skip` trusts file size, not content.** A CSV that exists and is non-empty is never rescored;
  the counterpart is that a scene whose evaluator wrote the CSV and then exited non-zero is `fail`
  on this run and `skip` on the next.
- **Depth columns are the paired model's, by construction.** The `depth/` requirement at line 24 is
  satisfied only by the symlink into the CUT3R run; AbsRel / d<1.25 in the OpenCV rows are those of
  `augfull_lr1e5` (finetuned rows) or the zero-shot run, exactly equal to the model rows
  (0.179 / 0.786 and 0.479 / 0.558 in the full-4292 table). This is decision 0.4 and honesty-audit
  item 10 ("depth columns would be the paired model's"), which is pending the user's ruling.
- **The scorer does not see failed frames.** Decision 3.2 holds the last pose on a failed frame and
  `opencv_vo.py` writes a pose for every frame, so this evaluator scores held poses like any
  other; the failed-frame percentage (audit item 12: it "excludes retro-filled frames") is only in
  `opencv_vo.py`'s `diag.csv` / `summary.json`. The accepted non-causal retro-fill (audit item 1,
  footnoted in the full table) is likewise invisible here; its measured effect on the 12 smoke
  scenes is +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r.
- **Positional renumbering.** `eval_depth_poses.py` renumbers files by sorted position; this is
  safe only because `opencv_vo.py` emits `camera/%06d.npz` for every frame. Scoring a subset of
  frames through this script would silently misalign predictions and GT (the harness note in the
  design doc: "every frame must get a pose").
- **Metric floors apply to everything this script produces.** With the default Sim(3) scorer,
  RPE-trans is saturated at the no-motion floor (0.0079 m for a constant pose versus 0.008 m for the
  finetuned and OpenCV rows), and the OpenCV rows' ATE (0.104 m finetuned+OpenCV, 0.120 m
  zero-shot+OpenCV) sits at the constant-velocity floor (0.125 m); the design doc concludes only
  RPE-rot and ATE are informative, and this script cannot change that.
- **`--workers 48` is an allocation-matched constant**, not a measured optimum, and each evaluator
  child may start its own numeric threads; oversubscription on smaller nodes was not measured.
- **Scorer mismatch with the sweeps.** The design doc's 12-scene tables (sweeps 1-3 and the v0
  FINAL recipe name `eval_pipeline/pose_sim3_both.py`; ATE / RPE-t in millimetres, means over
  12 scenes) were not produced by this script; only the FULL HARNESS table was. Do not compare
  the two sets of numbers as if they were the same evaluator.
