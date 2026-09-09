## `eval_pipeline/opencv_vo_probes/*.sbatch` — the six Slurm launchers (`corr_bench.sbatch` 1–20, `pnp_bench.sbatch` 1–20, `tri_kf.sbatch` 1–28, `e_diag.sbatch` 1–22, `sweep.sbatch` 1–38, `pf.sbatch` 1–34)

These six files are the batch jobs that produced the benchmark tables in `OPENCV_VO_DESIGN.md`. They sit at the very front of the pipeline, before any pose is written: the first four launch the standalone GT-scored probes (`corr_bench.py` for decision 1.5, `pnp_bench.py` for 2.4, `tri_bench.py` in keyframe-trigger mode for 2.12, `e_diag.py` for 2.10) that settled individual design decisions, and the last two launch the real pipeline `eval_pipeline/opencv_vo.py` as a sharded sweep over its CLI switches and then score every arm with `eval_pipeline/pose_sim3_both.py` (v0 sweep 1, and the inference-time-focal run). All six share one skeleton: a MareNostrum5 `acc_debug` GPU-partition allocation of one node, a conda activation, a single-thread BLAS/OpenMP environment on top of the `cv2.setNumThreads(1)` each probe already calls, one backgrounded Python process per scene (or per scene-shard) drawn from `eval_pipeline/cg_smoke_scenes_12.txt` (decision 0.3), a `wait`, and an `ALLDONE` sentinel.

Everything the jobs read or write lives on GPFS, never on the node-local `/scratch/tmp`: all outputs and logs go under `/gpfs/scratch/etur59/koc821022/…`, and all six read the scene list from `/gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt`. The *executed script* differs between the two groups. The four probe launchers do **not** point at the repository copies of the probe scripts: they execute the copies staged under `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/<probe>/`, the location the design doc records as "Script + per-scene JSON" for each benchmark (verified byte-identical to the repository copies by `diff`, re-checked 2026-09-09). The two pipeline launchers execute the repository copy `$WT/eval_pipeline/opencv_vo.py` from `/gpfs/home`. The Slurm job ids quoted in the design doc map onto the `--output` pattern below, e.g. `slurm_corr_bench_45403560.out`; for the four probe launchers that file contains exactly the one line `ALLDONE`, because every process writes its own per-scene log instead.

The `#SBATCH` header is explained in full once, for `corr_bench.sbatch`; for the other five files the identical lines are quoted (every line is covered) and only the differences are discussed.

---

### `corr_bench.sbatch` (lines 1–20) — correspondence-method benchmark, decision 1.5, Slurm job 45403560

```
1: #!/bin/bash
2: #SBATCH --account=etur59
3: #SBATCH --partition=acc
4: #SBATCH --qos=acc_debug
```

**What it does.** Line 1 makes the file a bash script; `sbatch` reads the `#SBATCH` comment directives from the top of it. Line 2 charges the job to the project account `etur59` (the same account that owns `/gpfs/scratch/etur59`, where all data lives). Line 3 selects the MareNostrum5 accelerated (GPU) partition `acc`. Line 4 selects the `acc_debug` quality-of-service, the short-turnaround debugging QoS of that partition. Per the cluster's QoS table (`sacctmgr show qos acc_debug`, queried 2026-09-08; not a number from the design doc) it carries `MaxWall 02:00:00`, `MaxJobsPU 1`, `MaxSubmitPU 1` and `Flags=DenyOnLimit`. `MaxSubmitPU 1` with `DenyOnLimit` means the *second* `sbatch` from the same user is **rejected at submission time** (`QOSMaxSubmitJobPerUserLimit`) rather than queued behind the first — so a user cannot even stack these launchers up and walk away.

**Alternatives considered.** Running the probe on the login node (no Slurm at all); the long-running `acc_ehpc` QoS used for training and full-harness runs; a CPU-only partition. The design doc's plumbing notes are explicit that the login node is not an option: "Run as a CPU job (login node has a 300 s per-process CPU cap)" (design doc, "Plumbing and day-one checks").

**Why this choice.** Workflow ruling recorded on 2026-09-05 (the "empirically verify options" rule): every design option is benchmarked "via Slurm on the acc_debug partition with everything staged on GPFS". `acc_debug` gives the fastest queue turnaround for jobs that run for minutes, and the probes finish well inside its wall-clock cap. The one-submission-at-a-time cap is a known nuisance (see limitations) but was accepted for the debug-cycle latency; it is also the reason every launcher below packs all of its work into a single job.

```
5: #SBATCH --nodes=1
6: #SBATCH --ntasks-per-node=1
7: #SBATCH --gres=gpu:1
8: #SBATCH --cpus-per-task=20
9: #SBATCH --time=00:30:00
```

**What it does.** One node, one Slurm task (the bash script itself is the task; nothing calls `srun`, so all parallelism comes from backgrounded processes inside that one task, all confined to the task's cgroup on that node). `--gres=gpu:1` requests one GPU. `--cpus-per-task=20` gives the task 20 CPU cores, which is where the twelve per-scene processes launched at line 17 run. `--time=00:30:00` is the wall-clock limit (30 min); the job is killed at that point.

**Alternatives considered.** A Slurm job array with one array element per scene (`--array=0-11`, one task each), `srun --ntasks=12` with one task per scene, or GNU `parallel`; a CPU partition with `--gres` omitted; more or fewer cores.

**Why this choice.** Simplicity of a debug job: one allocation, one log, one sentinel, and no per-element scheduling under a QoS that accepts one submission per user at a time. The GPU is requested only because `acc_debug` is a QoS of the GPU partition; `corr_bench.py` imports `cv2`, `numpy` and `json` and never touches the GPU — it sits idle for the whole job (see limitations). Twelve scenes, twelve single-threaded processes: 20 cores is enough with headroom. 30 min is a generous bound for a run whose per-pair method timings are 2–19 ms (design doc, correspondence table, `ms` column) over 4243 frames.

```
10: #SBATCH --job-name=corr_bench
11: #SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_corr_bench_%j.out
```

**What it does.** Names the job in `squeue`, and routes the job's stdout+stderr to a file on GPFS whose name embeds the Slurm job id (`%j`). This is the file the design doc's "Slurm job 45403560" refers to (`slurm_corr_bench_45403560.out`). The `logs/` directory must already exist; Slurm does not create it, and a missing `logs/` makes the job fail at launch, before the script runs at all.

**Alternatives considered.** The Slurm default (`slurm-%j.out` in the submission directory); a name without the job id.

**Why this choice.** All harness logs live in `$OUT/logs` (`$OUT = /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval`, the same root the pipeline sweeps and `pose_sim3_both.py` use), and the job id in the name is what lets the design doc cite a run unambiguously.

```
12: source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
13: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

**What it does.** Line 12 loads the system Miniconda's shell hooks and activates the `cuteanything` environment (the project's Python/OpenCV/PyTorch environment, found under `~/.conda/envs/cuteanything`); the `&&` skips activation if the hook file cannot be sourced. There is no `set -e`, so a failed activation would not abort the script — the subsequent `python` calls would simply fail in their per-scene logs. Line 13 pins every threaded numeric backend the process may pull in (OpenMP, OpenBLAS, MKL — the ones numpy and scipy dispatch to) to one thread. OpenCV's own thread pool is pinned separately, inside each probe, by `cv2.setNumThreads(1)` (line 14 of `corr_bench.py`).

**Alternatives considered.** Let each process use all cores (default numpy/OpenCV behaviour) and run scenes sequentially; set only `cv2.setNumThreads`; `taskset`/`--cpu-bind` affinities.

**Why this choice.** Twelve processes each allowed to size their thread pool to the 20 CPUs visible in the task cgroup would oversubscribe the node 12×; one thread per process times twelve processes uses the allocation exactly. It also makes the `ms` timings reported in the correspondence table single-core numbers, which is what the design doc quotes ("LK + fwd-bwd check … 4 ms"). The environment variables are needed on top of `cv2.setNumThreads(1)` because that call only governs OpenCV's pool, not numpy's BLAS. This is the "single-threaded OpenCV" clause of the 2026-09-05 workflow rule.

```
14: SC=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/corr_bench
15: R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
```

**What it does.** `SC` is the staging directory of this probe on GPFS: it holds the copy of `corr_bench.py` that is actually executed, and the `corr/` output subdirectory. `R` is the 4292-scene test split root (`$SCENES_ROOT/<scene>/dense/{rgb,cam,depth,outlier_mask,sky_mask}` in the design doc's dataset facts); the string is identical to the `SCENES_ROOT` default in `eval_pipeline/mn5_paths.sh`, hard-coded here because the launcher does not source that helper.

**Alternatives considered.** Point `SC` at the repository checkout (`$WT/eval_pipeline/opencv_vo_probes`); write outputs to node-local `/scratch/tmp`; source `mn5_paths.sh` for `$SCENES_ROOT`.

**Why this choice.** GPFS staging is the rule: `/scratch/tmp` is node-local, so anything written there is invisible from the login node once the job ends, and the tables in the design doc are built by reading the per-scene JSONs from a login-node session. Keeping script and outputs in one GPFS directory is also what makes the design doc's provenance line ("Script + per-scene JSON: …/opencv_vo_probe/corr_bench/") a single path. The cost is that the repository copy is not what runs (see limitations).

```
16: for S in $(cat /gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt); do
17:   python $SC/corr_bench.py $R/$S $SC/corr/$S.json 1 4 > $SC/corr/$S.log 2>&1 &
18: done
```

**What it does.** Reads the 12-scene smoke list (one scene name per line, e.g. `AUTOLab+0d4edc83+2023-10-21-19h-11m-38s`; the names contain `+` and `-` but no whitespace, so the unquoted `$(cat …)` word-splitting is safe) and, for each scene, launches one Python process in the background (`&`). The probe's positional interface (lines 16–17 of `corr_bench.py`: `scene, out = sys.argv[1], sys.argv[2]; gaps = [int(g) for g in sys.argv[3:]] or [1, 4]`) receives: `argv[1]` = the scene directory (the script appends `dense/` itself), `argv[2]` = the output JSON path `corr/<scene>.json`, `argv[3:]` = the frame gaps to benchmark, `1` and `4`. Both stdout and stderr go to `corr/<scene>.log`. The `corr/` directory must pre-exist: neither bash's redirection nor the script's `json.dump` creates it, and if it is missing bash cannot open the redirection target, so the process never starts (see limitations). Inside, for every pair `(i, i+gap)` stepping `i` by 2 (`corr_bench.py:98`), each of the seven methods (`lk_fb`, `lk_nofb`, `orb_ratio`, `sift_ratio`, `dis_corners`, `dis_grid`, `farneback_corners`, `corr_bench.py:93-94`) is scored against the GT flow induced by GT depth and GT relative pose (native 320x180 frames, calibrated `K` from `cam/000000.npz`). It also records an essential-matrix rotation error, computed with `findEssentialMat(..., method=cv2.RANSAC, prob=0.999, threshold=1.0)` on the subset of pairs with GT rotation > 0.5 deg **and** at least 8 correspondences (`corr_bench.py:114-118`) — note that 1 px is neither the pipeline's bootstrap threshold (0.5 px) nor the PnP threshold of decision 2.7 (2 px), which matters when the `rotErr` column is quoted as evidence.

**Alternatives considered.** Gaps other than 1 and 4 (the default in the script is also `[1, 4]`); more scenes (the 430 subset); one process looping over all scenes.

**Why this choice.** Decision 0.3: the smoke set is `cg_smoke_scenes_12.txt` because all 12 scenes are inside the 430 subset and both `augfull_lr1e5` and `augfull_cg_fuse_g7` have preds and eval on them, so classical and model arms are compared on identical scenes. Gap 1 is the frame-to-frame tracking regime (per-step motion median 9 mm / 1.3 deg) and gap 4 is the keyframe-scale regime; the design doc reports both columns. Per-scene processes give twelve-way parallelism and twelve independent logs, so one crashing scene does not take the others down.

```
19: wait
20: echo ALLDONE
```

**What it does.** `wait` blocks until every backgrounded process has exited (it does not propagate their exit codes). `echo ALLDONE` writes the sentinel to the Slurm `.out` file; for this job that file consists of that single line.

**Alternatives considered.** `wait -n` loops with exit-code checks; `set -e`/`pipefail`; a trailing aggregation step.

**Why this choice.** The Bash tool driving these sessions has a 10-minute foreground cap, so jobs are submitted with `sbatch` and polled; the `ALLDONE` line is the cheap, grep-able "the job reached the end of the script" marker. Per-scene success is judged from the JSON files, not from exit codes — `ALLDONE` prints even when every process failed to start (see limitations). The verdict this job produced (design doc, correspondence benchmark, 12 smoke scenes, 4243 frames): LK + fwd-bwd check 166/88 gap-1 correspondences/correct, rotErr 0.88 deg, 4 ms; DIS on an 8-px grid 880/478 and 0.75 deg at gap 1, 1.88 (p90 6.9) at gap 4; ORB 3.53 and Farneback 3.66 at gap 4 rejected; "Removing static tracks helps as much as changing method" — the origin of decision 1.5 (LK + FB at OpenCV defaults) and of the lever in decision 1.2.

---

### `pnp_bench.sbatch` (lines 1–20) — PnP-variant benchmark, decision 2.4, Slurm job 45407881

```
1: #!/bin/bash
2: #SBATCH --account=etur59
3: #SBATCH --partition=acc
4: #SBATCH --qos=acc_debug
5: #SBATCH --nodes=1
6: #SBATCH --ntasks-per-node=1
```

**What it does.** Identical to `corr_bench.sbatch` lines 1–6: bash, account `etur59`, GPU partition `acc`, debug QoS, one node, one task.

**Alternatives considered / Why this choice.** As above; the launcher is a clone of `corr_bench.sbatch` under the same workflow rule.

```
7: #SBATCH --gres=gpu:1
8: #SBATCH --cpus-per-task=20
9: #SBATCH --time=00:30:00
10: #SBATCH --job-name=pnp_bench
11: #SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_pnp_bench_%j.out
```

**What it does.** One (idle) GPU, 20 cores for twelve single-threaded processes, 30-minute limit, job name `pnp_bench`, log `slurm_pnp_bench_<jobid>.out` (job 45407881 in the design doc).

**Why this choice.** Same sizing as the correspondence benchmark: twelve scenes, and the per-pair solver cost is 0.3–2.3 ms (design doc, PnP table, `ms` column), so the job is short.

```
12: source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
13: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

**What it does.** Environment activation and one-thread pinning, as in `corr_bench.sbatch` lines 12–13; `pnp_bench.py` line 23 additionally calls `cv2.setNumThreads(1)`.

**Why this choice.** As above; the single-core `ms` column of the PnP table depends on it.

```
14: G=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/pnp_bench
15: R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
```

**What it does.** `G` is this probe's GPFS staging directory (the executed `pnp_bench.py` and its `out/` subdirectory; the design doc's "Script + JSON: …/opencv_vo_probe/pnp_bench/"); `R` is the scene root, as before. Only the variable name differs from `corr_bench.sbatch` (`G` instead of `SC`).

**Why this choice.** GPFS staging rule, as above.

```
16: for S in $(cat /gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt); do
17:   python $G/pnp_bench.py $R/$S $G/out/$S.json 1 4 > $G/out/$S.log 2>&1 &
18: done
```

**What it does.** One background process per smoke scene; `pnp_bench.py` (lines 25–26: `scene, out = sys.argv[1], sys.argv[2]; gaps = … or [1, 4]`) gets the scene directory, the output JSON `out/<scene>.json`, and gaps 1 and 4; stdout+stderr to `out/<scene>.log` (directory must pre-exist). Inside, the same LK+FB frontend as decision 1.5 produces 2D tracks from frame `i` to `i+gap`; 3D points come from GT depth on frame `i` (oracle 3D, native 320x180, calibrated `K`); each of ten variants — `solvePnPRansac` with flag in {ITERATIVE, EPNP, P3P, AP3P, SQPNP} (`FLAGS`, lines 46–47), each with and without `solvePnPRefineLM` on the inlier set (`VARIANTS`, line 48), all at `reprojectionError=2.0`, `confidence=0.999`, `iterationsCount=1000` (lines 53-54) — is scored against the GT relative pose under two conditions, `clean` and `outliers` (static tracks with no valid depth assigned a plausible wrong 3D point, i.e. identity-voting outliers). `solvePnPRansac` returns a world-to-camera `(rvec, tvec)`; the script composes it into `T_ji` for scoring.

**Alternatives considered.** Only ITERATIVE (the OpenCV default flag); scoring on the pipeline's own triangulated map instead of oracle 3D; other RANSAC settings.

**Why this choice.** Decision 2.4 asked which solver to hard-code; oracle 3D isolates the solver from map quality. The fixed RANSAC settings are decisions 2.7 (2 px: "8 px = 2.2 deg at f~210, admits gripper tracks; 1 px is at the LK noise floor"; the OpenCV default is 8 px) and 2.8 (0.999 / 1000, versus the OpenCV default 0.99 / 100; "1-2 ms per frame; removes the iteration cap as a variable").

```
19: wait
20: echo ALLDONE
```

**What it does.** Wait for the twelve processes, then print the sentinel (the only line in `slurm_pnp_bench_45407881.out`).

**Why this choice.** As above. Outcome, as the design doc's PnP-variant benchmark section states it: "Paired diffs vs ITERATIVE+LM: median |diff| < 0.01 deg for every variant", zero failures on 2118 pairs, and identity-voting outliers costing only 1.10 -> 1.14 deg at gap 4. The per-variant table itself: ITERATIVE 0.44 deg (p90 1.72) / 3.9 mm at gap 1 and 1.10 (6.86) / 8.3 mm at gap 4; SQPnP nominally best at both gaps, 0.37 at gap 1 and 1.06 at gap 4. Decision 2.4's cell records that as "SQPnP nominally best by 0.04 deg": 0.04 is the *gap-4* margin over ITERATIVE (1.06 vs 1.10), while at gap 1 the same margin is 0.07 (0.37 vs 0.44). Across all ten variants the rotation spread is 0.12 deg at gap 4 (1.06–1.18) and 0.07 deg at gap 1 (0.37–0.44). The tighter phrase "all 10 variants within 0.05 deg / 2 mm of each other" belongs to decision 2.4's summary cell in the decision table, not to this benchmark section; the paired-difference statement is the one the table supports. Hence decision 2.4 (`solvePnPRansac(ITERATIVE)` + `solvePnPRefineLM`, the LM step "a no-op after ITERATIVE-RANSAC but kept so the refine step exists for the P3P/SQPnP swaps"), 2.5 (no initial guess, no automatic fallback solver), and 2.6 (plain RANSAC, "0.04 deg penalty"). The same run established the absolute floor: even with oracle 3D, per-frame PnP rotation error is 0.44 deg median against a GT median step of 1.3 deg, and translation 3.9 mm against 9 mm — "the map arm cannot beat it".

---

### `tri_kf.sbatch` (lines 1–28) — keyframe-trigger sweep with `tri_bench.py`, decision 2.12, Slurm job 45443937

```
1: #!/bin/bash
2: #SBATCH --account=etur59
3: #SBATCH --partition=acc
4: #SBATCH --qos=acc_debug
5: #SBATCH --nodes=1
6: #SBATCH --ntasks-per-node=1
```

**What it does.** Same six header lines as the two launchers above.

**Why this choice.** As above.

```
7: #SBATCH --gres=gpu:1
8: #SBATCH --cpus-per-task=48
9: #SBATCH --time=00:50:00
10: #SBATCH --job-name=tri_kf
11: #SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_tri_kf_%j.out
```

**What it does.** One idle GPU, 48 cores, a 50-minute limit, job name `tri_kf`, log `slurm_tri_kf_<jobid>.out` (job 45443937 in the design doc's keyframe-trigger table).

**Alternatives considered.** 20 cores as in the first two launchers; one job per trigger setting (six jobs).

**Why this choice.** The 50-minute limit is what the six sequential configurations (lines 22–27) plus the heavier inner loop of `tri_bench.py` buy: chained LK over up to 32 frames, triangulation, five acceptance filters, and a PnP solve per filter under both the E-matrix and the GT pose — roughly twice the wall clock of the pair benchmarks. The 48 cores do **not** follow from that: the `run` function ends in `wait` (line 20), so at most twelve single-threaded processes are ever alive, and twelve processes cannot use more than twelve cores. The allocation is simply over-provisioned — the same `--cpus-per-task=48` as `e_diag.sbatch` and `sweep.sbatch`, which genuinely do run 36 and 108 concurrent processes — and no ruling or measurement in the design doc justifies 48 here. Six sequential groups inside one job are, by contrast, forced: under `MaxSubmitPU 1` with `DenyOnLimit`, six separate submissions would be refused outright.

```
12: source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
13: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KOFF=4 LEVER=0 E_THR=0.5
```

**What it does.** Activation as before, plus three probe-specific environment variables read by `tri_bench.py` lines 34–36: `KOFF=4` sets the PnP test frame to `k = j + 4` (the script's own default is `2*GAP` — line 34, `int(os.environ.get("KOFF", "0")) or 2 * GAP`; the inline comment beside it, "default: k = j + GAP", is stale and contradicts the expression it annotates), `LEVER=0` keeps the static-track lever OFF (line 35: `LEVER = os.environ.get("LEVER", "0") == "1"`), and `E_THR=0.5` sets the bootstrap essential-matrix RANSAC threshold to 0.5 px (line 36; the script default is 2.0). These exports are inherited by every python process launched below. `tri_bench.py` line 30 pins OpenCV to one thread.

**Alternatives considered.** The script defaults (`KOFF = 2*GAP`, lever off, `E_THR = 2.0`), i.e. what the earlier triangulation-acceptance run used; lever ON; `E_THR` 1 or 2 px.

**Why this choice.** By the time this sweep ran, decision 2.10 had been revised to 0.5 px on the strength of the E-diagnostic and the map-quality-vs-threshold table (design doc: at gap 4, `bad`/PnP-rot 0.55/4.19 at 2 px, 0.52/3.66 at 1 px, 0.51/3.02 at 0.5 px), so the keyframe sweep was run at the new threshold ("E 0.5 px" in the table caption). `KOFF=4` makes every trigger family comparable at a common horizon ("PnP test at keyframe+4"). Lever OFF matches the caption ("lever OFF") and the earlier triangulation benchmark ("Static tracks kept (lever OFF)"), so the trigger is the only variable; the lever's own effect is measured elsewhere (decision 1.2). Note this is the probe's lever, not the pipeline's: in `eval_pipeline/opencv_vo.py` the static-track lever has been ON by default since 2026-09-05 (`--no_lever` disables it).

```
14: G=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/tri_bench
15: R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
```

**What it does.** GPFS staging directory of the triangulation probe (the executed `tri_bench.py`; outputs go to its `out_kf/` subdirectory) and the scene root.

**Why this choice.** GPFS staging rule; `out_kf/` is one of several output subdirectories of this probe (`out/`, `out_chain/`, `out_ethr/`, `out_gap/`, `out_k4/`, `out_kf/` on scratch), one per question the same script answered under a different sibling launcher (`tri_bench.sbatch`, `tri_chain.sbatch`, `tri_ethr.sbatch`, `tri_gap.sbatch`, `tri_k4.sbatch`, `tri_kf.sbatch`).

```
16: run () {  # mode val tag
17:   for S in $(cat /gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt); do
18:     KF_MODE=$1 KF_VAL=$2 python $G/tri_bench.py $R/$S $G/out_kf/$S.$3.json 4 > $G/out_kf/$S.$3.log 2>&1 &
19:   done
20:   wait
21: }
```

**What it does.** Defines a shell function `run MODE VAL TAG`. For each smoke scene it launches `tri_bench.py` in the background with two per-invocation environment variables prefixed to the command: `KF_MODE` (script line 37: `stride | parallax | ratio`, how the second keyframe `j` is chosen) and `KF_VAL` (line 38: for `parallax`, the median track displacement in px since `i` that fires the trigger; for `ratio`, the surviving-track fraction below which it fires). Positional arguments: scene directory, output JSON `out_kf/<scene>.<tag>.json`, and `4` = `GAP` (script line 33). In `parallax`/`ratio` mode `GAP` no longer sets `j` (script lines 91–98: the walk advances one LK hop at a time over at most 32 frames until the trigger fires, and abandons the triplet if fewer than 20 tracks survive the walk, line 95, or if the trigger never fires, line 98 `if j is None: continue`); it only bounds the triplet loop (`range(0, n - GAP - KOFF, 2)`), and `KOFF=4` from line 13 places the PnP test frame at `k = j + 4`. `wait` inside the function means each `run` call finishes all twelve scenes before the next call starts, so at most twelve processes are alive at once. Output dir `out_kf/` must pre-exist.

**Alternatives considered.** Launch all six settings at once (72 processes on 48 cores); pass the mode/value as positional arguments instead of environment variables; a separate script per trigger family.

**Why this choice.** Environment variables were the least-invasive way to add sweep knobs to a probe originally written for a single question (2.11); the function keeps the launcher to six one-line calls. Sequential groups keep the process count at twelve — well inside the (over-provisioned) 48-core allocation — while the chained-LK inner loop runs.

```
22: run parallax 5 par5
23: run parallax 10 par10
24: run parallax 20 par20
```

**What it does.** Three sweeps of the parallax trigger: declare keyframe `j` at the first frame where the median displacement of the surviving tracks since `i` reaches 5, 10 or 20 px; JSONs tagged `par5`, `par10`, `par20`.

**Alternatives considered.** Decision 2.12's option list: fixed stride (2/4/8), median parallax since last KF >= P px (5/10/20), tracked-inlier ratio < r (0.9/0.8/0.7).

**Why this choice.** These are exactly the three parallax values in the decision's option list; 10 px was the original bootstrap threshold of decision 2.9 ("10 px ~ 3 typical frames ~ 25 mm baseline at 0.5 m"), 5 and 20 bracket it. Result (design doc, keyframe-trigger sweep): parallax 5 px -> KF gap median 2 (p10–p90 1–8), n_acc 91, bad 0.56, PnP rot 2.96 (p90 10.6), fail 11.9%; 10 px -> gap 4, 3.23 (11.9); 20 px -> gap 7, 4.25 (16.0). Decision 2.12 chose 5 px ("Parallax 5 gives median gap 2 (2.96 deg) and never fires on a paused camera"), and decision 2.9's bootstrap threshold was lowered from 10 to 5 px to share the parameter — the 5 px parallax trigger of the shipped pipeline.

```
25: run ratio 0.9 ratio0.9
26: run ratio 0.8 ratio0.8
27: run ratio 0.7 ratio0.7
28: echo ALLDONE
```

**What it does.** Three sweeps of the tracked-ratio trigger: declare `j` at the first frame where the fraction of tracks surviving since `i` drops below 0.9, 0.8 or 0.7 (`ok1.mean() < KF_VAL`, `tri_bench.py` line 96, the same expression whose `parallax` branch tests the median displacement; line 97 is where `j = f + 1` is set and the walk breaks); tags `ratio0.9` … `ratio0.7`. Line 28 prints the sentinel after the last group's `wait` (the only line in `slurm_tri_kf_45443937.out`).

**Alternatives considered.** The ORB-SLAM-style "tracked-ratio" family is the standard alternative to a parallax trigger; the values are the three from the decision's option list.

**Why this choice.** Results (same table): ratio 0.9 -> gap 2 (1–8), PnP rot 2.91 (10.4), fail 7.4%; 0.8 -> gap 5, 3.68 (14.4); 0.7 -> gap 7, 4.78 (19.1). The design doc's reading: "Every family improves monotonically as keyframes get closer; all adaptive triggers at their tightest setting converge to a median gap of 2." Parallax was preferred over ratio at equal median gap because it "never fires on a paused camera", whereas a ratio trigger fires on track loss regardless of motion and a stride trigger "would triangulate at ~zero baseline and pass cheir+rep" on the 25% of 2-frame windows with < 2 px motion. Note that the `stride 2/4/8` rows of that table were not produced by this file: they are the `E_THR=0.5` arm of the E-threshold sweep, run by the sibling launcher `tri_ethr.sbatch` (Slurm 45443823, outputs in `out_ethr/`, `KOFF=4 LEVER=0`, `for ETHR in 0.5 1.0; for GAP in 2 4 8`), which exists only in the scratch staging directory. That identification is exact, not inferred: the design doc's map-quality-vs-threshold table at 0.5 px (gap 2 `bad`/PnP-rot 0.54/2.62, gap 4 0.51/3.02, gap 8 0.48/4.49) is numerically the same run as the keyframe table's stride rows (bad 0.54/0.51/0.48, PnP rot 2.62/3.02/4.49). A third sibling, `tri_gap.sbatch`, is a *different* question entirely — a long-baseline sweep at gaps 8/16/32 with no `E_THR` and no `KOFF` export (so E RANSAC 2 px and `k = j + 2*GAP`), writing `out_gap/` — and contributes nothing to the keyframe table. This mixed provenance is why the doc warns that "Stride rows are conditioned on the 2 px motion gate … parallax/ratio rows fire on their own rule, so their populations differ slightly."

---

### `e_diag.sbatch` (lines 1–22) — two-view essential-matrix diagnostics, decision 2.10, Slurm job 45443683

```
1: #!/bin/bash
2: #SBATCH --account=etur59
3: #SBATCH --partition=acc
4: #SBATCH --qos=acc_debug
5: #SBATCH --nodes=1
6: #SBATCH --ntasks-per-node=1
```

**What it does.** Same header as the other launchers.

**Why this choice.** As above.

```
7: #SBATCH --gres=gpu:1
8: #SBATCH --cpus-per-task=48
9: #SBATCH --time=00:45:00
10: #SBATCH --job-name=e_diag
11: #SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_e_diag_%j.out
```

**What it does.** One idle GPU, 48 cores, 45-minute limit, job name `e_diag`, log `slurm_e_diag_<jobid>.out` (job 45443683; the design doc cites "Slurm 45443683 / 45443704" for this section — the second id is a follow-up launcher, `e_diag2.sbatch`, that lives only on scratch and re-runs the same `e_diag.py` at the same gaps into `out2/`).

**Why this choice.** Lines 16–20 launch three gaps × twelve scenes = 36 processes concurrently, so 48 cores are needed to keep them single-threaded without time-slicing; `e_diag.py` is also the only probe that pulls in scipy (`least_squares` for the Sampson refinement, `Rotation` for the parametrisation, lines 24-25), run once per gated pair. This is the one probe launcher whose core count is actually earned.

```
12: source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
13: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

**What it does.** Activation and one-thread pinning; `e_diag.py` line 23 pins OpenCV. The pinning matters more here than elsewhere because scipy's `least_squares` (imported at line 24 of the probe) would otherwise thread through BLAS in each of 36 processes.

**Why this choice.** As above.

```
14: G=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/e_diag
15: R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
```

**What it does.** GPFS staging directory of the diagnostic (executed `e_diag.py`, `out/` subdirectory; the design doc's "Scripts + JSON: …/opencv_vo_probe/e_diag/") and the scene root.

**Why this choice.** GPFS staging rule.

```
16: for GAP in 4 8 16; do
17:   for S in $(cat /gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt); do
18:     python $G/e_diag.py $R/$S $G/out/$S.gap$GAP.json $GAP > $G/out/$S.gap$GAP.log 2>&1 &
19:   done
20: done
```

**What it does.** Nested loops over three frame gaps and the twelve smoke scenes, launching all 36 processes in the background without an intermediate `wait`. `e_diag.py` (line 27–28: `scene, out = sys.argv[1], sys.argv[2]; GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 8`) receives the scene directory, output JSON `out/<scene>.gap<GAP>.json`, and the gap; stdout+stderr to the matching `.log` (`out/` must pre-exist). Its only inputs are `dense/{rgb, cam, depth, outlier_mask}` — it never reads a model prediction. For each pair `(i, i+GAP)`, `i` stepping by 2 (`e_diag.py:85`), it chains LK tracks and gates hard before scoring anything (`e_diag.py:88-98`): at least 20 surviving tracks, median track displacement >= 2 px (the parallax gate), the static lever in its absolute form (keep only tracks that moved >= 1 px) with at least 20 survivors, and a GT baseline of at least 2 mm. On what survives that gate it scores eleven two-view pose estimates against the GT relative pose in rotation error and translation-direction error (deg) — the full list is declared at `e_diag.py:84` and recorded at `117-136`:

- `e2` = `findEssentialMat` RANSAC 2 px (the pipeline at that time) + `recoverPose`, `e1` = 1 px, `e05` = 0.5 px, `magsac` = `cv2.USAC_MAGSAC` at 2 px, all on every surviving track;
- `e2_rel` = `e2` after a *relative* static lever (drop tracks moving < 20% of the median displacement, needs >= 20 survivors) — the design doc's "relative static lever" row;
- `e2_oracle` / `e2_oracle_rel` = `e2` restricted to tracks with valid, non-outlier-masked GT depth, with and without the relative lever (diagnostic-only gripper proxy);
- `synth1` / `synth0` = correspondences replaced by GT flow (GT depth + GT pose) with and without 1 px Gaussian noise, both estimated with plain RANSAC at 2 px (`e_diag.py:129-130`). These are **not** run on the same population as `e2`: `e_diag.py:127-130` restricts them to `a[zv]`, the tracks with valid, non-outlier-masked GT depth, and requires `zv.sum() >= 20`. That distinction matters because the design doc records that "GT depth is defined ON the gripper in many scenes", so `zv` is not a clean scene-geometry subset;
- `refine` = Sampson-distance least squares seeded from the `e2` inliers, and `gtR_t` = translation direction by SVD with the rotation fixed to GT.

On the OpenCV call itself (`e_diag.py:53-56`): `findEssentialMat(a, b, K, method, prob=0.999, threshold=thr)` takes **pixel** coordinates plus `K`, and `threshold` is likewise in **pixels** — the maximum Sampson distance to the epipolar line. OpenCV normalises the points by `K` internally and divides the threshold by the focal length so the pixel meaning is preserved, which is exactly the conversion the pipeline performs by hand once each view carries its own `K`: `opencv_vo.py:363-366` normalises each view by its own `K` (line 363), then passes those points with an identity `K` and `threshold=E_THRESH / (0.5 * (Ka[0, 0] + Kb[0, 0]))` (lines 365-366). That pixel reading is the only one under which this section's 2 px / 1 px / 0.5 px arms mean anything. `recoverPose` then picks one of the four `(R, t)` decompositions of `E` by a cheirality test over the points flagged in the inlier mask and returns `R, t` mapping the first camera into the second with `|t| = 1`, which is why translation is scored as a direction only.

**Alternatives considered.** The gap values: the earlier probes used gaps 1 and 4; the script default is 8; the design doc's decision 2.9 reasoning quotes "10 px ~ 3 typical frames"; here 4/8/16 span the keyframe-scale to long-baseline regime.

**Why this choice.** The question this run answered (script docstring: "Why is the essential-matrix translation direction ~35 deg off?") needed the trend with baseline, not one gap. Findings *from this job* (design doc, two-view diagnostics, ~1780 pairs per gap; rot / tdir at gap 4, 8, 16): E RANSAC 2 px 2.06 / 31.6, 3.80 / 31.0, 8.22 / 41.0; 1 px 1.64 / 24.3, 3.05 / 25.2, 7.32 / 35.9; 0.5 px 1.23 / 18.1, 2.88 / 23.5, 8.11 / 36.8; MAGSAC 2 px 1.34 / 22.9; GT flow + 1 px noise 1.43 / 13.8 -> 1.95 / 4.2; GT flow, no noise 0.01 / 0.0 (conventions verified); relative static lever 1.96 / 32.5. The two model rows in that same design-doc table — finetuned CUT3R 2.22 / 31.7, 3.76 / 29.0, 6.25 / 25.8, and champion 1.99 / 30.5, 3.33 / 27.1, 5.37 / 23.5 — were **added to the table from a separate model-pose comparison** ("all pairs, no gating", per the doc's own parenthetical); this launcher cannot have produced them, because `e_diag.py` never opens a preds root. Conclusions the launcher's three gaps made possible, once the model rows were placed beside them: the ~30 deg translation-direction error "is NOT specific to classical VO: the model has the same 30-40 deg error vs GT"; "E at 0.5 px BEATS the model at gaps 4 and 8 … and loses at gap 16"; "real tracks get WORSE with gap because chained LK drift grows with chain length. Keyframes must be close (gap 2-4), not far." This revised decision 2.10 (RANSAC threshold 2 -> 0.5 px, the bootstrap threshold the shipped pipeline uses) and motivated the tight keyframe trigger of 2.12 and the `E_THR=0.5` in `tri_kf.sbatch`.

```
21: wait
22: echo ALLDONE
```

**What it does.** Wait for all 36 processes, then print the sentinel (the only line in `slurm_e_diag_45443683.out`).

**Why this choice.** As above. One caveat from the same doc section applies to the JSONs this job wrote: "The gtR_t rows in the JSON are INVALID (sign-ambiguous linear solver); ignore them."

---

### `sweep.sbatch` (lines 1–38) — v0 pipeline sweep 1 over the open decisions, Slurm job 45453490

```
1: #!/bin/bash
2: #SBATCH --account=etur59
3: #SBATCH --partition=acc
4: #SBATCH --qos=acc_debug
5: #SBATCH --nodes=1
6: #SBATCH --ntasks-per-node=1
```

**What it does.** Same header as the probe launchers; this job runs the real pipeline, not a probe.

**Why this choice.** As above: the pipeline is pure CPU (`opencv_vo.py` is OpenCV + numpy), so it runs under the same debug allocation as the probes.

```
7: #SBATCH --gres=gpu:1
8: #SBATCH --cpus-per-task=48
9: #SBATCH --time=01:00:00
10: #SBATCH --job-name=vo_sweep
11: #SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_vo_sweep_%j.out
```

**What it does.** One idle GPU, 48 cores, a 60-minute limit, job name `vo_sweep`, log `slurm_vo_sweep_<jobid>.out` (job 45453490). Unlike the probe jobs, this `.out` file is not a bare sentinel: it also carries the scorer's validation block and its printed per-label summary (line 36). That printout is a **4-decimal digest in metres**, not the source of the design doc's tables — see the discussion at lines 35–38 below.

**Alternatives considered.** One job per variant; `acc_ehpc` for a longer limit.

**Why this choice.** Lines 28–34 start nine variants × twelve shards = 108 processes on 48 cores: deliberately oversubscribed, because the per-scene runtime spread is large (23 s/scene for the defaults, 104 and 189 s/scene for the two BA variants — design doc, sweep-1 table, `s/scene` column) and the kernel time-slicing the surplus is cheaper than nine sequential submissions under a QoS that refuses the second one outright (`MaxSubmitPU 1`, `DenyOnLimit`). One hour covers the slowest variant plus scoring.

```
12: source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
13: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

**What it does.** Activation and one-thread pinning, now for 108 concurrent pipeline processes and, later, the 12-worker scorer.

**Why this choice.** As above, and it matters most here. Without the pinning each of the 108 processes would size its OpenMP/BLAS pool to the CPUs visible in the task cgroup — `--cpus-per-task=48` — so the node would see up to 108 × 48 threads on 48 cores instead of 108 single-threaded processes: roughly a 48× thread multiplier, on top of the 2.25× process oversubscription that is deliberate. (The `corr_bench.sbatch` case at 12 processes on 20 cores is the same model at 12×.)

```
14: WT=/gpfs/home/koc/koc821022/my-da3
15: OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval
16: R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
17: L=$WT/eval_pipeline/cg_smoke_scenes_12.txt
```

**What it does.** `WT` is the repository checkout (worktree) — here, unlike the probes, the script that runs *is* the repository copy `$WT/eval_pipeline/opencv_vo.py`. `OUT` is the harness output root: every arm, model or classical, lives at `$OUT/<label>/preds/<scene>/camera/%06d.npz`, and the scorer reads the same tree. `R` is the scene root and `L` the smoke list; both are passed as CLI arguments below instead of being iterated in bash.

**Alternatives considered.** Sourcing `mn5_paths.sh` (which defines `OUT_ROOT`, `SCENES_ROOT`, and validates `WT`); a separate output root for classical arms.

**Why this choice.** Writing classical arms into the same `$OUT` tree as `augfull_lr1e5` and `augfull_cg_fuse_g7` is what lets one scorer invocation (line 36) score classical and model arms on identical scenes (decision 0.3) and what the later full-harness path relies on (design doc, plumbing: "Output per scene: `camera/%06d.npz` with `pose` (c2w, 4x4) and `intrinsics` … Score with eval_pipeline/pose_sim3_both.py").

```
18: declare -A V
19: V[vo_v0]=""
20: V[vo_np_every]="--new_points every"
21: V[vo_cull]="--cull outlier"
22: V[vo_mi10]="--min_inliers 10"
23: V[vo_mi30]="--min_inliers 30"
```

**What it does.** Declares a bash associative array `V` mapping an output label to the extra CLI flags of that variant. `vo_v0` = no flags = the `opencv_vo.py` defaults at the time (decided choices hard-coded; every open decision at its default switch). `--new_points every` (decision 2.13 switch, default `kf`) triangulates new map points at every frame instead of keyframe-to-keyframe. `--cull outlier` (2.14, default `none`) drops a map point after `--cull_hits` (default 3) consecutive PnP-outlier hits. `--min_inliers 10` / `30` (3.1, code default 20 at `opencv_vo.py:505`) sets the PnP inlier count below which the frame is a failure.

**Alternatives considered.** Exactly the option lists of decisions 2.13 (KF-to-KF only vs every frame), 2.14 (none; drop unseen for N KFs; drop after k outlier hits) and 3.1 (inlier count 10 / 20 / 30; ratio). The "unseen for N KFs" cull and the ratio floor were not run.

**Why this choice.** Design doc, sweep 1 (means over 12 scenes, ATE mm / RPE-t mm / RPE-r deg): `vo_v0` 122.9 / 9.65 / 1.129 with 37.1% failed frames, 23 reboots, 63 median inliers; `vo_np_every` 126.2 / 10.40 / 1.402 — worse on all three, so 2.13 stays KF-only; `vo_cull` 123.4 / 8.18 / 1.113 with failed frames 18.1% ("PnP failures 26% -> 5.6%, RPE-t 9.65 -> 8.18, 2x faster; ATE unchanged") — adopted as 2.14; `vo_mi10` 123.5 / 10.24 / 1.795 ("bad PnP accepted"); `vo_mi30` 123.3 / 9.09 / 1.063 — 30 adopted in 3.1 after sweep 2 confirmed it (0.861 vs 0.923 RPE-r against 20).

```
24: V[vo_fail_cv]="--fail_policy cv"
25: V[vo_ba5]="--refine ba --ba_window 5"
26: V[vo_ba10]="--refine ba --ba_window 10"
27: V[vo_lever]="--lever"
```

**What it does.** `--fail_policy cv` (decision 3.2, default `hold`) extrapolates a failed frame with constant velocity instead of holding the last pose. `--refine ba --ba_window N` (3.4, default `none`) enables the scipy sliding-window bundle adjustment over the last N keyframes. `--lever` (1.2) turns on the `reject_static_tracks` lever: on frames whose median track displacement is >= 2 px, drop tracks that moved < 1 px before any geometry.

**Alternatives considered.** Decision 3.2's options (hold last pose; constant velocity; drop frame — "drop" is impossible because "eval_depth_poses.py numbers frames by position, so every frame must get a pose"); 3.4's (none; sliding-window BA; pose graph — pose graph not run); 1.2's mask options (fixed polygon, Otsu temporal-variance mask, GT outlier mask), all rejected in the probe stage.

**Why this choice.** Sweep 1: `vo_fail_cv` 121.9 / 8.47 / 1.375 vs hold 122.9 / 9.65 / 1.129 ("mixed; rotation worse") — 3.2 keeps hold + flag; `vo_ba5` 128.8 / 8.49 / 1.068 and `vo_ba10` 130.7 / 9.04 / 1.175 at 104 / 189 s per scene ("5-9x runtime") — 3.4 keeps none; `vo_lever` 116.4 / 7.93 / 1.009 with 26.8% failed frames — better on every metric, which is why the lever default was flipped to ON on 2026-09-05 (user decision; `--no_lever` disables). Important consequence for reproduction: since that flip, `opencv_vo.py` line 509 has `default=True` for the lever, so re-running this file today would make `vo_v0` and `vo_lever` identical and would not reproduce the sweep-1 table (see limitations).

```
28: for lbl in "${!V[@]}"; do
29:   rm -rf $OUT/$lbl
30:   for sh in $(seq 0 11); do
```

**What it does.** `"${!V[@]}"` expands to the keys (labels) of the array in bash's hash order — unspecified, which is why the scorer's printout lists `vo_mi10` first. For each label it first deletes `$OUT/<label>` recursively (unconditionally, no confirmation), then opens a loop over twelve shard ids `0..11`.

**Alternatives considered.** Keeping old outputs and relying on the pipeline's own skip; a fixed label order (a plain indexed array); a bash loop over scenes as in the probes.

**Why this choice.** `opencv_vo.py` line 526 skips any scene whose `summary.json` already exists, so without the `rm -rf` a stale label would be silently kept and a changed flag set would never be re-run; a clean sweep needs the delete. Twelve shards because the smoke list has twelve scenes (one scene per shard, see the next group).

```
31:     python $WT/eval_pipeline/opencv_vo.py --scenes_root $R --scene_list $L --preds_root $OUT/augfull_lr1e5/preds \
32:       --out_root $OUT --label $lbl --shard_id $sh --num_shards 12 ${V[$lbl]} > $OUT/logs/vo_sweep_${lbl}_$sh.log 2>&1 &
33:   done
34: done
```

**What it does.** Launches one shard of the pipeline in the background per iteration. `opencv_vo.py` line 522 selects `scenes[shard_id::num_shards]`, so with twelve scenes each shard is exactly one scene; line 526 skips any scene whose `summary.json` already exists, which is what the `rm -rf` guards against (a stale label would otherwise be silently kept). `--preds_root $OUT/augfull_lr1e5/preds` is the model output root the pipeline reads the focal from: with no `--focal` flag the default `model` applies (line 512), i.e. the per-scene median of the finetuned model's per-frame focals with the principal point at the image centre, one `K` for the whole scene (decision 0.1b as originally decided, and the retroactive choice the campaign later abandoned). `${V[$lbl]}` is expanded unquoted so its words become separate arguments. Per-shard stdout+stderr go to `$OUT/logs/vo_sweep_<label>_<shard>.log`; each shard prints one JSON summary line per scene and writes the same object to `summary.json`. Its full field list is `opencv_vo.py:486-488`: `scene`, `frames`, `failed_frames`, `reboots`, `keyframes`, `median_inliers`, `map_points`, `seconds`, `focal` (the per-scene median of the per-frame focals actually used), `focal_min`, `focal_max`, and `selfcal_pairs`. Alongside it each scene gets `$OUT/<label>/preds/<scene>/{camera/%06d.npz, diag.csv}` — the diagnostics of decision 0.4. Nothing waits between labels, so all 108 processes coexist.

**Alternatives considered.** A single unsharded process per label (`--num_shards 1`, sequential scenes); a scene loop in bash as in the probes; keeping old outputs and relying on the skip; passing `--focal 203` (fixed nominal) or `--focal gt`; adding `--depth_link` so the arm inherits depth.

**Why this choice.** Sharding by scene inside the pipeline (rather than a bash loop) is the same mechanism the full-4292 run uses with `--num_shards` = number of cores, so the smoke launcher exercises the production path. `rm -rf` makes every sweep a clean run. The finetuned model's focal is decision 0.1b ("closed-loop rule (model output is not privileged)"); the zero-shot checkpoint's focal was already known to be unusable (229-570 px, 1.5x median, scene-inconsistent). `--depth_link` is unnecessary here because the smoke scorer at line 36 is pose-only; depth columns are inherited only when building the full harness table (decision 0.4).

```
35: wait
36: python $WT/eval_pipeline/pose_sim3_both.py --out_root $OUT --scenes_root $R --scene_list $L --workers 12 \
37:   --labels "${!V[@]}" augfull_lr1e5 augfull_cg_fuse_g7
38: echo ALLDONE
```

**What it does.** After all 108 pipeline processes exit, the pose-only scorer runs once over every label: the nine sweep labels plus the two model arms `augfull_lr1e5` (finetuned CUT3R) and `augfull_cg_fuse_g7` (the champion). `pose_sim3_both.py` loads only `camera/*.npz` c2w poses per scene (never depth), Umeyama-aligns each predicted trajectory to GT with scale (Sim(3)), and computes per-frame ATE, consecutive-pair RPE-trans and RPE-rot under both per-scene aggregations (nanmean and nan-RMSE); `--workers 12` is the multiprocessing pool size (default 48), one per scene. It produces two artefacts, and the distinction matters for provenance:

- **the per-label CSV** `$OUT/<label>/sim3_pose_both.csv` — one row per scene, full float precision, columns `scene, ate_mean, rpe_trans_mean, rpe_rot_mean, ate_rmse, rpe_trans_rmse, rpe_rot_rmse` (`pose_sim3_both.py:244-245`). This is where the design doc's smoke tables come from: the doc quotes the `[mean]` aggregation converted to mm at three significant digits, and only the CSV carries that many digits. For `vo_v0` the scene-mean over the CSV is `0.122885 / 0.009649 / 1.128524`, tabulated as **122.9 / 9.65 / 1.129**.
- **the printed digest** in the Slurm `.out`, a `PER-LABEL sim3 pose metrics (mean | rmse)` block formatted `%.4f` in metres/degrees (`pose_sim3_both.py:254`). For `vo_v0` it reads `ate=0.1229 rpe_t=0.0096 rpe_r=1.1285`. Note that `rpe_t=0.0096` is 9.6 mm, *not* the table's 9.65 — the print is a rounded digest, useful for reading the ATE and RPE-rot columns at a glance and for confirming the job ran, but it cannot be the source of the RPE-t column. The same holds for the other rows (`vo_fail_cv` prints 0.0085, CSV mean 0.008471 -> 8.47; `augfull_lr1e5` prints 0.0065, CSV 0.006537 -> 6.54).

The scorer also tries to validate its per-timestep values elementwise against any existing `<label>/eval_sim3/<scene>/eval_depth_pose_metrics.csv`; in this job every label printed `no eval_sim3 CSVs to validate against`. Line 38 prints the sentinel.

**Alternatives considered.** Scoring with the full harness (`eval_depth_poses.py` via `opencv_vo_eval.py`, which the 4292-scene run later used); scoring each label separately; `--workers 48`.

**Why this choice.** The fast pose-only scorer is what the design doc names for the arm ("Scoring: Sim(3)-aligned ATE / RPE (eval_pipeline/pose_sim3_both.py). A global scale is free; scale DRIFT is not"), and putting the two model arms in the same call is how the sweep-1 table got its reference rows: finetuned 72.3 / 6.54 / 0.960, champion 62.8 / 6.02 / 0.860 on the same twelve scenes. Twelve workers match twelve scene-tasks. The doc's reading of this job: "Per-scene ATE is worse than the finetuned model on every scene, including the one scene with zero failures (74 vs 56 mm)", and the failure anatomy (37% failed frames = 26% PnP < 20 inliers with a live map, 11% waiting for (re)bootstrap parallax, 0.5% map starved; the "death spiral" after a PnP failure) that led to sweeps 2 and 3.

---

### `pf.sbatch` (lines 1–34) — inference-time (per-frame / causal) focal on the frozen v0, Slurm job 45485470

```
1: #!/bin/bash
2: #SBATCH --account=etur59
3: #SBATCH --partition=acc
4: #SBATCH --qos=acc_debug
5: #SBATCH --nodes=1
6: #SBATCH --ntasks-per-node=1
```

**What it does.** Same header as `sweep.sbatch`.

**Why this choice.** As above.

```
7: #SBATCH --gres=gpu:1
8: #SBATCH --cpus-per-task=48
9: #SBATCH --time=01:00:00
10: #SBATCH --job-name=vo_pf
11: #SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_vo_pf_%j.out
```

**What it does.** One idle GPU, 48 cores, 60-minute limit, job name `vo_pf`, log `slurm_vo_pf_<jobid>.out` (job 45485470 in the design doc's "Inference-time (causal) focal" section), which again carries the scorer's validation block and its 4-decimal summary.

**Why this choice.** Four variants × twelve shards = 48 processes, exactly the core count; the limit is inherited from `sweep.sbatch` although the frozen configuration (no BA) is far cheaper.

```
12: source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
13: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
14: WT=/gpfs/home/koc/koc821022/my-da3
15: OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval
16: R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
17: L=$WT/eval_pipeline/cg_smoke_scenes_12.txt
```

**What it does.** Identical to `sweep.sbatch` lines 12–17: environment, one-thread pinning, worktree (again the repository copy of `opencv_vo.py` is what runs), harness output root, scene root, smoke list.

**Why this choice.** As above.

```
18: declare -A V
19: B="--cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff"
```

**What it does.** Declares the label-to-flags array and a base flag string `B` shared by all four variants. `B` is the frozen v0 FINAL configuration (design doc, "v0 FINAL configuration (frozen 2026-09-05)"): `--cull outlier` (2.14), `--reboot_after_fails 3` (3.2b: re-bootstrap after three consecutive PnP failures rather than waiting for map starvation), `--min_inliers 30` (3.1, the inlier floor of the shipped arm), `--scale_handoff` (2.15: at a re-bootstrap keep the old tracks and match the new segment's scale to the old points' depth through >= 10 shared tracks, 0.2 < s < 5). The lever is not listed because it is ON by default in the code since 2026-09-05 (`opencv_vo.py` line 509); `--pnp_thresh` is not listed because its default is already the decided 2 px (line 511).

**Alternatives considered.** Each flag's alternatives were measured in sweeps 1–3: `--reboot_after_fails 10` (sweep 2: 122.6 / 7.92 / 1.020 vs 119.9 / 7.99 / 1.009 for 3, "rb10 no better"); `--min_inliers 20` (sweep 3: 116.3 / 7.28 / 0.927 vs 111.9 / 6.93 / 0.871); no hand-off (sweep 3: 116.8 / 7.07 / 0.861); `--pnp_thresh 1` / `3` (sweep 2: 120.4 / 7.71 / 0.866 and 124.6 / 8.07 / 1.084, so 2.7 stayed at 2 px).

**Why this choice.** The point of this job is to vary one thing — the focal source — on top of the frozen configuration, per the 2026-09-06 user constraint that "OpenCV at frame t may only use what CUT3R has produced up to t; retroactive per-scene medians are not an inference-time method". The hand-off's own evidence (sweep 3): ATE 116.8 -> 111.9 mean, 117 -> 98 median; fired on 12 of 46 reboots.

```
20: V[vo6_ft_perframe]="$B --focal perframe:$OUT/augfull_lr1e5/preds"
21: V[vo6_ft_causal]="$B --focal causal:$OUT/augfull_lr1e5/preds"
22: V[vo6_zs_perframe]="$B --focal perframe:$OUT/cut3r_zeroshot/preds"
23: V[vo6_zs_causal]="$B --focal causal:$OUT/cut3r_zeroshot/preds"
```

**What it does.** Four arms = {finetuned `augfull_lr1e5`, zero-shot `cut3r_zeroshot`} × {`perframe`, `causal`}. In `opencv_vo.py` lines 265–277, `--focal perframe:<root>` reads, for every frame `t`, `intrinsics[0, 0]` from `<root>/<scene>/camera/%06d.npz` and builds a per-frame `K = [[f, 0, W/2], [0, f, H/2], [0, 0, 1]]` (principal point at the centre of the 320x192 cover frame); `causal:<root>` replaces `f_t` by the running median of `f_0..f_t`. Every geometric step then carries frame-specific `K`s (bootstrap E in per-view normalised coordinates with the pixel threshold divided by the mean focal, KF-to-KF triangulation with each keyframe's `K`, PnP with frame `t`'s `K`). The focal in those npz files is not a network head: it is derived after inference by `estimate_focal_knowing_depth` (Weiszfeld fit of `f = (u-cx)*z/x` over the predicted self-view point map), upstream `demo.py`'s convention.

**Alternatives considered.** Decision 0.1b's option list: calibrated `K` from `cam/*.npz` (`--focal gt`, diagnostic only), one nominal dataset focal (`--focal 203`), the model's per-scene median (`--focal model`, the retroactive default), self-calibration (`--focal selfcal`, "exists but is not recommended": per-scene values 132–510 px against a true ~203).

**Why this choice.** Results of this job (design doc, inference-time focal, focal range / ATE / RPE-t / RPE-r / ATE median / failed% / reboots): `vo6_ft_perframe` 205-221 px, 110.4 / 7.53 / 0.858 / 91.5 / 19.1% / 51; `vo6_ft_causal` 207-217, 113.8 / 6.85 / 0.844 / 98.9 / 19.6% / 51; `vo6_zs_perframe` 207-819, 124.8 / 10.33 / 1.059 / 109.4 / 14.8% / 63; `vo6_zs_causal` 213-599, 114.7 / 7.92 / 0.891 / 97.8 / 17.8% / 55. The "focal range (px)" column is not a scorer output: it is `focal_min`-`focal_max` from each scene's `summary.json` (`opencv_vo.py:486-488`), pooled over the twelve scenes — which is also the only place the zero-shot checkpoint's 819 px outlier is visible. Reading: "with the finetuned checkpoint the causal per-frame focal reproduces the retroactive numbers (110.4/7.53/0.858 vs 111.9/6.93/0.871)", so decision 0.1b was revised on 2026-09-06 to `perframe:` as the reported configuration, and the paired rows (each checkpoint with its own per-frame focal) became the basis of the full-4292 table. The per-frame variant, not the running median, was chosen as the report row because it uses strictly the frame's own inference output.

```
24: for lbl in "${!V[@]}"; do
25:   rm -rf $OUT/$lbl
26:   for sh in $(seq 0 11); do
```

**What it does.** Same label loop, unconditional delete of `$OUT/<label>`, and twelve-shard loop as `sweep.sbatch` lines 28–30.

**Alternatives considered / Why this choice.** As for `sweep.sbatch`: the delete defeats the `summary.json` skip in `opencv_vo.py` line 526; twelve shards = twelve scenes.

```
27:     python $WT/eval_pipeline/opencv_vo.py --scenes_root $R --scene_list $L --preds_root $OUT/augfull_lr1e5/preds \
28:       --out_root $OUT --label $lbl --shard_id $sh --num_shards 12 ${V[$lbl]} > $OUT/logs/vo_pf_${lbl}_$sh.log 2>&1 &
29:   done
30: done
```

**What it does.** Same per-shard launch as `sweep.sbatch` lines 31–34 with `vo_pf_` log prefixes. Note that `--preds_root $OUT/augfull_lr1e5/preds` is passed even for the two zero-shot arms: with `perframe:`/`causal:` the focal comes from the root named in `--focal`, and `preds_scene_dir` is only consumed by `model_K` in the `selfcal` fallback and the non-inference-time branch (`opencv_vo.py` lines 264 and 279), so the argument is inert here — it is a required CLI parameter, not a leak of finetuned information into the zero-shot rows. `rm -rf` again forces a clean run past the `summary.json` skip. Retro-fill of pre-bootstrap frames is active (no `--no_retro_fill`).

**Alternatives considered.** Making `--preds_root` optional; running the zero-shot arms with `--no_retro_fill`.

**Why this choice.** Consistency with the v0 FINAL command line in the design doc, which passes `--preds_root $OUT/augfull_lr1e5/preds` and `--focal perframe:…`. Retro-fill is audit item 1, retained by explicit user ruling on 2026-09-07 with a footnote clause; its ablation on this same configuration is +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r (110.4 / 7.5 / 0.858 with vs 110.6 / 7.7 / 0.881 without).

```
31: wait
32: python $WT/eval_pipeline/pose_sim3_both.py --out_root $OUT --scenes_root $R --scene_list $L --workers 12 \
33:   --labels "${!V[@]}" vo3_best_ho vo4_fnominal cut3r_zeroshot augfull_lr1e5 augfull_cg_fuse_g7
34: echo ALLDONE
```

**What it does.** Waits for the 48 pipeline processes, then scores the four new labels together with five reference labels already on disk: `vo3_best_ho` (v0 FINAL with the retroactive per-scene-median focal, the previous report row), `vo4_fnominal` (v0 FINAL with the fixed nominal focal `--focal 203`, the strictly model-free row), `cut3r_zeroshot` (the pretrained `cut3r_512_dpt_4_64.pth` run fresh on the 12 scenes), `augfull_lr1e5` and `augfull_cg_fuse_g7`. Same scorer semantics as `sweep.sbatch` line 36 — per-label `sim3_pose_both.csv` files with full precision (the source of the doc's mm figures) plus the 4-decimal printed digest; every label again reported `no eval_sim3 CSVs to validate against`. Line 34 prints the sentinel.

**Alternatives considered.** Scoring only the four new labels; using the full harness scorer.

**Why this choice.** Putting the reference rows in the same call yields the design doc's comparison table in one printout: `vo3_best_ho` 111.9 / 6.93 / 0.871 (retroactive; "no longer the reported configuration"), `vo4_fnominal` 109.1 / 7.04 / 0.869, `cut3r_zeroshot` 118.0 / 10.12 / 1.506, finetuned 72.3 / 6.54 / 0.960, champion 62.8 / 6.02 / 0.860. The doc's conclusion from this job: "The OpenCV arm beats each checkpoint on RPE-rot, ties/loses on ATE, and never beats the finetuned model on ATE"; the table builder `eval_pipeline/build_opencv_vo_table.py` "now reports the inference-time rows".

---

### Known limitations / honesty notes for the launchers

- **Test scenes throughout (audit items 7 and 8).** All six launchers run on the twelve reported smoke scenes, which are test scenes; the four probe launchers score with GT depth and GT pose (`dense/depth`, `dense/cam`), and the two sweep launchers are model selection on the reported scenes. The design doc records the user's disposition: item 7 "is addressed by reporting on all 4292 scenes with the configuration frozen"; item 8 is listed as pending the user's decision.
- **`sweep.sbatch` is no longer reproducible as written.** It produced sweep 1 with the lever OFF as the default and `--lever` as a variant; since the 2026-09-05 flip (`opencv_vo.py` line 509, `default=True`), `vo_v0` and `vo_lever` would be identical and `vo_v0` would no longer be the 122.9 / 9.65 / 1.129 row. Its outputs also use the retroactive per-scene-median focal (`--focal` unset), which the doc lists as audit item 13, "stale per-scene-median rows on disk".
- **The repository copies are not what ran, for the probes.** The four probe launchers execute `$SC`/`$G` copies on GPFS, not `$WT/eval_pipeline/opencv_vo_probes/*.py`; editing the repository copy changes nothing until it is re-staged. Verified byte-identical on 2026-09-08. The two pipeline launchers are the exception: they run `$WT/eval_pipeline/opencv_vo.py` straight from the repository.
- **Partial coverage of the doc's tables.** `tri_kf.sbatch` yields only the six parallax/ratio rows of the keyframe-trigger table; the three `stride 2/4/8` rows are the `E_THR=0.5` arm of `tri_ethr.sbatch` (Slurm 45443823, `out_ethr/`, scratch only) reused in that table — the identical `bad`/PnP-rot pairs 0.54/2.62, 0.51/3.02, 0.48/4.49 appear in both. A third sibling, `tri_gap.sbatch`, answers a different question (gaps 8/16/32 at E 2 px, `out_gap/`) and appears in neither table. `e_diag.sbatch` is the first of three E-diagnostic jobs (45443683); the follow-ups 45443704 and 45443759 used `e_diag2.sbatch` / `e_diag3.*`, which are not in the repository, and the model relative-pose rows printed in the same design-doc table came from a separate model-pose comparison, not from `e_diag.py` at all. The keyframe table's stride and adaptive rows have slightly different populations (doc note).
- **Zero-shot arms had no configuration search (audit item 11).** `pf.sbatch`'s `B` was tuned with the finetuned focal (sweeps 1–3); the `vo6_zs_*` rows only swap the focal source.
- **Reference rows are not like-for-like (audit item 9).** In `pf.sbatch` line 33, `augfull_lr1e5` is a causal forward pass while `augfull_cg_fuse_g7` is a confidence-gated forward x backward fusion.
- **Retro-fill is non-causal (audit item 1, retained by explicit user ruling with a footnote).** Neither sweep launcher passes `--no_retro_fill`; pre-bootstrap frames are re-posed against the bootstrap map, accepted at >= 4 inliers rather than the live 30, and counted as not failed (audit items 2 and 12). Measured effect on the `pf.sbatch` configuration: +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r. It is the one non-causal step in the reported configuration.
- **No error handling, and the failure mode is loud but unattended.** None of the files uses `set -e`; `wait` discards exit codes; `ALLDONE` only means the script reached its last line. If an *output* directory (`corr/`, `out/`, `out_kf/`, `out_ethr/`, …) is missing, bash cannot open the redirection target, so it writes a `…: No such file or directory` line to the **Slurm `.out`** and never execs python — there is no per-scene log for the error to land in. This is on record: `slurm_corr_bench_45403549.out` — an earlier submission of this same launcher whose `$SC` pointed at a session scratchpad under the node-local `/scratch/tmp`, a path that does not exist on the compute node — contains exactly twelve such lines (one per scene, naming `.../scratchpad/corr/<scene>.log`) followed by `ALLDONE`. That job is also the empirical case for the GPFS-staging rule: the same launcher with `$SC` on `/gpfs/scratch` (45403560) printed only `ALLDONE` and wrote all twelve JSONs. A missing `logs/` directory fails differently and earlier: Slurm cannot open the `--output` path and the job dies before the script runs. Either way the sentinel is not evidence of success — per-scene success must be checked from the JSON / `summary.json` files.
- **`rm -rf $OUT/$lbl` is unconditional.** It is safe only because the labels are defined a few lines above; a label colliding with a model arm (`augfull_lr1e5`, `augfull_cg_fuse_g7`, `cut3r_zeroshot`) would delete that arm's predictions.
- **The GPU is idle.** All six jobs request `--gres=gpu:1` because `acc_debug` belongs to the GPU partition; none of the launched Python processes uses it. The design doc's runtime statement for the arm is CPU-only ("OpenCV 30 ms/frame on one CPU core").
- **`acc_debug` accepts one submission per user.** `MaxJobsPU 1` *and* `MaxSubmitPU 1` with `Flags=DenyOnLimit` (cluster QoS table, not the design doc), so a second `sbatch` is refused at submission rather than queued. That is why every launcher packs its whole sweep into one job: the six sequential `run` groups in `tri_kf.sbatch` and the 108-process oversubscription in `sweep.sbatch` are workarounds for that cap, not performance choices.
- **`tri_kf.sbatch` is over-provisioned.** Its `run` function's `wait` caps concurrency at twelve single-threaded processes, so 36 of the 48 requested cores are never used. The uniform `--cpus-per-task=48` is copied from `e_diag.sbatch` / `sweep.sbatch`, where 36 and 108 concurrent processes make it real; no measurement in the design doc justifies it here.
- **Scorer validation did not fire.** In both sweep jobs `pose_sim3_both.py` found no `eval_sim3` CSVs for any label, so its elementwise check against `eval_depth_poses.py` was not exercised here; the full-4292 numbers were produced by `eval_depth_poses.py` itself via `opencv_vo_eval.py`.
- **Units and precision.** The scorer stores metres and degrees at full precision in `sim3_pose_both.csv` and prints the same values rounded to four decimals in the Slurm `.out`; the design doc's smoke tables are in mm / mm / deg read from the CSV, and the 4292 table is in metres. Reading a three-digit mm figure off the printed digest does not work (`rpe_t=0.0096` is 9.6, the table says 9.65).
