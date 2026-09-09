# OpenCV VO probes — the benchmarks that settled each design decision

These are the standalone benchmarks referenced by `OPENCV_VO_DESIGN.md`. Each one uses ground
truth **only to score** the options; nothing here feeds GT into the pipeline. They are archived
here exactly as they were run, so the numbers in the design log are reproducible.

| Script | Settles | Design-log section |
|---|---|---|
| `corr_bench.py` | Correspondence method: LK ± forward-backward, ORB, SIFT, DIS at corners, DIS on a grid, Farneback | decision 1.5 |
| `pnp_bench.py` | PnP solver: ITERATIVE / EPnP / P3P / AP3P / SQPnP, each ± `solvePnPRefineLM` | decision 2.4 |
| `tri_bench.py` | Triangulation acceptance filters, keyframe trigger, bootstrap-E threshold, static-track lever | decisions 2.11, 2.12, 2.10, 1.2 |
| `e_diag.py` | Why two-view translation direction is ~30° off (for the classical arm *and* for CUT3R) | "Two-view diagnostics" |

The `.sbatch` launchers are archived verbatim, including the `SC=` / `G=` staging paths under
`/gpfs/scratch/.../opencv_vo_probe/`, because `/scratch/tmp` is node-local on MN5 and compute nodes
cannot see it — everything had to be staged on GPFS. To re-run from this checkout, point those
variables at this directory instead:

```bash
G=/gpfs/home/koc/koc821022/my-da3/eval_pipeline/opencv_vo_probes
sbatch --export=ALL,G=$G $G/corr_bench.sbatch
```

Two constraints they encode, both learned the hard way and worth keeping:

- `cv2.setNumThreads(1)` plus `OMP_NUM_THREADS=1`: the login node enforces a 300 s per-process CPU
  cap, and multi-threaded OpenCV trips it in seconds.
- `--qos=acc_debug` allows only one job per user at a time; longer sweeps use `acc_ehpc`.

The line-by-line explanation of every script and launcher is in
[`docs/opencv_vo/README.md`](../../docs/opencv_vo/README.md), Part III.
