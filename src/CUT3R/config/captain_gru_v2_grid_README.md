# captain_gru_v2 lever grid — 12 overfit configs (A1-A4 × G0-G2)

Every config is the exact `captain_gru_v2.yaml` overfit recipe (single wrist_test
episode, 64 views, batch 4, 50 epochs, TBPTT chunk 4, aux `PoseGRULoss` weight 1.0,
`pose_gru_lr_scale` 100, hidden 128) differing ONLY in the four lever keys + `exp_name`.
Launch one arm: `sbatch train_captain_gru_overfit.sbatch captain_gru_v2_a<i>_g<j>`.

## A lever — what the GRU consumes and emits

| arm | `pose_gru_input` | `pose_gru_mode` | meaning |
|-----|------------------|-----------------|---------|
| A1  | `pose` (7-d)       | `direct`   | input P(x−1) → regress pose directly |
| A2  | `pose` (7-d)       | `residual` | input P(x−1) → pose + learned correction (zero-init head) |
| A3  | `pose_delta` (14-d) | `direct`   | input P(x−1) ⊕ inv(P(x−2))·P(x−1) → regress pose |
| A4  | `pose_delta` (14-d) | `residual` | input P(x−1) ⊕ inv(P(x−2))·P(x−1) → pose + correction |

The delta is the **relative rigid transform** between the last two raw head poses
(identity motion at view 1), not a componentwise encoding difference — quaternion
subtraction is meaningless under the double cover.

## G lever — where gradients flow

| arm | `pose_gru_bptt` | `pose_gru_e2e` | meaning |
|-----|-----------------|----------------|---------|
| G0  | False | False | sealed: aux loss reaches exactly one GRUCell call per step |
| G1  | True  | False | hidden tape spans one 4-view TBPTT chunk (boundary-detached like state/mem) |
| G2  | False | True  | GRU output into the ray build NOT detached → main reconstruction loss also trains the GRU |

The fed-back head pose (`_prev_pred_pose_enc`) stays detached under every setting —
structural, not a lever (the previous step's graph is freed under TBPTT).
G flags change **only gradient flow, never forward math**: eval/inference and the
falsifier probe treat all G arms identically, and a G1/G2 checkpoint loads and runs
exactly like a G0 one.

## Equivalences — don't retrain these

- **A2×G0 ≡ `captain_gru_v2.yaml`** (the original residual run) — reuse its checkpoints.
- **A1×G0 ≡ `captain_gru_v2_direct.yaml`** — reuse if that twin was trained.

## Protocol

One lever at a time; compare at fixed epochs (best-ckpt selection sees the aux term).
After each run, falsify against the trained checkpoint:
`sbatch probe_gru_hidden.sbatch <ckpt_dir>/checkpoint-<N>.pth`
(keep_freq snapshots only — never `checkpoint-last/best.pth` while the job lives).
G1 "worked" ⇔ `hidden_zero`/`hidden_shuffle` arms start degrading gru error (>5%).
Any arm "learned" ⇔ normal gru error beats the lag baseline by >5%.
