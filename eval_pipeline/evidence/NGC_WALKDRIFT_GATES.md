# NGC-WalkDrift short rung — FROZEN GATES (committed before sbatch submission)
Arm: captain_gru_v3_ngc_walkdrift_short_finetune (phase A, trunk-pure,
10 epochs, 8 nodes/32 GPU, lr 1e-5 compressed cosine; GRU grafted in phase B).
Source: design-lock round 2 synthesis (wf_3010e1ec-5ea); noise params frozen
in ngc_drift_warmstart.json (d_t=0.0362/view, d_r=0.34 deg/view from the
B-closed-loop Theil-Sen fit) and ngc_t_frac_calibration.json (t_frac=0.3837).
Subset references (430, paired): A .0771 / C .0850 / r8 .0816 / refine .0929
/ B-closed .1307. Floor .00264.

## Pre-launch (all must pass; any failure = do not launch)
F1-F5 verify_ray_cond_noise.py exit 0 (archived log). F6 drift fit finite,
positive, within 5x of naive slope (PASS: 0.0362 vs 0.0444 = 1.23x). F7 graft
viability on the CURRENT B checkpoint (phase-B machinery proven before phase
A exists). F8 config-diff hygiene (PASS: diff shows only intended keys;
test_data clean; ORACLE pinned off since commit 90143dc). F9 t_frac
round-trip (PASS: 0.3837 x 0.2021 = 0.07754 exact).

## Early-out (ckpt-5, ~2.2 khours in)
Honest closed-loop (prev_pred, no GRU) subset ATE > .1307 (worse than
B-closed) => KILL, skip all remaining eval spend. Sole retune exception:
rot-loss divergence visible in wandb early => one relaunch at rotation
m-cap 0.5, consuming the single retune iteration.

## Gates on fixed ckpt-10 (never checkpoint-best)
G1 clean ceiling (eval clean prev-GT rays): PASS ate_clean <= .045;
   KILL > .060. GT_RAY_MAP_SHUFFLE spot check must degrade metrics, else the
   conditioning channel is dead => KILL.
G2 (advisory, required context): drifted-replay walk-1.0 delta_ate vs clean
   <= .015 (<= 0.5x B's ~.029) = curve flattening demonstrated.
G3 DECISIVE (honest closed-loop prev_pred, no GRU):
   PROMOTE  closed_ate <= .0771 (beats A subset) AND rpe_rot <= 1.10x A
            AND absrel within +/-3% of A.
   CONDITIONAL .0771 < closed_ate <= .0850 (beats C): exactly ONE retune
            iteration informed by G1/G2, then re-score; second failure = KILL.
   KILL     closed_ate > .0850.
   R = closed_ate / drifted-replay ate interpolated at the arm's own measured
   conditioning error (procedure frozen: median per-view closed-loop error,
   /1.5382 convention, view-0 frame; replay at {0.5,1.0,1.5}x sigma_ref with
   fitted drift on THIS checkpoint; log-linear interpolation).
G4 graft (phase B1, zero training): splice r8-refine pose_gru into ckpt-10,
   verify battery + 5-scene smoke; then closed-loop prev_pred_gru subset run
   must not degrade ATE by > .00264 vs G3's read (GRU at worst neutral).
   If degraded: phase B2 graft-fit (2 epochs, GRU-only lr, trunk quasi-frozen)
   is the one permitted fallback.

## STOP RULE (terminates the campaign with a final negative verdict)
S1: G3 KILL with R >= 1.5 while G2 shows flattening (delta <= .015) — the
trunk demonstrably learned exogenous robustness including the fitted first
moment, yet self-generated error still detonates the loop. Pre-registered
decision: do NOT fund self-fed/DAgger co-training (already indicted by
evidence triad (1)); write the final campaign verdict.

## Promotion to the 50-epoch benchmark bet (~35 khours) requires ALL
P1: F/G0 battery + G1 PASS + G3 PROMOTE (not CONDITIONAL) + G4 PASS
    (G2 and R reported in the GO record).
P2: full-harness transfer check on ckpt-10 via arm_eval_for_run.sh
    (prev_pred_gru, 4292 scenes): harness ate <= .0855 AND ATE win% vs C
    within its replicate band of the trajectory implied by the subset read.
P3: 50-epoch arm = same config with epochs 50, keep_freq 10, fresh exp_name,
    wiped output dir, fresh wandb id; scored only by PREREG_VERDICT_RULES.md.
