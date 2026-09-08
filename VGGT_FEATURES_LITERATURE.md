# VGGT-Omega latents into CUT3R: evidence base (2026-09-07)

Question: is a small trained adapter the right way to make VGGT-Omega features
consumable by CUT3R, and is it the most effective route? Compiled from two web
surveys (primary sources: arXiv, official repos) plus local benchmark data.
[V] = number re-read verbatim in two passes; [1] = single pass; NR = not retrieved.

## A. Local evidence (DROID wrist benchmark, this project)

Full 4292 scenes (cut3r_eval/summary/averages_table.csv):

| arm | ATE | rpe_rot | absrel | a1 |
|---|---|---|---|---|
| augfull_lr1e5 (plain finetune, "Regular CUT3R") | .0759 | 1.101 | .1794 | .786 |
| gtray_lr1e5 (current-GT ray map) | .0134 | .337 | .1886 | .787 |
| prevgt_lr1e5 (previous-frame GT ray map) | .0151 | .470 | .1861 | .789 |
| prevpred_lr1e5 (own previous pose) | .0821 | 1.197 | .1928 | .770 |
| augfull_cg_fuse_g7 (eval-side champion, zero training) | .0641 | .960 | .1749 | .792 |

Single smoke episode RAIL+80edfcb1+2023-07-14-14h-28m-45s (128 wrist frames):

| system | ATE | rpe_trans | rpe_rot | absrel |
|---|---|---|---|---|
| VGGT-Omega offline, all 128 wrist frames jointly (Sep 4 run) | .0061 | .0027 | 0.35 | .167 |
| CUT3R augfull_lr1e5 (online) | .0252 | .0057 | 0.87 | .129 |
| CUT3R gtray_lr1e5 (GT rays) | .0070 | .0030 | 0.20 | .167 |
| CUT3R prevgt_lr1e5 | .0108 | .0032 | 0.26 | .164 |
| CUT3R augfull_cg_fuse_g7 | .0153 | .0044 | 0.57 | .137 |

Laws already measured here: conditioning value is a train/test distribution
matching phenomenon (four-corner square); a one-step-lagged accurate pose
captures 97% of the conditioning prize; pose is the prize, depth has no
headroom from pose conditioning; training-side co-adaptation ate every
eval-side gate gain. DA3->CUT3R token swap (with_cut3r_v3.py) was never evaluated.

## B. Zero-initialised injection (consumer unchanged at init)

| paper | ablation |
|---|---|
| Flamingo (2204.14198) | tanh(0) gate: 70.7 vs 66.5 without, plus instabilities; every-layer xattn 70.7 vs single layer 59.8 |
| LLaMA-Adapter (2303.16199) | zero-init gate 83.85 vs random-init 40.77 (ScienceQA) |
| ControlNet (2302.05543) | Gaussian-init convs "destroy" the trainable copy; zero conv gives sudden convergence <10k steps |
| G-CUT3R (2508.11379) | zero-conv additive fusion into CUT3R decoder: Waymo L2 none/all 1.796/1.734 without zero-conv vs 1.235/1.042 with [V] |
| OmniVGGT (2511.10560) | Sintel AbsRel RGB/full-aux: replace-tokens 0.845/0.655; one-layer adapter 0.604/0.133; zero-conv GeoAdapter 0.558/0.106 [V] |

## C. Projector size once the consumer is finetuned

| paper | finding |
|---|---|
| Honeybee (2312.06742) Table 9 | Linear 70.0, 2-layer MLP 70.2, 6-layer MLP 69.1, Resampler 64.2, C-Abstractor 70.6 |
| LLaVA-1.5 (2310.03744) | linear->MLP: +0.5 GQA, +31 MME, +1.5 MM-Vet (smallest step in the table) |
| MM1 (2403.09611) | "type of VL connector has little effect"; tokens and resolution matter most |
| Cambrian-1 (2406.16860) | learned-query resampler loses dense info: OCR 27.1 vs concat 50.1 vs SVA 55.5 |
| Mai et al. 2026 (2603.12433) | linear vs MLP stitch DINOv2->SigLIP2: 26.1 vs 51.7 at layer 2, 69.1 vs 72.0 at layer 18 |
| Idefics2 (2405.02246) | frozen consumer: Linear 44.5, MLP 51.8, Perceiver 60.3 (resampler wins only when consumer frozen) |

## D. Adapter-only vs finetuning the consumer

| paper | frozen consumer | updated consumer |
|---|---|---|
| LLaVA (2304.08485) | 21.5 (stage 1 only) | 85.1 |
| VILA (2312.07533) 4-shot avg | 57.6 | 70.9 (linear projector) |
| NVLM-X 34B (2409.11402) OCRBench | 696 | 802 |
| CogVLM (2311.03079) VQAv2 | 73.8 | 78.9 / 80.0 (visual expert) |
| Flamingo | 70.7 | 62.7 (counter-example: small paired data, big LM forgets) |

## E. Where to inject with an updated consumer

| paper | finding |
|---|---|
| Idefics2 Table 3 | frozen: xattn 66.7 > token concat 60.3; with LoRA: token concat 69.5 > xattn 67.3 |
| NVLM (2409.11402) | decoder-only token concat >= gated xattn on OCR/DocVQA (89.1 vs 79.2 DocVQA) |
| VG-LLM (2505.24625) Table 7 | VGGT tokens: element-wise add 36.4 > cross-attn 34.4 > concat+MLP 27.7 (ScanRefer) |
| GAP-MLLM (2603.16461) | VGGT layers 5/11/17/24: gated add 50.6 > cross-attn 49.1 > add 47.9 |
| DeepStack (2406.04334) | inject in first half of layers; past midpoint hurts; 4 layers best |
| WorldMirror (2510.10726) | one global token per prior beats dense raymap for pose (33.82 vs 33.07) and intrinsics (86.48 vs 84.43) [1] |
| Pow3R (2503.17316) Table 7 | additive embed 77.4 vs MLP inject-1 77.7 vs inject-4 78.3 |
| MapAnything (2509.13414) | LN -> sum -> LN of all encoded inputs into per-view tokens; no extra tokens |
| DA3 (2511.10647) | camera token prepended, participates in all attention; pose active p=0.2 in training |

## F. 3D-family prior injection results (gain from an accurate external input)

| paper | metric | none | with pose prior |
|---|---|---|---|
| G-CUT3R (CUT3R + zero-conv priors) | 7-Scenes Acc/Comp | .098/.106 | pose .061/.075; all .048/.056 [V] |
| MapAnything | 2-view pose AUC | 56.0 | +K 64.7; +K+pose 93.6 [V] |
| DA3 Table 3 | VGGT HiRoom F1 | 56.7 | 70.2 [V] |
| Pow3R Table 1 | RRA@2 | 71.9 | RT 92.3; all 99.0 [V] |
| CUT3R (2501.12387) | 7-Scenes Acc/Comp | online .126/.154 | revisit pass, zero training .113/.107 [V] |

## G. VGGT latents consumed by another model via a small connector

| paper | connector | consumer | isolated gain |
|---|---|---|---|
| Spatial-MLLM (2505.23747) | MLP on concat | Qwen2.5-VL-3B SFT | +1.2 to +3.5 VSI-Bench [V] |
| VLM-3R (2505.20279) | cross-attn to CUT3R tokens + projector | LoRA r=128 | 57.74 -> 60.90; explicit points 57.87 [V] |
| VG-LLM (2505.24625) | MLP + element-wise add | LLM unfrozen | latent 36.4 vs predicted cam 32.1 / depth 32.3 / points 31.7 [V] |
| SpatialStack (2603.27437) | additive residuals from VGGT layers 11/17/23 into LLM layers 0/1/2 | fusion only | VSI-Bench 53.6 -> 67.5 [1] |
| VGGT-Omega (2605.15195) | 16 registers/frame concatenated | OpenVLA-OFT | LIBERO avg 97.1 -> 98.5 [V] |
| eVGGT (2509.15880) | MLP | diffusion policy | latents 35% vs explicit point clouds 14.5% [1] |
| StreamVGGT (2507.11539) | output distillation into causal student | from scratch | 7-Scenes Acc .202 -> .129 (teacher .088) |

Layer choice: RegimeVGGT (2606.18439) ridge probes: layers 11-18 carry most
geometry; 19-24 pose-critical, low effective rank. Consumers pick 11/17/23 or 5/11/17/24.

## H. Optional-input dropout

Pow3R: uniform random subset of modalities; no-prior path 39.4/79.4 vs DUSt3R
36.6/77.6 (no cost). MapAnything: overall p=0.9, each factor 0.5. G-CUT3R and
Rig3R (2506.02265): 50%. DA3: pose present only 20%. CFG (2207.12598):
p=0.5 costs FID 1.55 -> 1.91 in diffusion, 0.1-0.2 is free.

## Gaps
No paper feeds one 3D foundation model's latents into another 3D
reconstruction model. No paper uses static exterior cameras as context for a
streaming model. TTT3R (2509.26645) numbers are figure-only.
