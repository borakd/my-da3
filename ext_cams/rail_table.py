#!/usr/bin/env python
"""RAIL-scene comparison table (LaTeX, standalone): CUT3R alone vs the exterior-camera rig and the two fusions, both backbones.
Numbers are read from the JSON products of rig_fuse_probe.py (shelf fusion) and rig_fuse2.py (honest fusion).
Shading: best (red) / second (orange) per metric within each frame group, at displayed precision; rows that use ground truth in
the method path (dagger) are excluded from the shading, as oracles are in the master table.

    python rail_table.py   -> $OUT/ext_cams/tables/rail_fusion.tex (compile separately)
"""
import json, os
OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams"
EP = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
TAB = f"{OUT}/tables"; os.makedirs(TAB, exist_ok=True)

shelf = {b: json.load(open(f"{OUT}/rig_kabsch/{EP}/fuse_summary{'' if b == 'finetuned' else '_zeroshot'}.json"))["rows"] for b in ("finetuned", "zeroshot")}
honest = {b: json.load(open(f"{OUT}/fuse2/rail_boxclick_v2/fuse2_{b}.json"))["info"]["rows"] for b in ("finetuned", "zeroshot")}
blend = {b: json.load(open(f"{OUT}/fuse2/rail_boxclick_v2_rotshared/fuse2_{b}.json"))["info"]["rows"] for b in ("finetuned", "zeroshot")}

def m(r): return (r["ate"], r["rpe_trans"], r["rpe_rot"])
PREC = (4, 4, 3)

def rows_for(b):
    s, h, bl = shelf[b], honest[b], blend[b]
    return [
        ("All 128 frames", [
            ("CUT3R alone (wrist stream only)", "no", "yes", 128, m(s["cut3r_all_frames"])),
            ("Fusion, shelf probe (GT lens pose at the reference frame)$^{\\dagger}$", "yes", "no$^{\\ast}$", 128, m(s["fused_all_frames"])),
            ("Fusion, honest (GT-free, tracker v2 anchors, CUT3R rotation)", "no", "yes", 128, m(h["fused_all_frames"])),
            ("Fusion, honest, rig rotation blended", "no", "yes", 128, m(bl["fused_all_frames"])),
        ]),
        ("Anchored frames only, shelf rig (84 frames)", [
            ("Rig alone, lens trajectory (GT lens pose at the reference frame)$^{\\dagger}$", "yes", "no$^{\\ast}$", 84, m(s["rig_anchored_frames_only"])),
            ("CUT3R alone, same frames", "no", "yes", 84, m(s["cut3r_anchored_frames_only"])),
            ("Fusion, shelf probe$^{\\dagger}$", "yes", "no$^{\\ast}$", 84, m(s["fused_anchored_frames_only"])),
        ]),
        ("Anchored frames only, tracker v2 (73 frames)", [
            ("Rig alone, body-centroid trajectory (GT-free)", "no", "yes", 73, m(h["rig_anchored_frames_only(centroid)"])),
            ("CUT3R alone, same frames", "no", "yes", 73, m(h["cut3r_anchored_frames_only"])),
            ("Fusion, honest", "no", "yes", 73, m(h["fused_anchored_frames_only"])),
            ("Fusion, honest, rig rotation blended", "no", "yes", 73, m(bl["fused_anchored_frames_only"])),
        ]),
    ]

def shade(group):
    """per metric column: red best, orange second, among rows with gt == 'no'; ties share; displayed precision."""
    cells = {}
    for j in range(3):
        vals = sorted({round(r[4][j], PREC[j]) for r in group if r[1] == "no"})
        if sum(1 for r in group if r[1] == "no") < 2: vals = []
        for i, r in enumerate(group):
            v = round(r[4][j], PREC[j])
            col = ""
            if r[1] == "no" and vals:
                if v == vals[0]: col = r"\cellcolor{red!30}"
                elif len(vals) > 1 and v == vals[1]: col = r"\cellcolor{orange!30}"
            cells[(i, j)] = col
    return cells

L = [r"\documentclass[11pt,border=8pt]{standalone}", r"\usepackage{booktabs}", r"\usepackage[table]{xcolor}", r"\usepackage{amsmath}",
     r"\newsavebox{\tblbox}", r"\begin{document}", r"\sbox{\tblbox}{%", r"\begin{tabular}{@{}l c c r r r r@{}}", r"\toprule",
     r"Method & GT in method & causal & frames & ATE$\downarrow$ & RPE\textsubscript{trans}$\downarrow$ & RPE\textsubscript{rot}$\downarrow$ \\", r"\midrule"]
for b, title in (("finetuned", r"Finetuned CUT3R backbone (augfull\_lr1e5)"), ("zeroshot", r"Pretrained CUT3R backbone (zero-shot)")):
    L += [r"\addlinespace[3pt]", rf"\multicolumn{{7}}{{@{{}}l}}{{\bfseries {title}}} \\", r"\addlinespace[1pt]"]
    for gname, group in rows_for(b):
        L += [rf"\multicolumn{{7}}{{@{{}}l}}{{\itshape\footnotesize {gname}}} \\"]
        sh = shade(group)
        for i, (name, gt, causal, n, vals) in enumerate(group):
            cells = " & ".join(f"{sh[(i, j)]}{vals[j]:.{PREC[j]}f}" for j in range(3))
            L.append(rf"\quad {name} & {gt} & {causal} & {n} & {cells} \\")
    L.append(r"\midrule" if b == "finetuned" else r"\bottomrule")
L += [r"\end{tabular}}%", r"\begin{minipage}{\wd\tblbox}", r"\usebox{\tblbox}\par\vspace{3pt}",
      r"{\footnotesize One DROID test scene, RAIL+80edfcb1+2023-07-14-14h-28m-45s, 128 frames, exterior cameras about 0.5\,m from the workspace. Metrics as in the master table: poses Sim3-aligned on camera centres and RMSE-reduced across frames, RPE over consecutive frames, RPE\textsubscript{rot} in degrees, against the kinematic wrist-camera ground truth (scoring only). Shading: best (red) and second (orange) per column within each frame group, among rows with no ground truth in the method path; rows marked $^{\dagger}$ use the GT wrist pose once, at the rig's reference frame, as a hand-eye stand-in and are excluded from the shading.\par}",
      r"{\footnotesize \emph{Rig}: gripper points tracked in the two static exterior cameras (hand-prompted SAM3 masks on this scene), paired across views by the epipolar constraint, triangulated with the PointWorld extrinsics and factory intrinsics, aligned to a growing body model by RANSAC + Kabsch. \emph{Shelf probe}: the single-episode pipeline of \texttt{rig\_kabsch\_probe.py} + \texttt{rig\_fuse\_probe.py} (84 anchored frames; $^{\ast}$its cross-view pairing used each track's whole lifetime, so it is not strictly causal). \emph{Honest}: \texttt{rig\_track2.py} + \texttt{rig\_fuse2.py}, pairs confirmed online, 73 anchored frames, no GT anywhere: CUT3R's own relative motion every frame, pulled toward the rig's body-centroid anchor with scalar Kalman gains from variances measured online; rotation is CUT3R's unless ``rig rotation blended''. The rig-alone rows are the anchor source scored as a trajectory: the shelf row is the lens (GT reference), the tracker-v2 row the body centroid (a point about 10\,cm from the lens, hence its larger RPE\textsubscript{trans}).\par}",
      r"\end{minipage}", r"\end{document}"]
open(f"{TAB}/rail_fusion.tex", "w").write("\n".join(L) + "\n")
print("wrote", f"{TAB}/rail_fusion.tex")
