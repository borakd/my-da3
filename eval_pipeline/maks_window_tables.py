#!/usr/bin/env python3
"""Six LaTeX tables (+ PDF + PNG) in the gru_pose_focus_table format (CUT3R
metrics only) for the single-scene memory-write experiments:

  window_D{1,2,3}.tex   baseline (window included) vs that window's frames
                        excluded from memory writes, plus all windows excluded
  preblock_D{1,2,3}.tex baseline vs the pre-window frames excluded, plus the
                        window itself excluded (for direct comparison)

Values are the scene's end-of-sequence table numbers (per-scene MEAN row of
eval_depth_poses.py: depth mean over frames, pose RMSE over frames, Sim3).
Top-3 per column highlighted red / orange / yellow (ties share), best in bold.
Frame ranges are printed 1-indexed.

  python maks_window_tables.py --scene_dir <scene eval dir> --block_root <maks_block_windows>
      --scene <name> --out_dir <dir>
"""
import argparse
import csv
import json
import math
import os
import subprocess

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER = {"a1"}
COLORS = ["red!30", "orange!30", "yellow!40"]


def scene_row(p):
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["camera_id"] == "ALL" and r["local_timestep"] == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    for r in rows:
        if r["local_timestep"] == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    raise ValueError(p)


def fmt(m, v):
    return f"{v:.3f}"


def esc(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("+", r"{+}")


COMMENT_CAPTIONS = False


def table(rows, title_note, out_tex):
    """rows: list of (label, {metric: value})."""
    # rank per column: best -> COLORS[0]; ties share
    cell = {}
    bold_only = len(rows) <= 2
    for m in METRICS:
        vals = sorted({round(v[m], 3) for _, v in rows}, reverse=(m in HIGHER))
        rank = {v: i for i, v in enumerate(vals[:3])}
        for i, (lab, v) in enumerate(rows):
            r = rank.get(round(v[m], 3))
            s = fmt(m, v[m])
            if r is None:
                cell[(i, m)] = s
            elif bold_only:
                cell[(i, m)] = f"\\textbf{{{s}}}" if r == 0 else s
            else:
                cell[(i, m)] = f"\\cellcolor{{{COLORS[r]}}}" + (f"\\textbf{{{s}}}" if r == 0 else s)
    have_standalone = subprocess.run(["kpsewhich", "standalone.cls"], capture_output=True, text=True).stdout.strip() != ""
    head = ([r"\documentclass[11pt,border=8pt]{standalone}"] if have_standalone else
            [r"\documentclass[11pt]{article}", r"\usepackage[margin=8pt,paperwidth=30cm,paperheight=12cm]{geometry}",
             r"\pagestyle{empty}"])
    L = head + [r"\usepackage{booktabs}",
         r"\usepackage[table]{xcolor}", r"\newsavebox{\tblbox}", r"\begin{document}",
         r"\sbox{\tblbox}{%", r"\begin{tabular}{@{}l r r | r r r@{}}", r"\toprule",
         r" & \multicolumn{2}{c|}{Depth} & \multicolumn{3}{c}{CUT3R pose metrics} \\",
         r"Method & AbsRel$\downarrow$ & $\delta{<}1.25\uparrow$",
         r" & ATE$\downarrow$ & RPE\textsubscript{trans}$\downarrow$ & RPE\textsubscript{rot}$\downarrow$ \\",
         r"\midrule"]
    w = max(len(lab) for lab, _ in rows)
    for i, (lab, v) in enumerate(rows):
        L.append(f"{esc(lab):<{w+6}} & " + " & ".join(cell[(i, m)] for m in METRICS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}%", r"\begin{minipage}{\wd\tblbox}",
          r"\usebox{\tblbox}\par\vspace{2pt}",
          r"{\footnotesize " + title_note + r"\par}",
          r"{\footnotesize End-of-scene averages: depth = mean over frames (per-frame median-scaled), "
          r"pose = Sim3-aligned RMSE over frames. RPE\textsubscript{rot} in degrees. Frame ranges are 1-indexed.\par}",
          (r"{\footnotesize \textbf{Bold} = best per column.\par}" if bold_only else
           r"{\footnotesize Highlighting: \colorbox{red!30}{\textbf{best}}, \colorbox{orange!30}{2nd}, "
           r"\colorbox{yellow!40}{3rd} per column (ties share)\par}"),
          r"\end{minipage}", r"\end{document}"]
    if COMMENT_CAPTIONS:
        L = [("% " + l) if l.startswith(r"{\footnotesize") else l for l in L]
    open(out_tex, "w").write("\n".join(L) + "\n")


def compile_png(tex):
    d, base = os.path.split(tex)
    stem = os.path.splitext(base)[0]
    r = subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error", base], cwd=d,
                       capture_output=True, text=True)
    pdf = os.path.join(d, stem + ".pdf")
    if r.returncode != 0 or not os.path.isfile(pdf):
        log = open(os.path.join(d, stem + ".log")).read()[-1500:] if os.path.isfile(os.path.join(d, stem + ".log")) else r.stdout[-1500:]
        raise RuntimeError(f"pdflatex failed for {tex}:\n{log}")
    png = os.path.join(d, stem + ".png")
    r = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r220",
                        f"-sOutputFile={png}", pdf], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(png):
        raise RuntimeError(f"gs failed for {pdf}: {r.stderr[-800:]}")
    # article fallback renders a full page: trim the whitespace, keep a small border
    import shutil
    if shutil.which("convert"):  # ImageMagick is absent on compute nodes; standalone output needs no trim
        subprocess.run(["convert", png, "-trim", "-bordercolor", "white", "-border", "16", png],
                       capture_output=True, text=True)
    for ext in (".aux", ".log"):
        try:
            os.remove(os.path.join(d, stem + ext))
        except OSError:
            pass
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="baseline scene eval dir (holds decline_windows.json, exp1_preblock/)")
    ap.add_argument("--block_root", required=True, help="maks_block_windows root (maks_block_windows_D*/ALL)")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_label", default="augfull\\_lr1e5 (finetuned, 50 ep, lr 1e-5)")
    ap.add_argument("--comment_captions", action="store_true", help="keep caption text in the .tex but commented out")
    args = ap.parse_args()
    global COMMENT_CAPTIONS
    COMMENT_CAPTIONS = args.comment_captions
    os.makedirs(args.out_dir, exist_ok=True)
    S = args.scene
    base = scene_row(os.path.join(args.scene_dir, "eval_depth_pose_metrics.csv"))
    wins = [tuple(r["window"]) for r in json.load(open(os.path.join(args.scene_dir, "decline_windows.json")))["declines"]]
    n_frames = len([l for l in csv.DictReader(open(os.path.join(args.scene_dir, "eval_depth_pose_metrics.csv")))
                    if l["camera_id"] == "0" and l["local_timestep"].isdigit()])
    blk = {k: scene_row(os.path.join(args.block_root, f"maks_block_windows_D{k}", "eval", S, "eval_depth_pose_metrics.csv"))
           for k in range(1, len(wins) + 1)}
    allb = scene_row(os.path.join(args.block_root, "maks_block_windows_ALL", "eval", S, "eval_depth_pose_metrics.csv"))
    pre, pre_rng = {}, {}
    for k in range(1, len(wins) + 1):
        pre[k] = scene_row(os.path.join(args.scene_dir, "exp1_preblock", f"pre_D{k}", "eval", S, "eval_depth_pose_metrics.csv"))
        fr = json.load(open(os.path.join(args.scene_dir, "exp1_preblock", "masks", f"pre_D{k}.affected.json")))[S]
        pre_rng[k] = (fr[0], fr[-1])

    def r1(a, b):  # 1-indexed display
        return f"{a + 1}--{b + 1}"
    all_rng = ", ".join(r1(a, b) for a, b in sorted(wins))
    scene_note = (f"Scene {esc(S)}, {n_frames} frames, checkpoint {args.ckpt_label}. "
                  "Excluded = frames still decoded and scored but blocked from writing state/memory.")
    outs = []
    for k, (a, b) in enumerate(wins, 1):
        rows = [("Regular CUT3R (window included)", base),
                (f"Window D{k} excluded (frames {r1(a, b)})", blk[k])]
        tex = os.path.join(args.out_dir, f"window_D{k}.tex")
        table(rows, scene_note, tex); outs.append(compile_png(tex))
    for k, (a, b) in enumerate(wins, 1):
        pa, pb = pre_rng[k]
        rows = [("Regular CUT3R (all frames included)", base),
                (f"Pre-window frames excluded ({r1(pa, pb)})", pre[k]),
                (f"Window D{k} itself excluded ({r1(a, b)})", blk[k])]
        tex = os.path.join(args.out_dir, f"preblock_D{k}.tex")
        table(rows, scene_note, tex); outs.append(compile_png(tex))
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
