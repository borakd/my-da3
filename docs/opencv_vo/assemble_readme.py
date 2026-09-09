#!/usr/bin/env python
"""Assemble docs/opencv_vo/README.md from the per-file sections in docs/opencv_vo/sections/.

The sections are the source of truth; README.md is the concatenation with a generated table of
contents and part dividers. Run from anywhere: python docs/opencv_vo/assemble_readme.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "sections")
ORDER = ["00_overview.md", "10_vo_header_constants.md", "11_vo_loader_focal.md", "12_vo_tracks_geometry.md",
         "13_vo_local_ba.md", "14_vo_scene_setup.md", "15_vo_bootstrap_loop.md", "16_vo_tracking_output.md",
         "17_vo_cli.md", "20_vo_eval.md", "21_build_smoke_table.md", "22_build_full_table.md",
         "30_probe_corr_bench.md", "31_probe_pnp_bench.md", "32_probe_tri_bench.md", "33_probe_e_diag.md",
         "34_probe_launchers.md", "90_results.md"]
PART = {"10_vo_header_constants.md": "Part I — The pipeline: `eval_pipeline/opencv_vo.py`",
        "20_vo_eval.md": "Part II — Scoring and tables",
        "30_probe_corr_bench.md": "Part III — The benchmarks that settled each decision",
        "90_results.md": "Part IV — Results"}
NOTE = ("> This is one self-contained document (~7.2k lines). It is assembled from the per-file\n"
        "> sections in [`sections/`](sections/); if you prefer to read one source file at a time,\n"
        "> open the matching `sections/*.md` — the content is identical. Rebuild with\n"
        "> `python docs/opencv_vo/assemble_readme.py`.\n\n")


def slug(h):
    s = h.strip().lstrip("#").strip().lower()
    s = re.sub(r"[`*]", "", s)
    s = re.sub(r"[^a-z0-9 \-–—]", "", s).replace(" ", "-")
    return re.sub(r"-+", "-", s)


def main():
    parts, toc = [], []
    for f in ORDER:
        body = open(os.path.join(S, f)).read().rstrip() + "\n"
        if f in PART:
            parts.append(f"\n---\n\n# {PART[f]}\n")
            toc.append(f"\n**{PART[f]}**\n")
        head = body.splitlines()[0]
        if f != "00_overview.md":
            toc.append(f"- [{head.lstrip('#').strip()}](#{slug(head)})")
        parts.append(body)
    marker = "## 1. What the backbone is"
    toc_md = "## Contents\n\n" + "\n".join(toc) + "\n"
    parts[0] = parts[0].replace(marker, NOTE + toc_md + "\n" + marker, 1)
    out = ("<!-- Generated document: assembled from docs/opencv_vo/sections/*.md.\n"
           "     Every design decision is cross-referenced to OPENCV_VO_DESIGN.md (repo root). -->\n\n"
           + "\n".join(parts))
    open(os.path.join(HERE, "README.md"), "w").write(out)
    print(f"README.md: {len(out.splitlines())} lines")


if __name__ == "__main__":
    main()
