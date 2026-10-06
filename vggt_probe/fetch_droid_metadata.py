#!/usr/bin/env python
"""Fetch the DROID raw `metadata_<EP>.json` for every scene in the Leonardo test split (login
node; outbound HTTPS). The raw path is derived from the episode name:
  <LAB>+<user8>+<YYYY-MM-DD>-<HH>h-<MM>m-<SS>s  ->  <LAB>/success/<YYYY-MM-DD>/<Www_Mon_%e_HH:MM:SS_YYYY>/
(`%e` is space-padded, spaces become underscores, e.g. `Sat_Oct__7_...`). Episodes not found under
success/ are retried under failure/. Also fetches the ZED factory calibration for every distinct
exterior-camera serial. Writes a URL list and drives `curl` with xargs -P so no single process
approaches the login-node CPU limit.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys

STORE = "/leonardo_scratch/large/userexternal/bdursun0/pointworld_droid_splits/test/dl3dv_multi/wrist"
OUT_ROOT = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames"
META_DIR = f"{OUT_ROOT}/metadata"
CALIB_DIR = f"{OUT_ROOT}/zed_calib"
BASE = "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1"
PAR = int(os.environ.get("PAR", "16"))


def raw_dir(ep, outcome="success", day_offset=0, sep=":"):
    """The date FOLDER can differ from the episode's own date by one day (recordings shortly
    after midnight sit under the previous day's folder), so callers try day_offset in (0,-1,+1).
    The inner `Www_Mon_%e_HH:MM:SS_YYYY` stamp always follows the episode's own timestamp; a few
    early (IRIS, March 2023) folders use `HH_MM_SS` instead of `HH:MM:SS` (sep="_")."""
    lab, _, ts = ep.split("+")
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h-(\d{2})m-(\d{2})s", ts)
    y, mo, d, hh, mm, ss = map(int, m.groups())
    date = dt.date(y, mo, d)
    folder = date + dt.timedelta(days=day_offset)
    stamp = f"{date.strftime('%a')}_{date.strftime('%b')}_{d:>2d}_{hh:02d}{sep}{mm:02d}{sep}{ss:02d}_{y}".replace(" ", "_")
    return f"{lab}/{outcome}/{folder.isoformat()}/{stamp}"


def run_curls(pairs):
    """pairs: list of (url, dest). Skips existing non-empty dests."""
    todo = [(u, d) for u, d in pairs if not (os.path.isfile(d) and os.path.getsize(d) > 0)]
    if not todo:
        return
    lst = "\n".join(f"{u}\t{d}" for u, d in todo) + "\n"
    cmd = (f"xargs -P {PAR} -L 1 bash -c 'curl -sfL --retry 2 -o \"$1\" \"$0\" || rm -f \"$1\"'")
    subprocess.run(cmd, input=lst.encode(), shell=True, check=False)


def main():
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)
    eps = sorted(d for d in os.listdir(STORE) if os.path.isdir(f"{STORE}/{d}"))
    print(f"{len(eps)} scenes", flush=True)
    for sep in (":", "_"):
        for day_offset in (0, -1, +1):
            for outcome in ("success", "failure"):
                todo = [ep for ep in eps if not os.path.isfile(f"{META_DIR}/{ep}.json")]
                if not todo:
                    break
                run_curls([(f"{BASE}/{raw_dir(ep, outcome, day_offset, sep)}/metadata_{ep}.json",
                            f"{META_DIR}/{ep}.json") for ep in todo])
                have = sum(os.path.isfile(f"{META_DIR}/{ep}.json") for ep in eps)
                print(f"after {outcome} (day {day_offset:+d}, sep '{sep}'): {have}/{len(eps)} metadata files", flush=True)
    missing = [ep for ep in eps if not os.path.isfile(f"{META_DIR}/{ep}.json")]
    json.dump(missing, open(f"{OUT_ROOT}/metadata_missing.json", "w"), indent=1)
    print(f"missing: {len(missing)}", flush=True)

    serials = set()
    for ep in eps:
        p = f"{META_DIR}/{ep}.json"
        if os.path.isfile(p):
            m = json.load(open(p))
            serials.add(str(m["ext1_cam_serial"]))
            serials.add(str(m["ext2_cam_serial"]))
    print(f"{len(serials)} distinct exterior serials", flush=True)
    run_curls([(f"https://calib.stereolabs.com/?SN={sn}", f"{CALIB_DIR}/SN{sn}.conf") for sn in sorted(serials)])
    bad = [sn for sn in serials if not (os.path.isfile(f"{CALIB_DIR}/SN{sn}.conf")
                                        and "[LEFT_CAM_HD]" in open(f"{CALIB_DIR}/SN{sn}.conf").read())]
    print(f"calibration files missing/invalid: {len(bad)} {bad[:10]}", flush=True)


if __name__ == "__main__":
    main()
