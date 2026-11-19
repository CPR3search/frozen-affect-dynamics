"""Aff-Wild2 valence clip construction and label rules."""
from __future__ import annotations
import glob, json, os, sys
from collections import Counter

import numpy as np


ANN = "data/8th_ABAW_Annotations/VA_Estimation_Challenge"
FRAMES = "data/cropped_aligned"
OUT = "runs/affwild_clips_features"
T = 16                      # Match the UNBC clip length.
STRIDE = T // 2
INVALID = -4.0              # Aff-Wild2 marks unannotated frames with -5.
MIN_STD = 0.03              # Exclude windows with insufficient affect variation.
DIR_THR = 0.15              # Require this absolute endpoint valence change for direction labels.
PEAK_MARGIN = 0.10          # Require the interior maximum to exceed both endpoints by this margin.
MAXPER = 40                 # Cap each video-ID group's clip contribution.


def _subject(video):
    """Return the video-ID prefix before the first hyphen; this is a cluster key, not a verified person
    identity. Stored records call this field 'subject'.
    """
    return video.split("-")[0]


def _read_va(path):
    vals = []
    for line in open(path).read().splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line.split(",")[0]))          # Read valence from the first column.
        except ValueError:
            vals.append(INVALID - 1)
    return np.array(vals, np.float32)


def _label(win, dir_thr, peak_margin):
    """Return (direction, endpoint_change, traj). direction is 1=onset, 0=offset, -1=ineligible using
    inclusive +/-dir_thr margins. traj is 2=peaked, 0=rising non-peaked, 1=falling non-peaked,
    -1=ineligible. A peak takes precedence: the first maximum must lie at frames 4-13 inclusive and
    exceed both endpoints by at least peak_margin; otherwise absolute endpoint change must reach
    peak_margin.
    """
    d = float(win[-1] - win[0])
    a = int(np.argmax(win))
    direction = 1 if d >= dir_thr else (0 if d <= -dir_thr else -1)
    peaked = (3 <= a <= T - 4 and win[a] >= win[0] + peak_margin
              and win[a] >= win[-1] + peak_margin)
    traj = 2 if peaked else (0 if d >= peak_margin else (1 if d <= -peak_margin else -1))
    return direction, d, traj


def build_clips(dir_thr=DIR_THR, peak_margin=PEAK_MARGIN, maxper=MAXPER, require_frames=True):
    clips, per = [], Counter()
    for f in sorted(glob.glob(f"{ANN}/*/*.txt")):
        vid = os.path.basename(f)[:-4]
        vdir = os.path.join(FRAMES, vid)
        if require_frames and not os.path.isdir(vdir):
            continue
        va = _read_va(f)
        subj = _subject(vid)
        # Historical extraction boundary: the stop is exclusive at length-T, so a window starting exactly there is omitted, including the sole window of a T-frame sequence. Retained to reproduce the released clip selection.
        for i in range(0, len(va) - T, STRIDE):
            if per[subj] >= maxper:
                break
            win = va[i:i + T]
            if (win <= INVALID).any() or float(win.std()) < MIN_STD:
                continue
            direction, slope, traj = _label(win, dir_thr, peak_margin)
            if direction == -1 and traj == -1:
                continue
            paths = [os.path.join(vdir, f"{i + k + 1:05d}.jpg") for k in range(T)]
            if require_frames and not all(os.path.exists(p) for p in paths):
                continue
            clips.append({"video": vid, "subject": subj, "paths": paths,
                          "direction": direction, "slope": slope, "traj": traj})
            per[subj] += 1
    return clips


def compose(clips):
    d = Counter(c["direction"] for c in clips)
    t = Counter(c["traj"] for c in clips)
    return {"clips": len(clips), "subjects": len({c["subject"] for c in clips}),
            "videos": len({c["video"] for c in clips}),
            "onset": d[1], "offset": d[0], "direction_labelled": d[1] + d[0],
            "rise": t[0], "fall": t[1], "peaked": t[2],
            "traj_labelled": t[0] + t[1] + t[2]}


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/manifest.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log("W3 Aff-Wild2 valence-track clip mining")
    log(f"  T={T} stride={STRIDE} min_std={MIN_STD} maxper={MAXPER}")
    log("  thresholds are stated, not tuned; the grid below shows what they buy\n")

    log("=== sensitivity of the clip set to the two thresholds (frame check skipped for speed) ===")
    log(f"  {'dir_thr':>8s} {'peak_m':>7s} {'clips':>7s} {'subj':>5s} {'dir-lab':>8s} "
        f"{'peaked':>7s} {'rise':>6s} {'fall':>6s}")
    for dt in (0.10, 0.15, 0.20):
        for pm in (0.05, 0.10, 0.15):
            c = compose(build_clips(dt, pm, MAXPER, require_frames=False))
            log(f"  {dt:8.2f} {pm:7.2f} {c['clips']:7d} {c['subjects']:5d} "
                f"{c['direction_labelled']:8d} {c['peaked']:7d} {c['rise']:6d} {c['fall']:6d}")

    log(f"\n=== chosen: dir_thr={DIR_THR} peak_margin={PEAK_MARGIN}, frames verified on disk ===")
    clips = build_clips()
    c = compose(clips)
    for k, v in c.items():
        log(f"  {k:20s} {v}")

    pk = c["peaked"]
    log(f"\n  peaked clips: {pk} against 27 on UNBC "
        f"({pk / 27:.1f}x), over {c['subjects']} subjects against 24")
    if pk < 100:
        log("  WARNING: fewer peaked clips than hoped; consider a lower peak margin or a longer "
            "clip length before spending extraction compute.")

    json.dump({"config": {"T": T, "stride": STRIDE, "min_std": MIN_STD, "dir_thr": DIR_THR,
                          "peak_margin": PEAK_MARGIN, "maxper": MAXPER, "invalid": INVALID},
               "composition": c,
               "clips": [{k: v for k, v in cl.items() if k != "paths"} | {"start": cl["paths"][0]}
                         for cl in clips]},
              open(f"{OUT}/manifest.json", "w"), indent=2)
    np.savez_compressed(f"{OUT}/labels.npz",
                        subject=np.array([cl["subject"] for cl in clips]),
                        video=np.array([cl["video"] for cl in clips]),
                        direction=np.array([cl["direction"] for cl in clips], np.int64),
                        slope=np.array([cl["slope"] for cl in clips], np.float32),
                        traj=np.array([cl["traj"] for cl in clips], np.int64))
    log(f"\nwrote {OUT}/manifest.json and {OUT}/labels.npz")
    log("Feature extraction is a separate step; this run costs no GPU.")
    logf.close()


if __name__ == "__main__":
    main()
