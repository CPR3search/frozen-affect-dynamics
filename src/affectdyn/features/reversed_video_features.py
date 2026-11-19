"""Video-encoder features for time-reversed clips.

Usage
  python -m affectdyn.features.reversed_video_features --gate            # driver check against the published caches
  python -m affectdyn.features.reversed_video_features --extract [names] # the reversed caches
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, sys, time

import numpy as np

from affectdyn.data.unbc_clip_pipeline import build_clips                                           # noqa: E402

OUT = "runs/video_reversal_transfer"
GATE_N = 20
GATE_TOL = 1e-4

PUBLISHED = {
    "vjepa":        "runs/temporal_aggregation/vjepa_windows.npz",
    "videomae":     "runs/temporal_aggregation/videomae_windows_fixed.npz",
    "videomaev2":   "runs/temporal_aggregation/videomaev2_windows.npz",
    "timesformer":  "runs/pain_encoder_maps/timesformer_windows.npz",
    "xclip":        "runs/pain_encoder_maps/xclip_windows.npz",
    "internvideo2": "runs/pain_encoder_maps/internvideo2_windows.npz",
}


def extractor(name):
    if name in ("timesformer", "xclip"):
        import affectdyn.features.video_window_features as w2e
        return lambda clips, cache: w2e.extract(name, clips, cache)
    if name == "vjepa":
        import affectdyn.features.temporal_aggregation as g0r
        return lambda clips, cache: g0r.extract_vjepa(clips, cache)
    if name in ("videomae", "videomaev2"):
        import affectdyn.features.videomae_bias_correction as fix
        loader = fix.load_videomae if name == "videomae" else fix.load_videomae_v2
        layout = "BCTHW" if name == "videomaev2" else "BTCHW"
        return lambda clips, cache: fix._extract(name, loader, layout, clips, cache)
    if name == "internvideo2":
        import affectdyn.features.internvideo2_window_features as iv2
        return lambda clips, cache: iv2.extract(clips, cache)
    raise SystemExit(f"no extractor for {name}")


def reversed_clips(clips):
    return [{**c, "paths": list(c["paths"])[::-1]} for c in clips]


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="re-extract a sample in ORIGINAL order")
    ap.add_argument("--extract", action="store_true", help="write the reversed caches")
    ap.add_argument("names", nargs="*", default=None)
    a = ap.parse_args()
    names = a.names or list(PUBLISHED)
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/extract.log", "a")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    clips = build_clips()
    p(f"W25 | {len(clips)} UNBC clips | encoders {', '.join(names)}")

    if a.gate:
        res = {}
        sample = clips[:GATE_N]
        for name in names:
            cache = f"{OUT}/_gate_{name}.npz"
            if os.path.exists(cache):
                os.remove(cache)
            try:
                extractor(name)(sample, cache)
                got, ref = np.load(cache), np.load(PUBLISHED[name])
                worst, where = 0.0, ""
                for k in got.files:
                    if k not in ref.files:
                        continue
                    d = float(np.abs(got[k].astype(np.float64)
                                     - ref[k][:GATE_N].astype(np.float64)).max())
                    if d > worst:
                        worst, where = d, k
                res[name] = {"max_abs_delta": worst, "worst_key": where,
                             "passes": bool(worst <= GATE_TOL)}
                p(f"  gate {name:14s} max |delta| {worst:.3e} on {where} -> "
                  f"{'PASS' if worst <= GATE_TOL else 'FAIL'}")
            except Exception as e:                                             # noqa: BLE001
                res[name] = {"error": repr(e)[:200], "passes": False}
                p(f"  gate {name:14s} ERROR {e!r}")
        res["_all_pass"] = all(v.get("passes") for k, v in res.items() if not k.startswith("_"))
        res["_tolerance"] = GATE_TOL
        res["_n_clips"] = GATE_N
        json.dump(res, open(f"{OUT}/gate.json", "w"), indent=2)
        p(f"gate summary: all pass {res['_all_pass']} (tolerance {GATE_TOL})")

    if a.extract:
        rev = reversed_clips(clips)
        assert rev[0]["paths"][0] == clips[0]["paths"][-1], "reversal did not take"
        for name in names:
            cache = f"{OUT}/{name}_windows_reversed.npz"
            if os.path.exists(cache):
                p(f"  {name}: cache present, skipping"); continue
            p(f"  {name}: extracting {len(rev)} reversed clips")
            extractor(name)(rev, cache)
            p(f"  {name}: wrote {cache}")
    logf.close()


if __name__ == "__main__":
    main()
