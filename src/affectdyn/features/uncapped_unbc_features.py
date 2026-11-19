"""Feature extraction for the uncapped UNBC clip set.

Usage
  python -m affectdyn.features.uncapped_unbc_features --plan                 # clip counts and the subset map, no GPU
  python -m affectdyn.features.uncapped_unbc_features --extract [names]      # features into runs/uncapped_pain_peak_analysis
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, re, sys, time


import affectdyn.data.unbc_clip_pipeline as stage1_probe                                                            # noqa: E402

OUT = "runs/uncapped_pain_peak_analysis"
VIDEO = ("vjepa", "videomae", "videomaev2", "timesformer", "xclip", "internvideo2")


def build(maxper):
    """stage1_probe.build_clips with the cap raised; MAXPER is a module global it reads."""
    old = stage1_probe.MAXPER
    try:
        stage1_probe.MAXPER = maxper
        return stage1_probe.build_clips()
    finally:
        stage1_probe.MAXPER = old


def key(c):
    """A clip's identity independent of which build produced it."""
    first = c["paths"][0]
    m = re.search(r"(\d+)\.\w+$", os.path.basename(first))
    return (c["subject"], os.path.dirname(first), int(m.group(1)) if m else -1)


def plan():
    capped, uncapped = build(30), build(10 ** 9)
    idx = {key(c): i for i, c in enumerate(uncapped)}
    subset = [idx[key(c)] for c in capped]
    assert len(set(subset)) == len(capped), "published clips not uniquely located in the uncapped set"
    pk = lambda cs: sum(1 for c in cs if c.get("traj") == 2)                    # noqa: E731
    info = {"n_capped": len(capped), "n_uncapped": len(uncapped),
            "n_peaked_capped": pk(capped), "n_peaked_uncapped": pk(uncapped),
            "subjects_capped": len({c["subject"] for c in capped}),
            "subjects_uncapped": len({c["subject"] for c in uncapped}),
            "published_subset_positions": subset}
    return uncapped, info


def extractor(name):
    if name == "images":
        import affectdyn.features.temporal_aggregation as g0r
        return lambda clips, cache: g0r.extract_image_frames(clips, cache)
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


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("names", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/extract.log", "a")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    uncapped, info = plan()
    json.dump(info, open(f"{OUT}/clips.json", "w"), indent=2)
    p(f"W24 | uncapped {info['n_uncapped']} windows over {info['subjects_uncapped']} subjects, "
      f"{info['n_peaked_uncapped']} peaked")
    p(f"     | published subset {info['n_capped']} windows over {info['subjects_capped']} subjects, "
      f"{info['n_peaked_capped']} peaked, located uniquely inside it")

    if a.extract:
        names = a.names or (["images"] + list(VIDEO))
        for name in names:
            cache = f"{OUT}/img_frames.npz" if name == "images" else f"{OUT}/{name}_windows.npz"
            if os.path.exists(cache):
                p(f"  {name}: cache present, skipping"); continue
            p(f"  {name}: extracting {len(uncapped)} clips")
            extractor(name)(uncapped, cache)
            p(f"  {name}: wrote {cache}")
    logf.close()


if __name__ == "__main__":
    main()
