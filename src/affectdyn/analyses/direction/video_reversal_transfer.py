"""Reversal control for the video encoders."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import json, os, sys, time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # Reserve the GPU for extraction.

from affectdyn.features.unbc_pretraining_comparison import _auc                                         # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                   # noqa: E402
from affectdyn.features.temporal_aggregation import video_readout, KS, N_REP                         # noqa: E402
from affectdyn.analyses.peak.pain_peak_encoder_map import boot_auc_ci                                         # noqa: E402
from affectdyn.analyses.direction.image_reversal_transfer import oof_transfer                                     # noqa: E402
from affectdyn.features.reversed_video_features import PUBLISHED                                 # noqa: E402
from affectdyn.provenance import run_provenance, audit_loso                              # noqa: E402

OUT = "runs/video_reversal_transfer"


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    import torch
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/reversal_video.log", "w")
    t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int)
    mask = lab["direction"] != -1
    folds = _loso_folds(sub)
    audit_loso(sub, folds)
    published = json.load(open("runs/pain_encoder_maps/map.json"))["table"]

    have = {n: f"{OUT}/{n}_windows_reversed.npz" for n in PUBLISHED}
    missing = [n for n, p in have.items() if not os.path.exists(p)]
    if missing:
        log(f"reversed caches missing for {', '.join(missing)}; run --extract first")
    names = [n for n in PUBLISHED if n not in missing]
    if not names:
        raise SystemExit("no reversed caches present")

    log(f"W25 analysis | {int(mask.sum())} direction-labelled clips, {len(set(sub.tolist()))} "
        f"subjects | encoders {', '.join(names)} | {N_REP} seeded fits | CPU")
    log(f"  {'encoder':14s} {'K':>3s} {'ordered':>9s} {'reversed':>9s} {'95% CI (reversed)':>22s} "
        f"{'published':>10s}")

    rows = []
    for name in names:
        o, r = np.load(PUBLISHED[name]), np.load(have[name])
        for k in KS:
            key = f"K{k}_up"
            if key not in o.files or key not in r.files:
                continue
            Xo = video_readout(o["full"], o[key])
            Xr = video_readout(r["full"], r[key])
            so = sr = None
            for rep in range(N_REP):
                torch.manual_seed(1000 + rep)
                a = oof_transfer(Xo, Xo, y, sub, mask, folds)
                torch.manual_seed(1000 + rep)
                b = oof_transfer(Xo, Xr, y, sub, mask, folds)
                so = a if so is None else so + a
                sr = b if sr is None else sr + b
            so, sr = so / N_REP, sr / N_REP
            keep = mask & ~np.isnan(so) & ~np.isnan(sr)
            auc_o, auc_r = _auc(so[keep], y[keep]), _auc(sr[keep], y[keep])
            lo, hi = boot_auc_ci(sr[keep], y[keep])
            pub = published.get(f"{name}_K{k}", {}).get("pooled_oof")
            rows.append({"encoder": name, "K": k, "auc_ordered": float(auc_o),
                         "auc_reversed": float(auc_r), "ci_reversed": [float(lo), float(hi)],
                         "inverts": bool(hi < 0.5), "n": int(keep.sum()),
                         "published_pooled_oof": pub,
                         "delta_vs_published": (float(auc_o - pub) if pub is not None else None)})
            log(f"  {name:14s} {k:3d} {auc_o:9.3f} {auc_r:9.3f}      [{lo:.3f}, {hi:.3f}] "
                f"{(pub if pub is not None else float('nan')):10.3f}")

    rev = [x["auc_reversed"] for x in rows]
    inv = [x for x in rows if x["inverts"]]
    by_enc = {}
    for x in rows:
        by_enc.setdefault(x["encoder"], []).append(x)
    per_encoder = {e: {"n_cells": len(v), "n_inverting": sum(a["inverts"] for a in v),
                       "reversed_min": min(a["auc_reversed"] for a in v),
                       "reversed_max": max(a["auc_reversed"] for a in v)}
                   for e, v in by_enc.items()}
    dpub = [abs(x["delta_vs_published"]) for x in rows if x["delta_vs_published"] is not None]
    res = {"provenance": run_provenance(seeds=list(range(N_REP))), "n_rep": N_REP,
           "rows": rows, "per_encoder": per_encoder,
           "summary": {"n_cells": len(rows), "n_inverting": len(inv),
                       "reversed_span": [min(rev), max(rev)] if rev else None,
                       "image_encoder_reference": [0.090, 0.153],
                       "max_abs_delta_ordered_vs_published": (max(dpub) if dpub else None)},
           "status": "COMPLETE"}
    json.dump(res, open(f"{OUT}/reversal_video.json", "w"), indent=2)
    log("")
    log(f"reversed AUC spans {min(rev):.3f} to {max(rev):.3f} over {len(rows)} cells; "
        f"{len(inv)} of {len(rows)} have an interval entirely below one half")
    for e, v in per_encoder.items():
        log(f"  {e:14s} {v['n_inverting']}/{v['n_cells']} inverting, reversed AUC "
            f"{v['reversed_min']:.3f} to {v['reversed_max']:.3f}")
    log(f"ordered AUC against the published map: max |delta| "
        f"{(max(dpub) if dpub else float('nan')):.4f}")
    log(f"image-encoder reference for comparison: 0.090 to 0.153 reversed")
    log(f"wrote {OUT}/reversal_video.json")
    logf.close()


if __name__ == "__main__":
    main()
