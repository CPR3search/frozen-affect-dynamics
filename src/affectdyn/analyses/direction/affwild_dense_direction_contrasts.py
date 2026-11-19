"""CLIP-video direction contrasts on Aff-Wild2 at K=16.

Usage:  python -m affectdyn.analyses.direction.affwild_dense_direction_contrasts
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""                    # Run on the CPU.
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.features.temporal_aggregation import image_readout, video_readout
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof
from affectdyn.provenance import run_provenance

OUT = "runs/affwild_dense_direction_contrasts"
SEED = 20260922
NBOOT = 10_000
ALPHA = 0.05
TOL = 0.005
K = 16
VIDEO = ("vjepa", "videomae", "videomaev2", "timesformer", "xclip", "internvideo2")
IMG_SEEDS, VID_SEEDS = range(100, 103), range(300, 305)
MASKED_AE = {"videomae", "videomaev2"}
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-16 — W32-W34 PRE-DECLARED CRITERIA"


def _members(groups):
    codes = np.unique(groups, return_inverse=True)[1]
    return [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]


def subject_bootstrap_vs_half(s, y, groups, rng, nboot):
    mem = _members(groups); G = len(mem); vals = []
    for _ in range(nboot):
        i = np.concatenate([mem[k] for k in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(_auc(s[i], y[i]))
    v = np.asarray(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return {"ci": [float(lo), float(hi)], "p_boot": float(min(1.0, max(p, 1.0 / len(v)))),
            "ci_contains_half": bool(lo <= 0.5 <= hi)}


def paired_bootstrap(sa, sb, y, groups, rng, nboot):
    mem = _members(groups); G = len(mem); vals = []
    for _ in range(nboot):
        i = np.concatenate([mem[g] for g in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(_auc(sa[i], y[i]) - _auc(sb[i], y[i]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return {"ci": [float(lo), float(hi)], "p_boot": float(min(1.0, max(p, 1.0 / len(v))))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/aw2_contrasts.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/aw2_contrasts.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import torch
    torch.set_num_threads(2)
    clips = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    lab = np.load("runs/affwild_clips_features/labels.npz", allow_pickle=True)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    subj_all = np.array([str(s) for s in lab["subject"]])
    d_all = lab["direction"]
    sub = subj_all[idx]
    y = (d_all[idx] == 1).astype(int)
    mask = d_all[idx] != -1
    rel = np.empty(len(clips))
    for s in np.unique(subj_all):
        ix = np.flatnonzero(subj_all == s)
        rel[ix] = np.arange(len(ix)) / max(1, len(ix) - 1)
    rng = np.random.default_rng(SEED)
    log(f"W34 | subset {len(idx)} clips, direction-labelled {int(mask.sum())}, onset {int(y[mask].sum())}, "
        f"subjects {len(np.unique(sub[mask]))} | seed {SEED}")

    # Evaluate position diagnostics for direction labels.
    pos = subject_bootstrap_vs_half(rel[idx][mask], y[mask].astype(np.int64), sub[mask], rng, NBOOT)
    pos["auc"] = _auc(rel[idx][mask], y[mask].astype(np.int64))
    pos["balanced"] = bool(pos["ci_contains_half"])
    log(f"position diagnostic (onset vs offset): AUC {pos['auc']:.4f} CI {pos['ci'][0]:.3f} to {pos['ci'][1]:.3f} "
        f"p {pos['p_boot']:.4f} balanced {pos['balanced']}")

    frames = np.load("runs/affwild_clips_features/img_frames.npz")
    X = {"clip": image_readout(frames["clip"], K)[idx]}
    for e in VIDEO:
        z = np.load(f"runs/affwild_clips_features/{e}_windows.npz")
        X[e] = video_readout(z["full"], z["K16_up"])
    assign = grouped_folds(sub)
    S, rows = {}, []
    published = {r["encoder"]: r["auc"] for r in json.load(open("runs/affwild_clips_features/video_probe.json"))["direction"]}
    for e, Xe in X.items():
        seeds = IMG_SEEDS if e == "clip" else VID_SEEDS
        s = np.nanmean([oof(Xe, y, assign, mask, sd, "cpu") for sd in seeds], axis=0)
        keep = mask & ~np.isnan(s)
        S[e] = (s, keep)
        auc = float(_auc(s[keep], y[keep].astype(np.int64)))
        rows.append({"encoder": e, "auc": auc, "published_auc": published.get(e), "n": int(keep.sum())})
        log(f"  {e:13s} direction AUC {auc:.4f}" + (f" (published {published[e]:.4f})" if e in published else ""))
    dmax = max(abs(r["auc"] - r["published_auc"]) for r in rows if r["published_auc"] is not None)
    gate = {"max_abs_auc_delta": dmax, "tolerance": TOL, "passes": bool(dmax <= TOL),
            "note": "CLIP's subset direction AUC has no published target and is ungated"}
    log(f"gate: {gate}")
    if not gate["passes"]:
        json.dump({"provenance": run_provenance(seeds=[SEED], device="cpu"), "gate": gate, "refits": rows,
                   "position_diagnostic": pos, "status": "HOLD: gate failed"}, open(out_json, "w"), indent=2)
        log("HOLD"); return

    # Evaluate the six contrasts.
    sc, kc = S["clip"]
    contrasts = []
    for e in VIDEO:
        se, ke = S[e]
        keep = kc & ke
        yk, gk = y[keep].astype(np.int64), sub[keep]
        b = paired_bootstrap(sc[keep], se[keep], yk, gk, rng, NBOOT)
        contrasts.append({"video": e, "auc_clip": float(_auc(sc[keep], yk)), "auc_video": float(_auc(se[keep], yk)),
                          "gap": float(_auc(sc[keep], yk) - _auc(se[keep], yk)), "n": int(keep.sum()), **b})
        log(f"  CLIP vs {e:13s} gap {contrasts[-1]['gap']:+.4f} CI {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f} "
            f"p {b['p_boot']:.4f}")
    for r, a in zip(contrasts, holm([x["p_boot"] for x in contrasts])):
        r["holm"] = a; r["survives"] = bool(a < ALPHA)
    surv = [r["video"] for r in contrasts if r["survives"]]
    outside = [v for v in surv if v not in MASKED_AE]
    verdict = {"n_survivors": len(surv), "survivors": surv, "survivors_outside_masked_ae": outside,
               "all_survivors_masked_ae": bool(surv and not outside),
               "position_diagnostic_balanced": pos["balanced"],
               "reading": ("survivors all against masked autoencoding supports the pain reading transferring; "
                           "survivors elsewhere or none surviving is reported as it falls"),
               "caveat": ("the subset's direction labels are position-balanced" if pos["balanced"] else
                          "the subset's direction labels are position-confounded, so every number here carries that caveat")}
    log(f"verdict: {verdict}")
    res = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "nboot": NBOOT, "position_diagnostic": pos, "gate": gate, "refits": rows, "contrasts": contrasts,
           "verdict": verdict, "status": "COMPLETE"}
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | COMPLETE")
    logf.close()


if __name__ == "__main__":
    main()
