"""Cluster bootstrap and permutation inference for the peak-task families.

Usage:  CUDA_VISIBLE_DEVICES="" python -m affectdyn.inference.peak_cluster_inference
"""
from __future__ import annotations
import os

N_THREADS = 6
os.environ["CUDA_VISIBLE_DEVICES"] = ""                     # Disable CUDA before importing torch.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_THREADS))

import argparse, datetime, json, subprocess, sys, time, warnings

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof, N_SEED
from affectdyn.provenance import audit_loso

NPERM = 20000
NBOOT = 10000
ALPHA = 0.05
SEED = 20260911
TOL = 0.005                        # Bound AUC differences between CPU refits and logged fits.
AW2_VIDEO = ("vjepa", "videomae", "videomaev2", "timesformer", "xclip", "internvideo2")


def torch_cpu_only():
    import torch
    if torch.version.cuda is not None:
        raise SystemExit(f"refusing to run: torch {torch.__version__} is built with CUDA "
                         f"{torch.version.cuda}; a CPU-only build is required so the GPU cannot be touched")
    if torch.cuda.is_available():
        raise SystemExit("refusing to run: a CUDA device is visible to this process")
    torch.set_num_threads(N_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch


def auc_ranks(s):
    """Return historical ordinal ranks, including distinct ranks for tied scores."""
    o = np.argsort(s)                                        # Match the cross-pretraining estimator's sort order.
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return r


def perm_test(s, y, groups, unit, rng, nperm):
    """Vectorized two-sided label-permutation test on absolute AUC departure from 0.5; unit selects
    global clip permutations or permutations within clusters.
    """
    y = y.astype(np.int64); n = len(y); r = auc_ranks(s)
    n1 = int(y.sum()); n0 = n - n1
    obs = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    thr = abs(obs - 0.5) - 1e-12
    codes = np.unique(groups, return_inverse=True)[1].astype(np.float64)
    slot = np.argsort(codes, kind="stable")                  # Group positions by subject.
    batch = max(1, min(nperm, 4_000_000 // n))
    cnt = done = 0
    while done < nperm:
        b = min(batch, nperm - done)
        u = rng.random((b, n))                               # Offsets in [0, 1) preserve subject-code boundaries.
        if unit == "clip":
            yp = y[np.argsort(u, axis=1)]
        else:
            yp = np.empty((b, n), np.int64)
            yp[:, slot] = y[np.argsort(codes[None, :] + u, axis=1)]
        a = (yp @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
        cnt += int((np.abs(a - 0.5) >= thr).sum())
        done += b
    return float(obs), float((cnt + 1) / (nperm + 1))


def _members(groups):
    codes = np.unique(groups, return_inverse=True)[1]
    return [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]


def bootstrap(s, y, groups, unit, rng, nboot):
    """Percentile bootstrap interval and two-sided bootstrap p value against AUC=0.5; resample clips or
    whole supplied clusters while holding fitted scores fixed.
    """
    y = y.astype(np.int64); vals = []
    if unit == "clip":
        n = len(y)
        for _ in range(nboot):
            i = rng.integers(0, n, n)
            if y[i].min() == y[i].max():
                continue
            vals.append(_auc(s[i], y[i]))
    else:
        mem = _members(groups); G = len(mem)
        for _ in range(nboot):
            i = np.concatenate([mem[g] for g in rng.integers(0, G, G)])
            if y[i].min() == y[i].max():
                continue
            vals.append(_auc(s[i], y[i]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return {"ci": [float(lo), float(hi)], "ci_halfwidth": float((hi - lo) / 2),
            "p_boot": float(min(1.0, max(p, 1.0 / len(v)))), "resamples_skipped": int(nboot - len(v))}


def diff_bootstrap(a, b, rng, nboot):
    """AUC(b) - AUC(a), resampling supplied clusters independently within each corpus."""
    (sa, ya, ga), (sb, yb, gb) = a, b
    ma, mb = _members(ga), _members(gb)
    obs = _auc(sb, yb) - _auc(sa, ya)
    vals = []
    for _ in range(nboot):
        ia = np.concatenate([ma[k] for k in rng.integers(0, len(ma), len(ma))])
        ib = np.concatenate([mb[k] for k in rng.integers(0, len(mb), len(mb))])
        if ya[ia].min() == ya[ia].max() or yb[ib].min() == yb[ib].max():
            continue
        vals.append(_auc(sb[ib], yb[ib]) - _auc(sa[ia], ya[ia]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return {"diff": float(obs), "ci": [float(lo), float(hi)],
            "p_boot": float(min(1.0, max(p, 1.0 / len(v)))), "resamples_skipped": int(nboot - len(v))}


def provenance(torch, args, nperm, nboot):
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:                                                      # noqa: BLE001
            return "unavailable"
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_threads_max": N_THREADS, "niceness": os.nice(0),
        "seed": SEED, "nperm": nperm, "nboot": nboot, "smoke": False,
        "probe_seeds": {"unbc": [2000 + r for r in range(N_REP)],
                        "aw2_image": [100 + r for r in range(N_SEED)],
                        "aw2_video": [300 + r for r in range(N_REP)]},
        "note": ("CPU refit of the logged probes on cached features. The logged Aff-Wild2 fits ran on "
                 "CUDA, so small AUC differences there are expected; they are measured under "
                 "'reproduction' rather than assumed away."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/peak_cluster_inference")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    od = os.path.normpath(args.out_dir)
    for forbidden in ("runs/pain_encoder_maps", "runs/affwild_clips_features"):
        if od == forbidden or od.startswith(forbidden + os.sep):
            raise SystemExit(f"refusing to write under {forbidden}: the manuscript is validated against it")
    os.makedirs(od, exist_ok=True)
    out_json, out_npz, out_log = (os.path.join(od, f) for f in
                                  ("cluster.json", "cluster_oof_scores.npz", "cluster.log"))
    if not args.force and (os.path.exists(out_json) or os.path.exists(out_npz)):
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")

    torch = torch_cpu_only()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    rng = np.random.default_rng(SEED)
    nperm, nboot = (NPERM, NBOOT)
    want = lambda corpus, name: True

    logf = open(out_log, "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"W7 cluster-robust inference | torch {torch.__version__} cuda_build={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()} niceness={os.nice(0)} smoke=False")

    # Match the UNBC shape-fitting procedure.
    torch.set_num_threads(1)                                  # Reduce threading overhead for small matrices.
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    u_sub, u_traj = lab["subject"], lab["traj"]
    u_y = (u_traj == 2).astype(int); u_mask = u_traj != -1
    folds = _loso_folds(u_sub); audit_loso(u_sub, folds)
    IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    u_X = {}
    for e in IMG_ENCODERS:
        for k in KS:
            name = f"{e}_K{k}"
            if want("unbc", name):
                assert len(IMG[e]) == len(u_sub)
                u_X[name] = image_readout(IMG[e], k)
    for vname, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            key, name = tmpl.format(k=k), f"{vname}_K{k}"
            if key in z.files and want("unbc", name):
                assert len(z["full"]) == len(u_sub)
                u_X[name] = video_readout(z["full"], z[key])
    log(f"UNBC: {len(u_X)} cells, {int(u_y[u_mask].sum())} peaked of {int(u_mask.sum())} clips")
    u_scores = {}
    for name, X in u_X.items():
        acc = None
        for rep in range(N_REP):
            torch.manual_seed(2000 + rep)
            s = _oof_scores(X, u_y, u_sub, u_mask, folds)
            acc = s if acc is None else acc + s
        u_scores[name] = acc / N_REP
        log(f"  UNBC refit {name}")
    u_keep = u_mask.copy()
    for s in u_scores.values():
        u_keep &= ~np.isnan(s)

    # Match the Aff-Wild2 image-probe procedure.
    torch.set_num_threads(N_THREADS)
    alab = np.load("runs/affwild_clips_features/labels.npz")
    a_sub, a_traj = alab["subject"], alab["traj"]
    a_y = (a_traj == 2).astype(int); a_mask = a_traj != -1
    a_assign = grouped_folds(a_sub)
    frames = np.load("runs/affwild_clips_features/img_frames.npz")
    ai_scores = {}
    for e in IMG_ENCODERS:
        names = [f"{e}_K{k}" for k in KS if want("aw2_image", f"{e}_K{k}")]
        if not names:
            continue
        fr = frames[e]; assert len(fr) == len(a_sub)
        for name in names:
            X = image_readout(fr, int(name.rsplit("_K", 1)[1]))
            ai_scores[name] = np.nanmean([oof(X, a_y, a_assign, a_mask, 100 + r, "cpu")
                                          for r in range(N_SEED)], axis=0)
            del X
            log(f"  Aff-Wild2 image refit {name}")
        del fr

    # Match the Aff-Wild2 video-probe procedure.
    torch.set_num_threads(2)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    v_sub, v_traj = alab["subject"][idx], alab["traj"][idx]
    v_y = (v_traj == 2).astype(int); v_mask = v_traj != -1
    v_assign = grouped_folds(v_sub)
    av_scores = {}
    for vname in AW2_VIDEO:
        name = f"{vname}_K16"
        path = f"runs/affwild_clips_features/{vname}_windows.npz"
        if not want("aw2_video", name) or not os.path.exists(path):
            continue
        z = np.load(path)
        X = video_readout(z["full"], z["K16_up"]); assert len(X) == len(v_sub)
        av_scores[name] = np.nanmean([oof(X, v_y, v_assign, v_mask, 300 + r, "cpu")
                                      for r in range(N_REP)], axis=0)
        log(f"  Aff-Wild2 video refit {name}")

    L_unbc = {r["cell"]: r for r in json.load(open("runs/pain_encoder_maps/shape.json"))["rows"]}
    L_img = {f"{r['encoder']}_K{r['K']}": r for r in json.load(open("runs/affwild_clips_features/probe.json"))["shape"]["rows"]}
    L_vid = {f"{r['encoder']}_K16": r for r in json.load(open("runs/affwild_clips_features/video_probe.json"))["shape"]}

    # Apply per-cell inference to identical scores.
    corpora = {
        "unbc": (u_scores, u_y, u_keep, u_sub, L_unbc),
        "aw2_image": (ai_scores, a_y, a_mask, a_sub, L_img),
        "aw2_video": (av_scores, v_y, v_mask, v_sub, L_vid),
    }
    cells, kept = [], {}
    for corpus, (scores, y, mask, groups, logged) in corpora.items():
        for name, s in scores.items():
            keep = mask & ~np.isnan(s)
            sk, yk, gk = s[keep], y[keep].astype(np.int64), groups[keep]
            kept[(corpus, name)] = (sk, yk, gk)
            auc = _auc(sk, yk)
            L = logged[name]
            _, p_perm_clip = perm_test(sk, yk, gk, "clip", rng, nperm)
            _, p_perm_subj = perm_test(sk, yk, gk, "subject", rng, nperm)
            b_clip = bootstrap(sk, yk, gk, "clip", rng, nboot)
            b_subj = bootstrap(sk, yk, gk, "subject", rng, nboot)
            cells.append({
                "corpus": corpus, "cell": name, "family": "unbc" if corpus == "unbc" else "aw2",
                "n": int(len(yk)), "n_pos": int(yk.sum()),
                "n_subjects": int(len(np.unique(gk))), "n_subjects_with_pos": int(len(np.unique(gk[yk == 1]))),
                "auc": float(auc), "auc_logged": float(L["auc"]), "auc_delta": float(auc - L["auc"]),
                "p_logged_perm_clip": float(L["p"]),
                "clip": {"p_perm": p_perm_clip, **b_clip},
                "subject": {"p_perm": p_perm_subj, **b_subj},
                "ci_width_ratio_subject_over_clip": float(b_subj["ci_halfwidth"] / b_clip["ci_halfwidth"]),
            })
            log(f"  {corpus:9s} {name:18s} AUC {auc:.4f} (logged {L['auc']:.4f}) | "
                f"CI clip {b_clip['ci'][0]:.3f}-{b_clip['ci'][1]:.3f} "
                f"subject {b_subj['ci'][0]:.3f}-{b_subj['ci'][1]:.3f}")

    # Apply Holm within each pooled corpus family.
    for fam in ("unbc", "aw2"):
        fc = [c for c in cells if c["family"] == fam]
        for key, get in (("logged_perm_clip", lambda c: c["p_logged_perm_clip"]),
                         ("boot_clip", lambda c: c["clip"]["p_boot"]),
                         ("boot_subject", lambda c: c["subject"]["p_boot"]),
                         ("perm_subject", lambda c: c["subject"]["p_perm"])):
            for c, a in zip(fc, holm([get(c) for c in fc])):
                c.setdefault("holm_pooled", {})[key] = float(a)
        for c in fc:
            h = c["holm_pooled"]
            c["decision"] = {
                "logged": bool(h["logged_perm_clip"] < ALPHA),
                "cluster_boot": bool(h["boot_subject"] < ALPHA),
                "unit_effect_boot_clip": bool(h["boot_clip"] < ALPHA),
                "direction": "above_chance" if c["auc"] > 0.5 else "below_chance",
            }
            c["decision"]["changed"] = c["decision"]["logged"] != c["decision"]["cluster_boot"]

    # Compare corpus AUCs at matched granularities.
    pairs = []
    for name in sorted(set(u_scores) & (set(ai_scores) | set(av_scores))):
        other = "aw2_image" if name in ai_scores else "aw2_video"
        d = diff_bootstrap(kept[("unbc", name)], kept[(other, name)], rng, nboot)
        pairs.append({"cell": name, "unbc_auc": float(_auc(*kept[("unbc", name)][:2])),
                      "aw2_auc": float(_auc(*kept[(other, name)][:2])), **d})
        log(f"  cross-corpus {name:18s} diff {d['diff']:+.3f} CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}")
    for p, a in zip(pairs, holm([p["p_boot"] for p in pairs])):
        p["p_holm"] = float(a); p["excludes_zero_after_holm"] = bool(a < ALPHA)

    rep = {}
    for corpus in corpora:
        ds = [abs(c["auc_delta"]) for c in cells if c["corpus"] == corpus]
        if ds:
            rep[corpus] = {"n_cells": len(ds), "max_abs_auc_delta": float(max(ds)),
                           "within_tolerance": bool(max(ds) <= TOL)}
    changed = [{"cell": f"{c['corpus']}:{c['cell']}", "auc": c["auc"], "direction": c["decision"]["direction"],
                "logged_holm": c["holm_pooled"]["logged_perm_clip"],
                "cluster_holm": c["holm_pooled"]["boot_subject"],
                "logged_sig": c["decision"]["logged"], "cluster_sig": c["decision"]["cluster_boot"]}
               for c in cells if c["decision"]["changed"]]
    ratios = {fam: float(np.median([c["ci_width_ratio_subject_over_clip"] for c in cells if c["family"] == fam]))
              for fam in ("unbc", "aw2") if any(c["family"] == fam for c in cells)}

    out = {
        "provenance": provenance(torch, args, nperm, nboot),
        "criterion": ("Pre-declared: under pooled-per-corpus Holm (UNBC 36, Aff-Wild2 18), a logged "
                      "conclusion survives if replacing its clip permutation p with the subject-level "
                      "cluster-bootstrap p leaves its decision at alpha 0.05 unchanged, in either direction."),
        "reproduction": {"tolerance": TOL, **rep},
        "summary": {"n_cells": len(cells), "n_decisions_changed": len(changed), "changed": changed,
                    "median_ci_width_ratio_subject_over_clip": ratios,
                    "cross_corpus_pairs_excluding_zero": [p["cell"] for p in pairs if p["excludes_zero_after_holm"]]},
        "cells": cells, "cross_corpus": pairs,
    }
    json.dump(out, open(out_json, "w"), indent=2)
    arrays = {"unbc__subject": u_sub, "unbc__y": u_y, "unbc__keep": u_keep,
              "aw2img__subject": a_sub, "aw2img__y": a_y, "aw2img__mask": a_mask,
              "aw2vid__subject": v_sub, "aw2vid__y": v_y, "aw2vid__mask": v_mask, "aw2vid__idx": idx}
    arrays.update({f"unbc__{k}": v for k, v in u_scores.items()})
    arrays.update({f"aw2img__{k}": v for k, v in ai_scores.items()})
    arrays.update({f"aw2vid__{k}": v for k, v in av_scores.items()})
    np.savez_compressed(out_npz, **arrays)

    log(f"reproduction: {rep}")
    log(f"decisions changed under cluster bootstrap: {len(changed)} of {len(cells)}")
    for c in changed:
        log(f"   {c}")
    log(f"median CI width ratio subject/clip: {ratios}")
    log(f"wrote {out_json} and {out_npz}")
    logf.close()


if __name__ == "__main__":
    main()
