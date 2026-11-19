"""Cluster bootstrap for the 24 pain direction contrasts.

Usage:  CUDA_VISIBLE_DEVICES="" python -m affectdyn.inference.pain_direction_cluster_inference
"""
from __future__ import annotations
import os

N_THREADS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = ""                     # Disable CUDA before importing torch.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import argparse, datetime, json, subprocess, sys, time, warnings, zlib

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, video_readout, N_REP
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS
from affectdyn.inference.peak_cluster_inference import torch_cpu_only, bootstrap, _members
from affectdyn.provenance import audit_loso

from affectdyn.paths import REPO_STR as REPO
NBOOT = 10000
NPERM = 20000
SEED = 20260911
ALPHA = 0.05
TOL = 0.005
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-11 — W9 PRE-DECLARED CRITERIA"


def paired_bootstrap(sa, sb, y, groups, unit, rng, nboot):
    """AUC(a) - AUC(b) on identical clips, resampling clips or whole subjects, one resample for both.

    Scores remain fixed; probes are not refitted. Single-AUC and independent-corpus variants are in
    peak_cluster_inference. Related paired implementations also occur in
    analyses.peak.position_matched_peak_analysis and analyses.sensitivity.fixed_step_penalty_sweep.
    """
    mem = _members(groups); G = len(mem); n = len(y); vals = []
    for _ in range(nboot):
        i = rng.integers(0, n, n) if unit == "clip" else np.concatenate([mem[g] for g in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(_auc(sa[i], y[i]) - _auc(sb[i], y[i]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return {"ci": [float(lo), float(hi)], "ci_halfwidth": float((hi - lo) / 2),
            "p_boot": float(min(1.0, max(p, 1.0 / len(v)))), "resamples_skipped": int(nboot - len(v))}


def subject_swap_perm(sa, sb, y, groups, rng, nperm):
    """Two-sided paired AUC score-swap test, drawing one swap decision per cluster."""
    codes = np.unique(groups, return_inverse=True)[1]; G = int(codes.max()) + 1
    obs = _auc(sa, y) - _auc(sb, y); cnt = 0
    for _ in range(nperm):
        flip = (rng.random(G) < 0.5)[codes]
        pa, pb = np.where(flip, sb, sa), np.where(flip, sa, sb)
        if abs(_auc(pa, y) - _auc(pb, y)) >= abs(obs) - 1e-12:
            cnt += 1
    return float((cnt + 1) / (nperm + 1))


def criteria_verbatim():
    log = os.path.join(REPO, "RESEARCH_LOG-D3-W1.md")
    if not os.path.exists(log):   # Allow the optional criteria file to be absent.
        return None
    lines = open(log, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(CRITERIA_HEADING))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    return "\n".join(lines[start:end]).strip()


def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip()
    except Exception:                                                          # noqa: BLE001
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/pain_direction_cluster_inference")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit(f"run from the repository root {REPO}")
    od = os.path.abspath(args.out_dir)
    for forbidden in ("runs/pain_encoder_maps", "runs/affwild_clips_features"):
        f = os.path.join(REPO, forbidden)
        if od == f or od.startswith(f + os.sep):
            raise SystemExit(f"refusing to write under {forbidden}: the manuscript is validated against it")
    os.makedirs(od, exist_ok=True)
    out_json, out_npz, out_log = (os.path.join(od, f) for f in
                                  ("direction_cluster.json", "direction_cluster_oof.npz", "direction_cluster.log"))
    if not args.force and (os.path.exists(out_json) or os.path.exists(out_npz)):
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")

    torch = torch_cpu_only()
    torch.set_num_threads(1)                                  # Single-threaded execution reduces overhead for 219 clips.
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    nboot, nperm = (NBOOT, NPERM)
    criteria = criteria_verbatim()
    logf = open(out_log, "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"W9 direction contrasts, subject-level | torch {torch.__version__} cuda_build={torch.version.cuda} "
        f"niceness={os.nice(0)} smoke=False")

    # Match the direction-mapping refit procedure.
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int)
    mask = lab["direction"] != -1
    folds = _loso_folds(sub); audit_loso(sub, folds)
    IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    cells = {f"clip_K{k}": image_readout(IMG["clip"], k) for k in KS}
    for name, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            key = tmpl.format(k=k)
            if key in z.files:
                cells[f"{name}_K{k}"] = video_readout(z["full"], z[key])
    pooled = {}
    for name, X in cells.items():
        assert len(X) == len(sub)
        reps = []
        for rep in range(N_REP):
            torch.manual_seed(1000 + rep)
            reps.append(_oof_scores(X, y, sub, mask, folds))
        pooled[name] = np.mean(reps, axis=0)
        log(f"  refit {name:18s} dim {X.shape[1]:5d}")
    keep = mask.copy()
    for s in pooled.values():
        keep &= ~np.isnan(s)
    yk, gk = y[keep].astype(np.int64), sub[keep]
    log(f"clips kept {int(keep.sum())}, positives {int(yk.sum())}, subjects {len(np.unique(gk))}")

    M = json.load(open("runs/pain_encoder_maps/map.json"))
    cell_rep = {n: {"auc": float(_auc(s[keep], yk)), "logged": float(M["table"][n]["pooled_oof"])} for n, s in pooled.items()}
    for v in cell_rep.values():
        v["abs_delta"] = abs(v["auc"] - v["logged"])
    logged_rows = [r for r in M["primary"] if r["a"] in pooled and r["b"] in pooled]
    gap_rep = []
    for r in logged_rows:
        g = float(_auc(pooled[r["a"]][keep], yk) - _auc(pooled[r["b"]][keep], yk))
        gap_rep.append({"label": r["label"], "gap": g, "logged": float(r["gap"]), "abs_delta": abs(g - r["gap"])})
    gate = {"tolerance": TOL, "n_clips": int(keep.sum()), "n_clips_logged": int(M["n_clips"]),
            "max_abs_auc_delta": max(v["abs_delta"] for v in cell_rep.values()),
            "max_abs_gap_delta": max(v["abs_delta"] for v in gap_rep)}
    gate["passes"] = bool(gate["n_clips"] == gate["n_clips_logged"] and gate["max_abs_auc_delta"] <= TOL
                          and gate["max_abs_gap_delta"] <= TOL
                          and (len(cell_rep) == 28 and len(gap_rep) == 24))
    log(f"pipeline gate: {gate}")

    def rng_for(name):
        return np.random.default_rng([SEED, zlib.crc32(name.encode())])

    # Apply contrast inference to identical scores.
    rows = []
    for r in logged_rows:
        sa, sb = pooled[r["a"]][keep], pooled[r["b"]][keep]
        b_subj = paired_bootstrap(sa, sb, yk, gk, "subject", rng_for(r["label"] + ":subject"), nboot)
        b_clip = paired_bootstrap(sa, sb, yk, gk, "clip", rng_for(r["label"] + ":clip"), nboot)
        p_swap = subject_swap_perm(sa, sb, yk, gk, rng_for(r["label"] + ":swap"), nperm)
        rows.append({"label": r["label"], "a": r["a"], "b": r["b"],
                     "auc_a": float(_auc(sa, yk)), "auc_b": float(_auc(sb, yk)),
                     "gap": float(_auc(sa, yk) - _auc(sb, yk)),
                     "logged": {"gap": float(r["gap"]), "p_perm_clip": float(r["p"]), "p_holm": float(r["p_holm"]),
                                "ci_clip_boot": r["ci"]},
                     "subject_boot": b_subj, "clip_boot": b_clip, "subject_swap_perm_p": p_swap,
                     "ci_width_ratio_subject_over_clip": float(b_subj["ci_halfwidth"] / b_clip["ci_halfwidth"])})
        log(f"  {r['label']:28s} gap {rows[-1]['gap']:+.4f} | subject CI {b_subj['ci'][0]:+.3f} to "
            f"{b_subj['ci'][1]:+.3f} p {b_subj['p_boot']:.4f} | clip CI {b_clip['ci'][0]:+.3f} to "
            f"{b_clip['ci'][1]:+.3f} | swap p {p_swap:.4f} | logged p {r['p']:.4f}")

    for key, get in (("logged_perm_clip", lambda c: c["logged"]["p_perm_clip"]),
                     ("boot_subject", lambda c: c["subject_boot"]["p_boot"]),
                     ("boot_clip", lambda c: c["clip_boot"]["p_boot"]),
                     ("perm_subject_swap", lambda c: c["subject_swap_perm_p"])):
        for c, a in zip(rows, holm([get(c) for c in rows])):
            c.setdefault("holm", {})[key] = float(a)
    for c in rows:
        h = c["holm"]
        c["decision"] = {"logged": bool(h["logged_perm_clip"] < ALPHA), "subject_boot": bool(h["boot_subject"] < ALPHA),
                         "clip_boot": bool(h["boot_clip"] < ALPHA), "subject_swap": bool(h["perm_subject_swap"] < ALPHA),
                         "favours": "image" if c["gap"] > 0 else "video"}
        c["decision"]["changed"] = c["decision"]["logged"] != c["decision"]["subject_boot"]
        c["holm_logged_matches_map"] = bool(abs(h["logged_perm_clip"] - c["logged"]["p_holm"]) < 1e-9)

    # Describe each cell's direction AUC with subject resampling.
    cell_rows = {n: {**cell_rep[n], "subject_boot": bootstrap(s[keep], yk, gk, "subject", rng_for(n + ":cell"), nboot)}
                 for n, s in pooled.items()}

    changed = [{"label": c["label"], "gap": c["gap"], "logged_holm": c["holm"]["logged_perm_clip"],
                "subject_holm": c["holm"]["boot_subject"]} for c in rows if c["decision"]["changed"]]
    survivors = [{"label": c["label"], "gap": c["gap"], "subject_holm": c["holm"]["boot_subject"],
                  "subject_ci": c["subject_boot"]["ci"]} for c in rows if c["decision"]["subject_boot"]]
    status = "COMPLETE" if gate["passes"] else "HOLD: pipeline gate failed, no decision is read"
    summary = ({"n_contrasts": len(rows), "n_significant_logged": sum(c["decision"]["logged"] for c in rows),
                "n_significant_subject_boot": len(survivors), "n_decisions_changed": len(changed),
                "changed": changed, "survivors_subject_boot": survivors,
                "n_significant_clip_boot": sum(c["decision"]["clip_boot"] for c in rows),
                "n_significant_subject_swap": sum(c["decision"]["subject_swap"] for c in rows),
                "median_ci_width_ratio_subject_over_clip": float(np.median([c["ci_width_ratio_subject_over_clip"] for c in rows]))}
               if gate["passes"] else None)

    out = {"provenance": {"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                          "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                          "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                          "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                          "niceness": os.nice(0), "seed": SEED, "nboot": nboot, "nperm": nperm, "smoke": False,
                          "probe_seeds": [1000 + r for r in range(N_REP)]},
           "criteria_verbatim": criteria, "status": status, "pipeline_gate": gate,
           "reproduction": {"cells": cell_rep, "gaps": gap_rep},
           "summary": summary, "contrasts": rows, "cells": cell_rows}
    json.dump(out, open(out_json, "w"), indent=2)
    np.savez_compressed(out_npz, subject=sub, y=y, keep=keep, **pooled)
    log(f"status {status}")
    if summary:
        log(f"significant: logged {summary['n_significant_logged']}, subject bootstrap "
            f"{summary['n_significant_subject_boot']}, clip bootstrap {summary['n_significant_clip_boot']}, "
            f"subject swap {summary['n_significant_subject_swap']} | changed {summary['n_decisions_changed']}")
        for c in changed:
            log(f"   changed {c}")
    log(f"wrote {out_json} and {out_npz}")
    logf.close()


if __name__ == "__main__":
    main()
