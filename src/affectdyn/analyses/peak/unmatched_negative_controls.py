"""Unmatched-negative control for the position-matched set.

Usage:  python -m affectdyn.analyses.peak.unmatched_negative_controls
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

N_THREADS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import argparse, json, sys, time                                                        # noqa: E402

import numpy as np                                                                      # noqa: E402

from affectdyn.features.unbc_pretraining_comparison import _auc                                                  # noqa: E402
from affectdyn.probes.affwild_image_fixed_step import grouped_folds                                                # noqa: E402
from affectdyn.analyses.peak.position_matched_peak_analysis import IMAGE, KS, VIDEO, W56_GRID                               # noqa: E402
from affectdyn.probes.affwild_image_fixed_step import oof                                                           # noqa: E402
import affectdyn.probes.nested_probe_fits as w51                                                     # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout                              # noqa: E402
from affectdyn.provenance import run_provenance                                                   # noqa: E402

OUT = "runs/unmatched_negative_controls"
SEED = 20260960
N_DRAWS = 20
FITS = ("fixed_step", "converged", "nested")
PRIMARY_FIT = "fixed_step"
MARGIN = 0.03
CLAIM_CELLS = {"dinov2_K2": "image", "vjepa_K16": "video"}
PUBLISHED = {"dinov2_K2": 0.594, "vjepa_K16": 0.622}
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRIT = "## 2026-09-23 — W60-W61 PRE-DECLARED CRITERIA"


def assemble(midx, lab, pub_pos, frames, vcache, vnew, want_video):
    X = {}
    for e in IMAGE:
        fr = frames[e][midx]
        for k in KS:
            X[f"{e}_K{k}"] = image_readout(fr, k)
    for e in want_video:
        cache, new = vcache[e], vnew[e]
        pos_new = {int(i): k for k, i in enumerate(new["idx"])} if new is not None else {}
        full = np.stack([cache["full"][pub_pos[i]] if i in pub_pos else new["full"][pos_new[i]] for i in midx])
        W = np.stack([cache["K16_up"][pub_pos[i]] if i in pub_pos else new["K16_up"][pos_new[i]] for i in midx])
        X[f"{e}_K16"] = video_readout(full, W)
    return X


def fit_scores(X, y, sub, mask, fit, image, audit=None):
    """stage_w59_analysis.fit_scores, with the optimality audit kept rather than discarded (G4)."""
    assign = grouped_folds(sub)
    if fit == "fixed_step":
        seeds = [100 + r for r in range(3)] if image else [300 + r for r in range(5)]
        return np.nanmean([oof(X, y, assign, mask, s, "cpu") for s in seeds], axis=0)
    grid = (1e-3,) if fit == "converged" else W56_GRID
    a = audit if audit is not None else {}
    for k_ in ("inner_splits", "max_inner_grad", "max_outer_grad", "outer_fits", "outer_uncertified"):
        a.setdefault(k_, 0 if "grad" not in k_ else 0.0)
    s, _ = w51.nested_scores(X, y, sub, mask, assign, grid, a, lambda m: None, "w60")
    return s


def aucs(X, y, g, cells, log, tag):
    """Pooled out-of-fold AUC of every cell under the 3 probe fits, with the G3 and G4 audits."""
    mask = np.ones(len(y), bool)
    out, audit = {}, {"outer_fits": 0, "outer_uncertified": 0, "inner_splits": 0,
                      "max_outer_grad": 0.0, "max_inner_grad": 0.0}
    nonfinite = []
    for fit in FITS:
        for c in cells:
            a = {}
            s = fit_scores(X[c], y, g, mask, fit, c.split("_K")[0] in IMAGE, a)
            if fit != "fixed_step":
                for k_ in ("outer_fits", "outer_uncertified", "inner_splits"):
                    audit[k_] += int(a.get(k_, 0))
                for k_ in ("max_outer_grad", "max_inner_grad"):
                    audit[k_] = max(audit[k_], float(a.get(k_, 0.0)))
            n_bad = int((~np.isfinite(s)).sum())
            if n_bad:
                nonfinite.append(f"{fit}:{c}={n_bad}")
            k = np.isfinite(s)
            out[f"{fit}:{c}"] = float(_auc(s[k], y[k]))
    out["_audit"] = {**audit, "n_scores_expected": int(len(y)), "cells_with_nonfinite_scores": nonfinite}
    log(f"  {tag}: {len(out) - 1} cell AUCs; certificates {audit}; nonfinite {nonfinite or 'none'}")
    return out


def merge():
    """Combine the shards into one output, the gates being identical in each and re-checked here."""
    import glob
    parts = sorted(glob.glob(f"{OUT}/size_prior_control_shard*of*.json"))
    if not parts:
        raise SystemExit("no shards to merge")
    R = [json.load(open(p)) for p in parts]
    seen = {d["draw"] for r in R for d in r["draws"]}
    for f in sorted(glob.glob(f"{OUT}/draw_*.json")):
        rec = json.load(open(f))
        if rec["draw"] not in seen:
            R[0]["draws"].append(rec); seen.add(rec["draw"])
    base = dict(R[0]); base.pop("shard", None)
    if len({json.dumps(r["gate_g2"], sort_keys=True) for r in R}) != 1:
        raise SystemExit("shards disagree on gate G2; refusing to merge")
    draws = sorted([d for r in R for d in r["draws"]], key=lambda d: d["draw"])
    if [d["draw"] for d in draws] != list(range(len(draws))):
        raise SystemExit(f"shards cover draws {[d['draw'] for d in draws]}, not a complete range")
    base["draws"] = draws
    base["gate_g3"] = {"passes": all(c["passes"] for d in draws for c in d["checks"].values())}
    base["merged_from"] = parts
    prev = {}
    for f in sorted(glob.glob("runs/unmatched_negative_controls_repeat/draw_*.json")):
        r = json.load(open(f)); prev[r["draw"]] = r
    if prev:
        deltas = [abs(d[k] - prev[d["draw"]][k]) for d in draws if d["draw"] in prev
                  for k in d if k.startswith(("fixed_step:", "converged:", "nested:"))]
        base["exact_replay_of_first_stream"] = {
            "n_compared": len(deltas), "max_abs_delta": max(deltas) if deltas else None,
            "passes": bool(deltas and max(deltas) <= 1e-12)}
    verdict = {}
    for cell in CLAIM_CELLS:
        key = f"{PRIMARY_FIT}:{cell}"
        vals = sorted(d[key] for d in draws if key in d)
        W59 = json.load(open("runs/position_matched_peak_analysis/analysis.json"))
        M = next(row["auc"] for row in W59["shape_family"][PRIMARY_FIT] if row["cell"] == cell)
        P = PUBLISHED[cell]; U = float(np.median(vals))
        inside = lambda x: vals[0] <= x <= vals[-1]                                     # noqa: E731
        verdict[cell] = {"published": P, "matched": M, "unmatched_median": U,
                         "unmatched_range": [vals[0], vals[-1]], "n_draws": len(vals),
                         "near_published": bool(abs(U - P) <= MARGIN and not inside(M)),
                         "near_matched": bool(abs(U - M) <= MARGIN and not inside(P))}
    if all(v["near_published"] for v in verdict.values()):
        outcome = "(a) the smaller sample and balanced prior do not account for the fall"
    elif all(v["near_matched"] for v in verdict.values()):
        outcome = "(b) they account for the fall"
    else:
        outcome = "(c) inconclusive"
    base["verdict"] = {"cells": verdict, "outcome": outcome}
    base["status"] = "COMPLETE"
    json.dump(base, open(f"{OUT}/size_prior_control.json", "w"), indent=1)
    print(f"merged {len(parts)} shards, {len(draws)} draws | {outcome}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    tag = "" if a.nshard == 1 else f"_shard{a.shard}of{a.nshard}"
    out_json = f"{OUT}/size_prior_control{tag}.json"
    if os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite")
    t0 = time.time(); logf = open(f"{OUT}/size_prior_control{tag}.log", "w")

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    rng = np.random.default_rng(SEED)
    lab = np.load("runs/affwild_clips_features/labels.npz"); traj = lab["traj"]
    midx = np.load("runs/position_matched_peak_analysis/matched_subset_idx.npy").astype(int)
    W59 = json.load(open("runs/position_matched_peak_analysis/analysis.json"))
    tested = [e for e, s in W59["encoder_status"].items() if s == "tested"]
    want_video = [e for e in VIDEO if e in tested]
    cells = [f"{e}_K{k}" for e in IMAGE for k in KS] + [f"{e}_K16" for e in want_video]
    pub = np.load("runs/affwild_clips_features/video_subset_idx.npy").astype(int)
    pub_pos = {int(i): k for k, i in enumerate(pub)}
    frames = np.load("runs/affwild_clips_features/img_frames.npz")
    vcache = {e: np.load(f"runs/affwild_clips_features/{e}_windows.npz") for e in want_video}
    vnew = {e: np.load(f"runs/position_matched_peak_analysis/{e}_ordered_new.npz") for e in want_video}

    pos_idx = midx[traj[midx] == 2]
    res = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRIT,
           "n_draws": N_DRAWS, "primary_fit": PRIMARY_FIT, "margin": MARGIN,
           "tested_encoders": tested, "cells": cells}

    w59_pos = np.sort(midx[traj[midx] == 2])
    res["gate_g1"] = {"n_positives": int(len(pos_idx)),
                      "passes": bool(np.array_equal(np.sort(pos_idx), w59_pos) and len(pos_idx) == 225)}
    log(f"G1 {res['gate_g1']}")

    y_m = (traj[midx] == 2).astype(int); g_m = lab["subject"][midx]
    rp = f"{OUT}/g2_replay.json"
    if os.path.exists(rp):
        replay = json.load(open(rp)); log("G2 replay loaded from cache")
    else:
        Xm = assemble(midx, lab, pub_pos, frames, vcache, vnew, want_video)
        replay = aucs(Xm, y_m, g_m, cells, log, "G2 replay")
        json.dump(replay, open(rp, "w"), indent=1)
    logged, deltas = {}, []
    for fit in FITS:
        for row in W59["shape_family"][fit]:
            if row.get("tested"):
                logged[f"{fit}:{row['cell']}"] = row["auc"]
    replay_audit = replay.pop("_audit", None)
    res["gate_g2_audit"] = replay_audit
    for k, v in logged.items():
        if k in replay:
            deltas.append(abs(replay[k] - v))
    res["gate_g2"] = {"n_compared": len(deltas), "max_abs_delta": max(deltas) if deltas else None,
                      "passes": bool(deltas and max(deltas) <= 1e-9)}
    log(f"G2 {res['gate_g2']}")
    if not (res["gate_g1"]["passes"] and res["gate_g2"]["passes"]):
        res["status"] = "HELD at a gate"
        json.dump(res, open(out_json, "w"), indent=1); log(res["status"]); return

    img_pool = np.flatnonzero((traj != 2) & (traj != -1))
    vid_pool = np.array([i for i in pub if traj[i] != 2 and traj[i] != -1], dtype=int)
    res["pools"] = {"image": int(len(img_pool)), "video": int(len(vid_pool))}
    log(f"pools: image {len(img_pool)}, video {len(vid_pool)}; cells {len(cells)}")
    n_draws = N_DRAWS
    seeds = [SEED + 1000 * d for d in range(n_draws)]
    res["draw_seeds"] = seeds
    draws = []
    res["shard"] = {"index": a.shard, "n": a.nshard}
    for d, sd in enumerate(seeds):
        if d % a.nshard != a.shard:
            continue
        one = f"{OUT}/draw_{d:02d}.json"
        if os.path.exists(one):                      # Resume from completed draws.
            draws.append(json.load(open(one))); log(f"draw {d} loaded from {one}"); continue
        r = np.random.default_rng(sd)
        neg_i = r.choice(img_pool, 225, replace=False)
        neg_v = r.choice(vid_pool, 225, replace=False)
        rec = {"draw": d, "seed": sd, "checks": {}}
        for fam, neg in (("image", neg_i), ("video", neg_v)):
            idx = np.concatenate([pos_idx, neg])
            y = (traj[idx] == 2).astype(int); g = lab["subject"][idx]
            assign = grouped_folds(g)
            ok = (len(neg) == 225 and len(set(neg.tolist())) == 225 and (traj[neg] != 2).all()
                  and len(idx) == 450
                  and all(0 < y[assign == k].sum() < (assign == k).sum() for k in range(assign.max() + 1)))
            rec["checks"][fam] = {"design_ok": bool(ok), "n_subjects": int(len(set(g.tolist()))),
                                  "n_folds": int(assign.max() + 1)}
            want = [c for c in cells if (c.split("_K")[0] in IMAGE) == (fam == "image")]
            X = assemble(idx, lab, pub_pos, frames, vcache, vnew,
                         [] if fam == "image" else want_video)
            A = aucs({c: X[c] for c in want}, y, g, want, log, f"draw {d} {fam}")
            au = A.pop("_audit")
            rec["checks"][fam].update(
                {"audit": au,
                 "scores_all_finite": not au["cells_with_nonfinite_scores"],
                 "all_fits_certified": au["outer_uncertified"] == 0,
                 "passes": bool(rec["checks"][fam]["design_ok"] and not au["cells_with_nonfinite_scores"]
                                and au["outer_uncertified"] == 0)})
            rec.update(A)
        draws.append(rec)
        json.dump(rec, open(f"{OUT}/draw_{d:02d}.json", "w"), indent=1)
        log(f"draw {d} done ({time.time() - t0:.0f}s)")
    res["draws"] = draws
    res["gate_g3"] = {"passes": all(c["passes"] for r in draws for c in r["checks"].values())}

    if a.nshard > 1:
        res["status"] = "SHARD COMPLETE"
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(out_json, "w"), indent=1)
        log(f"wrote {out_json} | {res['status']}"); return

    verdict = {}
    for cell, fam in CLAIM_CELLS.items():
        key = f"{PRIMARY_FIT}:{cell}"
        vals = sorted(r[key] for r in draws if key in r)
        M = next(row["auc"] for row in W59["shape_family"][PRIMARY_FIT] if row["cell"] == cell)
        P = PUBLISHED[cell]
        U = float(np.median(vals))
        inside = lambda x: vals[0] <= x <= vals[-1]                                      # noqa: E731
        verdict[cell] = {"published": P, "matched": M, "unmatched_median": U,
                         "unmatched_range": [vals[0], vals[-1]], "n_draws": len(vals),
                         "near_published": bool(abs(U - P) <= MARGIN and not inside(M)),
                         "near_matched": bool(abs(U - M) <= MARGIN and not inside(P))}
    if all(v["near_published"] for v in verdict.values()):
        outcome = "(a) the smaller sample and balanced prior do not account for the fall"
    elif all(v["near_matched"] for v in verdict.values()):
        outcome = "(b) they account for the fall"
    else:
        outcome = "(c) inconclusive"
    res["verdict"] = {"cells": verdict, "outcome": outcome}
    res["status"] = "COMPLETE"
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"verdict {outcome}")
    log(f"wrote {out_json} | {res['status']}")


if __name__ == "__main__":
    main()
