"""Pain peak task on the uncapped UNBC clip set."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, sys, time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # Reserve the GPU for extraction.

from affectdyn.features.unbc_pretraining_comparison import _auc                                         # noqa: E402
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds                      # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP  # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                                 # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import KS                                                    # noqa: E402
from affectdyn.features.uncapped_unbc_features import VIDEO, build                            # noqa: E402
from affectdyn.provenance import run_provenance                                          # noqa: E402

OUT = "runs/uncapped_pain_peak_analysis"
SEED = 20260916
NBOOT = 10_000
N_RANDOM = 5
CAP = 30
GATE_TOL = 1e-3


def subject_bootstrap(s, y, g, rng, nboot=NBOOT):
    """Two-sided p against one half, subjects resampled with their clips kept together."""
    subs = np.unique(g)
    idx = {u: np.flatnonzero(g == u) for u in subs}
    vals = []
    for _ in range(nboot):
        draw = rng.choice(len(subs), size=len(subs), replace=True)
        take = np.concatenate([idx[subs[j]] for j in draw])
        yb = y[take]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(_auc(s[take], yb))
    v = np.asarray(vals, float)
    share = min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return {"p_boot": max(2.0 * share, 1.0 / max(1, len(v))),
            "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "n_resamples_used": int(len(v))}


def oof_reps(X, y, sub, mask, folds, n_rep=N_REP):
    """The offset is 2000, not the 1000 the DIRECTION map uses."""
    import torch
    out = []
    for rep in range(n_rep):
        torch.manual_seed(2000 + rep)
        out.append(_oof_scores(X, y, sub, mask, folds))
    return out


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="subset,uncapped,random")
    ap.add_argument("--encoders", default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/cap_analysis.log", "a")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    clips = build(10 ** 9)
    sub_all = np.array([c["subject"] for c in clips])
    traj = np.array([c["traj"] for c in clips])
    y_all = (traj == 2).astype(int)
    mask_all = traj != -1
    info = json.load(open(f"{OUT}/clips.json"))
    subset_pos = np.array(info["published_subset_positions"])
    rng = np.random.default_rng(SEED)

    # Build cells from the uncapped extraction.
    want = None if a.encoders == "all" else set(a.encoders.split(","))
    cells = {}
    imgp = f"{OUT}/img_frames.npz"
    if os.path.exists(imgp):
        IMG = np.load(imgp)
        for e in IMG_ENCODERS:
            if want and e not in want:
                continue
            for k in KS:
                cells[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for name in VIDEO:
        if want and name not in want:
            continue
        pth = f"{OUT}/{name}_windows.npz"
        if not os.path.exists(pth):
            continue
        z = np.load(pth)
        for k in KS:
            kk = f"K{k}_up"
            if kk in z.files:
                cells[f"{name}_K{k}"] = video_readout(z["full"], z[kk])
    if not cells:
        raise SystemExit("no features present yet")
    p(f"W24 analysis | {len(clips)} uncapped windows, {int(y_all.sum())} peaked | "
      f"{len(cells)} cells | arms {a.arms} | CPU")

    def rand_subset(r):
        keep = []
        g = np.random.default_rng(SEED + 100 + r)
        for s in np.unique(sub_all):
            pos = np.flatnonzero(sub_all == s)
            keep.extend(pos if len(pos) <= CAP else g.choice(pos, CAP, replace=False))
        return np.sort(np.array(keep))

    arms = {}
    for nm in a.arms.split(","):
        if nm == "subset":
            arms["subset"] = subset_pos
        elif nm == "uncapped":
            arms["uncapped"] = np.arange(len(clips))
        elif nm == "random":
            for r in range(N_RANDOM):
                arms[f"random{r}"] = rand_subset(r)

    published = {r["cell"]: r for r in json.load(open("runs/pain_encoder_maps/shape.json"))["rows"]}
    res = {"provenance": run_provenance(seeds=[SEED]), "seed": SEED, "nboot": NBOOT,
           "cap": CAP, "n_random": N_RANDOM, "arms": {}}

    for nm, sel in arms.items():
        sub, y, mask = sub_all[sel], y_all[sel], mask_all[sel]
        folds = _loso_folds(sub)
        scores = {c: np.mean(oof_reps(X[sel], y, sub, mask, folds), axis=0)
                  for c, X in cells.items()}
        keep = mask.copy()                       # Use one clip set for every cell.
        for s in scores.values():
            keep &= ~np.isnan(s)
        rows = []
        for cname, s in scores.items():
            auc = float(_auc(s[keep], y[keep]))
            b = subject_bootstrap(s[keep], y[keep], sub[keep], rng)
            rows.append({"cell": cname, "auc": auc, **b})
        for r, adj in zip(rows, holm([x["p_boot"] for x in rows])):
            r["p_holm"] = float(adj)
            r["significant"] = bool(adj < 0.05)
        nsig = sum(r["significant"] for r in rows)
        entry = {"n_clips": int(len(sel)), "n_scored": int(keep.sum()),
                 "n_peaked": int(y[keep].sum()),
                 "n_subjects": int(len(np.unique(sub))), "n_significant": nsig,
                 "significant_cells": [r["cell"] for r in rows if r["significant"]],
                 "max_auc": max(r["auc"] for r in rows), "rows": rows}
        if nm == "subset":
            d = [(r["cell"], abs(r["auc"] - published[r["cell"]]["auc"]))
                 for r in rows if r["cell"] in published]
            worst = max(d, key=lambda x: x[1]) if d else ("", float("nan"))
            entry["gate"] = {"compared": len(d), "max_abs_delta_auc": worst[1],
                             "worst_cell": worst[0], "tolerance": GATE_TOL,
                             "passes": bool(d and worst[1] <= GATE_TOL)}
            p(f"  GATE subset vs runs/pain_encoder_maps/shape.json: compared {len(d)}, max |delta| "
              f"{worst[1]:.5f} on {worst[0]} -> {'PASS' if entry['gate']['passes'] else 'FAIL'}")
        res["arms"][nm] = entry
        p(f"  {nm:10s} n={len(sel):5d} peaked={entry['n_peaked']:3d} subjects={entry['n_subjects']:3d} "
          f"| significant {nsig}/{len(rows)} | max AUC {entry['max_auc']:.3f}"
          + (f" | {', '.join(entry['significant_cells'])}" if nsig else ""))

    rnd = [v for k, v in res["arms"].items() if k.startswith("random")]
    res["summary"] = {
        "subset_significant": res["arms"].get("subset", {}).get("n_significant"),
        "uncapped_significant": res["arms"].get("uncapped", {}).get("n_significant"),
        "random_significant_range": ([min(v["n_significant"] for v in rnd),
                                      max(v["n_significant"] for v in rnd)] if rnd else None),
        "encoders_ever_significant": sorted({c for v in res["arms"].values()
                                             for c in v["significant_cells"]}),
        "n_cells": len(cells)}
    g = res["arms"].get("subset", {}).get("gate")
    res["status"] = ("COMPLETE" if (g is None or g["passes"]) else "COMPLETE WITH GATE MISMATCH")
    json.dump(res, open(f"{OUT}/cap_analysis.json", "w"), indent=2)
    p("")
    p(f"summary: {json.dumps(res['summary'])}")
    p(f"wrote {OUT}/cap_analysis.json | {res['status']}")
    logf.close()


if __name__ == "__main__":
    main()
