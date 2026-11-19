"""Direction contrasts with a matched appearance term."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import json, os, sys, time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from affectdyn.features.unbc_pretraining_comparison import _auc                                         # noqa: E402
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds                      # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout, N_REP  # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                                 # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS                                     # noqa: E402
from affectdyn.inference.pain_direction_cluster_inference import paired_bootstrap                        # noqa: E402
from affectdyn.provenance import run_provenance, audit_loso                              # noqa: E402

OUT = "runs/matched_appearance_contrasts"
SEED = 20260914
NBOOT = 10_000
TOL_AUC = 0.002


def oof_reps(X, y, sub, mask, folds, n_rep=N_REP):
    """One seeded probe fit per rep, seeded exactly as stage_w2_map.oof_reps seeds them."""
    import torch
    out = []
    for rep in range(n_rep):
        torch.manual_seed(1000 + rep)
        out.append(_oof_scores(X, y, sub, mask, folds))
    return out


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/appearance_term.log", "w")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int)
    mask = lab["direction"] != -1
    folds = _loso_folds(sub)
    audit_loso(sub, folds)
    rng = np.random.default_rng(SEED)

    # Evaluate both appearance terms for each video model.
    cells = {}
    IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    for k in KS:
        cells[("both", f"clip_K{k}")] = image_readout(IMG["clip"], k)
    for name, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        mean_frame = z["K16_up"].mean(1)            # Each K=16 window contains one frame.
        for k in KS:
            key = tmpl.format(k=k)
            if key not in z.files:
                continue
            cells[("full", f"{name}_K{k}")] = video_readout(z["full"], z[key])
            cells[("meanframe", f"{name}_K{k}")] = video_readout(mean_frame, z[key])

    p(f"W21 | {int(mask.sum())} direction-labelled clips, {len(set(sub.tolist()))} subjects, LOSO "
      f"| {len(cells)} cell fits | {N_REP} seeded probe fits each | CPU")

    reps = {}
    for i, (tag, X) in enumerate(sorted(cells.items()), 1):
        reps[tag] = oof_reps(X, y, sub, mask, folds)
        if i % 8 == 0 or i == len(cells):
            p(f"  fitted {i}/{len(cells)} cells")

    keep = mask.copy()
    for rs in reps.values():
        for s in rs:
            keep &= ~np.isnan(s)
    yk = y[keep]
    p(f"clips retained: {int(keep.sum())}")

    pooled = {tag: np.mean(rs, axis=0) for tag, rs in reps.items()}
    auc = {tag: float(_auc(pooled[tag][keep], yk)) for tag in pooled}

    logged = json.load(open("runs/pain_encoder_maps/map.json"))["table"]
    grows = []
    for (arm, name), a in sorted(auc.items()):
        if arm == "meanframe" or name not in logged:
            continue
        grows.append({"cell": name, "logged": float(logged[name]["pooled_oof"]), "recomputed": a,
                      "delta": abs(a - float(logged[name]["pooled_oof"]))})
    max_d = max(g["delta"] for g in grows)

    # Evaluate 24 contrasts per arm.
    def run_arm(arm):
        rows = []
        for k in KS:
            for v in VIDEO_SOURCES:
                key = ("both", f"clip_K{k}"), (arm, f"{v}_K{k}")
                if key[1] not in pooled:
                    continue
                sa, sb = pooled[key[0]][keep], pooled[key[1]][keep]
                b = paired_bootstrap(sa, sb, yk, sub[keep], "subject", rng, NBOOT)
                rows.append({"label": f"K={k:2d}: CLIP vs {v}", "video": v, "K": k,
                             "auc_clip": float(_auc(sa, yk)), "auc_video": float(_auc(sb, yk)),
                             "gap": float(_auc(sa, yk) - _auc(sb, yk)),
                             "ci": b["ci"], "p_boot": b["p_boot"]})
        for r, a in zip(rows, holm([r["p_boot"] for r in rows])):
            r["p_holm"] = float(a)
            r["survives"] = bool(a < 0.05)
        return rows

    arms = {}
    for arm in ("full", "meanframe"):
        arms[arm] = run_arm(arm)
        surv = [r["label"] for r in arms[arm] if r["survives"]]
        p(f"\n=== appearance term: {arm} === {len(surv)} of {len(arms[arm])} survive Holm")
        for r in arms[arm]:
            p(f"  {r['label']:26s} gap {r['gap']:+.4f}  CI [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  "
              f"holm {r['p_holm']:.4f}{'  *' if r['survives'] else ''}")

    w9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))["summary"]
    w9_surv = sorted(x["label"] for x in w9["survivors_subject_boot"])
    full_surv = sorted(r["label"] for r in arms["full"] if r["survives"])
    mf_surv = sorted(r["label"] for r in arms["meanframe"] if r["survives"])
    changed = [{"label": a["label"], "full": a["survives"], "meanframe": b["survives"],
                "holm_full": a["p_holm"], "holm_meanframe": b["p_holm"]}
               for a, b in zip(arms["full"], arms["meanframe"]) if a["survives"] != b["survives"]]
    mae_only = all("videomae" in r["video"] for r in arms["meanframe"] if r["survives"])

    res = {"provenance": run_provenance(seeds=[SEED]), "seed": SEED, "nboot": NBOOT,
           "n_clips": int(keep.sum()), "auc": {f"{a}:{n}": v for (a, n), v in auc.items()},
           "gate": {"tolerance_auc": TOL_AUC, "max_abs_delta_auc": float(max_d),
                    "auc_rows": grows, "w9_survivors": w9_surv, "full_arm_survivors": full_surv,
                    "survivors_match_w9": full_surv == w9_surv,
                    "passes": bool(max_d <= TOL_AUC and full_surv == w9_surv)},
           "arms": arms,
           "summary": {"n_survivors_full": len(full_surv), "n_survivors_meanframe": len(mf_surv),
                       "survivors_meanframe": mf_surv, "decisions_changed": changed,
                       "all_meanframe_survivors_are_masked_autoencoding": bool(mae_only)}}
    res["status"] = "COMPLETE" if res["gate"]["passes"] else "COMPLETE WITH GATE MISMATCH"
    json.dump(res, open(f"{OUT}/appearance_term.json", "w"), indent=2)
    p("")
    p(f"gate: max |delta AUC| {max_d:.5f} (tol {TOL_AUC}), survivors match W9 "
      f"{res['gate']['survivors_match_w9']}, passes {res['gate']['passes']}")
    p(f"survivors, full-clip appearance: {len(full_surv)} -> {full_surv}")
    p(f"survivors, mean-of-frames appearance: {len(mf_surv)} -> {mf_surv}")
    p(f"decisions changed: {len(changed)}")
    for c in changed:
        p(f"   {c['label']}  full {c['holm_full']:.4f} -> meanframe {c['holm_meanframe']:.4f}")
    p(f"every survivor still against masked autoencoding: {mae_only}")
    p(f"wrote {OUT}/appearance_term.json | {res['status']}")
    logf.close()


if __name__ == "__main__":
    main()
