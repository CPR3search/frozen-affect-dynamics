"""Within-cluster estimand on the extended nested scores."""
from __future__ import annotations
import json, os, sys
import numpy as np

import affectdyn.inference.headline_refit_contrasts as w                                              # noqa: E402
from affectdyn.provenance import run_provenance                                             # noqa: E402


def main():
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz"); sub = lab["subject"]; y = (lab["direction"] == 1).astype(int)
    keep = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz")["keep"].astype(bool)
    M = json.load(open("runs/pain_encoder_maps/map.json")); Z = np.load("runs/extended_nested_probe_fits/nested_scores.npz")
    rows = [r for r in M["primary"] if f"f1__{r['a']}" in Z.files and f"f1__{r['b']}" in Z.files]
    assert len(rows) == 24
    W = {n: w.within_aucs(Z[f"f1__{n}"], y, sub, keep) for n in {r["a"] for r in rows} | {r["b"] for r in rows}}
    subs = sorted(set.intersection(*[set(W[r["a"]]) & set(W[r["b"]]) for r in rows])); G = len(subs)
    rng = np.random.default_rng(w.SEED); draws = rng.integers(0, G, (w.B_WITHIN, G)); res = []
    for r in rows:
        d = np.array([W[r["a"]][g] - W[r["b"]][g] for g in subs]); b = d[draws].mean(1)
        p = min(1.0, max(1.0 / w.B_WITHIN, 2 * min((b <= 0).mean(), (b >= 0).mean())))
        res.append({"label": r["label"], "delta_subject": float(d.mean()),
                    "ci": [float(x) for x in np.percentile(b, [2.5, 97.5])], "p_boot": float(p)})
    for x, a in zip(res, w.holm([x["p_boot"] for x in res])):
        x["holm"] = float(a)
    out = {"provenance": run_provenance(seeds=[w.SEED], device="cpu"), "fit": "nested_w56", "n_mixed": G,
           "contrasts": res, "significant": [x["label"] for x in res if x["holm"] < w.ALPHA]}
    json.dump(out, open("runs/headline_refit_contrasts/within_subject_w56.json", "w"), indent=1)
    print(out["significant"])


if __name__ == "__main__":
    main()
