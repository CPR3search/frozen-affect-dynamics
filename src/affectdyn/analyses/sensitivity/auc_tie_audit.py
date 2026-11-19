"""Effect of the AUC tie convention on reported values."""
from __future__ import annotations
import json, os, sys, time
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.provenance import run_provenance

OUT = "runs/auc_tie_audit"
W12_MAX = 0.00011646866992798266


def midrank(s, y):
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); ss = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tied_pairs(s, y):
    vals, inv = np.unique(s, return_inverse=True)
    n1 = np.bincount(inv, weights=(y == 1).astype(float), minlength=len(vals))
    n0 = np.bincount(inv, weights=(y == 0).astype(float), minlength=len(vals))
    return int((n1 * n0).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/tie_audit.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    rows = []
    z = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    pub7 = {(c["corpus"], c["cell"]): c["auc"] for c in json.load(open("runs/peak_cluster_inference/cluster.json"))["cells"]}
    for pre, corpus, mkey in (("unbc", "unbc", "keep"), ("aw2img", "aw2_image", "mask"), ("aw2vid", "aw2_video", "mask")):
        y, m = z[f"{pre}__y"].astype(int), z[f"{pre}__{mkey}"].astype(bool)
        for k in z.files:
            if not k.startswith(pre + "__") or k.split("__")[1] in ("subject", "y", "keep", "mask", "idx"):
                continue
            cell = k.split("__")[1]; s = z[k]; mm = m & np.isfinite(s)
            rows.append(("w7", corpus, cell, s[mm], y[mm], pub7.get((corpus, cell))))
    z9 = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    mp = json.load(open("runs/pain_encoder_maps/map.json"))["table"]
    y9, k9 = z9["y"].astype(int), z9["keep"].astype(bool)
    for cell in z9.files:
        if cell in ("subject", "y", "keep"):
            continue
        s = z9[cell]; mm = k9 & np.isfinite(s)
        rows.append(("w9", "unbc_direction", cell, s[mm], y9[mm], mp.get(cell, {}).get("pooled_oof")))

    out, gate_ok = [], True
    for src, corpus, cell, s, y, pub in rows:
        leg = float(_auc(s, y)); mid = midrank(s, y); tp = tied_pairs(s, y)
        exact = pub is not None and abs(leg - pub) <= 1e-12
        gate_ok &= exact
        out.append({"source": src, "corpus": corpus, "cell": cell, "n": int(len(y)), "n_pos": int(y.sum()),
                    "published": pub, "legacy": leg, "legacy_reproduces_published": bool(exact),
                    "midrank": mid, "abs_diff": abs(leg - mid), "cross_class_tied_pairs": tp,
                    "printed3_legacy": f"{leg:.3f}", "printed3_midrank": f"{mid:.3f}",
                    "printed3_changes": f"{leg:.3f}" != f"{mid:.3f}",
                    "printed4_changes": f"{leg:.4f}" != f"{mid:.4f}"})
    worst = max(out, key=lambda r: r["abs_diff"])
    w12_match = abs(worst["abs_diff"] - W12_MAX) <= 1e-12 and worst["cell"] == "xclip_K8" and worst["corpus"] == "unbc"
    gate = {"n_vectors": len(out), "all_legacy_reproduce_published": bool(gate_ok),
            "reproduces_w12_largest_difference": bool(w12_match), "passes": bool(gate_ok and w12_match)}
    log(f"gate: {gate}")
    changed3 = [r for r in out if r["printed3_changes"]]
    changed4 = [r for r in out if r["printed4_changes"]]
    summary = {"max_abs_diff": worst["abs_diff"], "where": f"{worst['corpus']}:{worst['cell']}",
               "vectors_with_ties": sum(r["cross_class_tied_pairs"] > 0 for r in out),
               "printed_3_decimal_changes": [(r["corpus"], r["cell"], r["printed3_legacy"], r["printed3_midrank"]) for r in changed3],
               "printed_4_decimal_changes": [(r["corpus"], r["cell"], f"{r['legacy']:.4f}", f"{r['midrank']:.4f}") for r in changed4]}
    log(f"summary: {summary}")
    json.dump({"provenance": run_provenance(device="cpu"),
               "criteria_heading": "## 2026-09-18 — W39-W44 PRE-DECLARED CRITERIA",
               "amendment_heading": "## 2026-09-18 — W39-W44 CRITERIA AMENDMENT",
               "gate": gate, "summary": summary, "rows": out,
               "scope": "pooled OOF point estimates only; not per-fold entries, intervals, p values or decisions; W34 excluded",
               "status": "COMPLETE" if gate["passes"] else "HOLD: gate failed"},
              open(f"{OUT}/tie_audit.json", "w"), indent=2)
    log(f"wrote {OUT}/tie_audit.json | {'COMPLETE' if gate['passes'] else 'HOLD'}")


if __name__ == "__main__":
    main()
