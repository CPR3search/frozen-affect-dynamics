"""Monte Carlo resolution of the family-size example."""
from __future__ import annotations
import json, os, sys

import affectdyn.reporting.make_tables as mt                                             # noqa: E402  (stdlib-only)
from affectdyn.inference.calibrated_probability_mc_bounds import clopper_pearson           # noqa: E402
from affectdyn.provenance import run_provenance                                # noqa: E402

OUT = "runs/family_size_mc_bounds"
N_PERM = 20_000
ALPHA = 0.05
M_UNION = 36
LEVEL = 1 - ALPHA / M_UNION


def classify(lo, hi):
    return "MC-stable significant" if hi < ALPHA else "MC-stable non-significant" if lo >= ALPHA else "MC-unresolved"


def main():
    shape = json.load(open("runs/pain_encoder_maps/shape.json"))
    cluster = json.load(open("runs/peak_cluster_inference/cluster.json"))
    rows = {r["cell"]: r for r in shape["rows"]}
    cb = {(x["corpus"], x["cell"]): x for x in cluster["cells"]}
    # Convert permutation p-values to integer exceedance counts.
    kperm = {}
    for c, r in rows.items():
        k = r["p"] * (N_PERM + 1) - 1
        assert abs(k - round(k)) < 1e-6, (c, k)
        kperm[c] = int(round(k))
    bperm = {c: clopper_pearson(kperm[c], N_PERM, LEVEL) for c in rows}
    # Convert bootstrap p-values to smaller-tail counts.
    tb, bboot = {}, {}
    for c in rows:
        x = cb[("unbc", c)]["subject"]
        n = 10_000 - int(x.get("resamples_skipped", 0))
        p = x["p_boot"]
        if abs(p - 1.0 / n) < 1e-12:
            t = 0
        elif p >= 1.0:
            t = n // 2
        else:
            t = p * n / 2
            assert abs(t - round(t)) < 1e-6, (c, t)
            t = int(round(t))
        tb[c] = (t, n)
        lo_s, hi_s = clopper_pearson(t, n, LEVEL)
        lo_l, hi_l = clopper_pearson(n - t, n, LEVEL)
        lo = max(min(2 * lo_s, 2 * lo_l, 1.0), 1.0 / n)
        hi = min(max(2 * min(hi_s, hi_l), 1.0 / n), 1.0)
        bboot[c] = (lo, hi)
    fam, pts, gate_ok = list(mt.IMAGE), [], True
    fig = open("out/tables/family_fig.tex").read()
    for extra in [None] + list(mt.ENTRY_ORDER):
        if extra:
            fam.append(extra)
        cells = [f"{e}_K{k}" for e in fam for k in mt.KS if f"{e}_K{k}" in rows]
        m = len(cells)
        i = cells.index("clip_K4")
        ap = mt._holm_adj([rows[c]["p"] for c in cells])[i]
        ab = mt._holm_adj([cb[("unbc", c)]["subject"]["p_boot"] for c in cells])[i]
        ap_lo = mt._holm_adj([bperm[c][0] for c in cells])[i]
        ap_hi = mt._holm_adj([bperm[c][1] for c in cells])[i]
        ab_lo = mt._holm_adj([bboot[c][0] for c in cells])[i]
        ab_hi = mt._holm_adj([bboot[c][1] for c in cells])[i]
        in_fig = f"({m},{ap:.4f})" in fig and f"({m},{ab:.4f})" in fig
        gate_ok &= in_fig
        pts.append({"family_size": m, "encoders": list(fam),
                    "permutation": {"holm": ap, "bounds": [ap_lo, ap_hi], "class": classify(ap_lo, ap_hi)},
                    "subject_bootstrap": {"holm": ab, "bounds": [ab_lo, ab_hi], "class": classify(ab_lo, ab_hi)},
                    "matches_figure": in_fig})
        print(f"m={m:2d}  perm {ap:.4f} [{ap_lo:.4f}, {ap_hi:.4f}] {classify(ap_lo, ap_hi):26s}  "
              f"boot {ab:.4f} [{ab_lo:.4f}, {ab_hi:.4f}] {classify(ab_lo, ab_hi)}  fig {in_fig}")
    os.makedirs(OUT, exist_ok=True)
    res = {"provenance": run_provenance(device="cpu"),
           "criteria_heading": "## 2026-09-18 — W51-W53 PRE-DECLARED CRITERIA",
           "amendment_heading": "## 2026-09-18 — W51-W53 CRITERIA AMENDMENT",
           "n_perm": N_PERM, "simultaneous_level": LEVEL, "clip_K4_k": kperm["clip_K4"],
           "clip_K4_bootstrap_tail": tb["clip_K4"], "gate_matches_figure": bool(gate_ok), "points": pts,
           "status": "COMPLETE" if gate_ok else "HOLD: gate failed"}
    json.dump(res, open(f"{OUT}/family_mc.json", "w"), indent=2)
    print(f"gate {gate_ok} | wrote {OUT}/family_mc.json")


if __name__ == "__main__":
    main()
