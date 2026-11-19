"""Clustered AUC test on the converged scores.

Usage:  python -m affectdyn.inference.converged_clustered_calibration
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, tempfile

import affectdyn.inference.calibrated_clustered_auc as w12   # noqa: E402
import numpy as np                                            # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                # noqa: E402
from affectdyn.inference.calibrated_probability_mc_bounds import clopper_pearson     # noqa: E402
from affectdyn.provenance import run_provenance                         # noqa: E402

OUT = "runs/converged_clustered_calibration"
CONV = "runs/converged_probe_fits/converged_scores.npz"
W9_NPZ = "runs/pain_direction_cluster_inference/direction_cluster_oof.npz"
W7_NPZ = "runs/peak_cluster_inference/cluster_oof_scores.npz"
META7 = {"unbc__subject", "unbc__y", "unbc__keep", "aw2img__subject", "aw2img__y", "aw2img__mask",
         "aw2vid__subject", "aw2vid__y", "aw2vid__mask", "aw2vid__idx"}
META9 = {"subject", "y", "keep"}
N = w12.N_NULL
ALPHA = 0.05
NAMED = ("K= 2: CLIP vs videomae", "unbc:xclip_K8")
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## Uncapped, common-set, and converged-fit criteria"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## Additional evaluation criteria"


class Substituted:
    """An NpzFile look-alike: the original keys in their original order, some arrays replaced."""

    def __init__(self, base, repl, log):
        self.files = list(base.files); self._base = base; self._repl = repl; self._log = log

    def __getitem__(self, k):
        if k in self._repl:
            self._log.add(k)
            return self._repl[k]
        return self._base[k]


def run_pass(work, name, substitute, freeze_nuisance, conv, pub_cal):
    od = os.path.join(work, name)   # Keep temporary outputs outside the repository.
    real_load = np.load
    consumed = set()
    repl9 = {k: conv[f"f1__{k}"] for k in real_load(W9_NPZ).files if k not in META9} if substitute else {}
    repl7 = {k: conv[k] for k in real_load(W7_NPZ).files if k not in META7} if substitute else {}
    if substitute:
        assert len(repl9) == 28 and len(repl7) == 54, (len(repl9), len(repl7))

    def patched(path, *a, **kw):
        z = real_load(path, *a, **kw)
        if substitute and path == W9_NPZ:
            return Substituted(z, repl9, consumed)
        if substitute and path == W7_NPZ:
            return Substituted(z, repl7, consumed)
        return z

    real_sp, real_spb = w12.sim_paired, w12.sim_paired_bootstrap
    th0, rho0 = pub_cal["theta_direction"], pub_cal["rho_direction"]

    def sp(d, theta, rho, rng, n):
        if freeze_nuisance and d.name == "unbc_direction":
            theta, rho = th0, rho0
        return real_sp(d, theta, rho, rng, n)

    def spb(cs, d, theta, rho, rng, n, nboot):
        if freeze_nuisance and d.name == "unbc_direction":
            theta, rho = th0, rho0
        return real_spb(cs, d, theta, rho, rng, n, nboot)

    np.load, w12.sim_paired, w12.sim_paired_bootstrap = patched, sp, spb
    argv = sys.argv
    try:
        sys.argv = ["stage_w12_calibrated_recheck.py", "--out-dir", od, "--force"]
        w12.main()
    finally:
        np.load, w12.sim_paired, w12.sim_paired_bootstrap = real_load, real_sp, real_spb
        sys.argv = argv
    if substitute:
        want = set(repl9) | set(repl7)
        assert want <= consumed, f"not consumed: {sorted(want - consumed)[:5]}"
    return json.load(open(os.path.join(od, "calibrated_recheck.json"))), sorted(consumed)


def compare(a, b, fams):
    """Exact equality of every row's p, adjusted p and decision in the named families."""
    bad = []
    for f in fams:
        ra, rb = a["families"][f]["rows"], b["families"][f]["rows"]
        for x, y in zip(ra, rb):
            for k in ("p_calibrated", "holm_calibrated", "decision_calibrated"):
                if k in x and x.get(k) != y.get(k):
                    bad.append((f, x["test"], k, x.get(k), y.get(k)))
    return bad


def mc(family):
    rows = family["rows"]
    p = np.array([r["p_calibrated"] for r in rows])
    sim = [r["p_method"] == "simulation-calibrated" for r in rows]
    m_sim = sum(sim)
    if m_sim == 0:
        return {"n_simulated": 0, "rows": []}
    level = 1 - ALPHA / m_sim
    lo_v, hi_v = p.copy(), p.copy()
    for i, r in enumerate(rows):
        if sim[i]:
            k = int(round((N + 1) * r["p_calibrated"])) - 1
            lo_v[i], hi_v[i] = clopper_pearson(k, N, level)
    adj, lo, hi = holm(list(p)), holm(list(lo_v)), holm(list(hi_v))
    out = []
    for i, r in enumerate(rows):
        near = 0.02 <= adj[i] <= 0.10 or r["test"] in NAMED
        if near:
            out.append({"test": r["test"], "holm": float(adj[i]), "holm_from_bounds": [float(lo[i]), float(hi[i])],
                        "classification": ("MC-stable significant" if hi[i] < ALPHA else
                                           "MC-stable non-significant" if lo[i] >= ALPHA else "MC-unresolved")})
    return {"n_simulated": m_sim, "rows": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--work-dir", default=None, help="where the W12 passes write; must be outside the repository")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != w12.REPO:
        raise SystemExit("run from the repository root")
    out_json = os.path.join(OUT, "calibrated_converged.json")
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    work = os.path.abspath(args.work_dir or tempfile.mkdtemp(prefix="w49_"))
    assert not work.startswith(w12.REPO + os.sep), "the work directory must lie outside the repository"
    prov = run_provenance(seeds=[w12.SEED], device="cpu")   # Capture provenance before writing outputs.
    pub = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))
    pub_cal = pub["calibration"]["parameters"]
    conv = np.load(CONV)
    sha = hashlib.sha256(open(CONV, "rb").read()).hexdigest()
    fams_all = [f for f in pub["families"]]

    gate_json, _ = run_pass(work, "gate", False, False, conv, pub_cal)
    g1_bad = compare(gate_json, pub, fams_all)
    g1 = {"mismatches": g1_bad[:20], "n_mismatches": len(g1_bad),
          "calibration_equal": gate_json["calibration"] == pub["calibration"], "passes": not g1_bad
          and gate_json["calibration"] == pub["calibration"]}
    print(f"[w49] G1 gate pass: {g1['passes']} ({g1['n_mismatches']} mismatches)", flush=True)
    if not g1["passes"]:
        os.makedirs(OUT, exist_ok=True)
        json.dump({"provenance": prov, "G1": g1, "status": "HOLD: gate failed"}, open(out_json, "w"), indent=2)
        return

    res = {"provenance": prov, "criteria_heading": CRITERIA_HEADING,
           "amendment_heading": AMENDMENT_HEADING, "converged_sha256": sha, "G1": g1, "passes": {}}
    for name, freeze in (("primary", False), ("secondary", True)):
        j, consumed = run_pass(work, name, True, freeze, conv, pub_cal)
        g2_bad = compare(j, pub, ["w8_order_controls_4", "w11_exploratory_5"])
        fam_out = {}
        for f in ("direction_contrasts_24", "unbc_shape_36", "aw2_shape_18", "cross_corpus_18"):
            P = {r["test"]: r for r in pub["families"][f]["rows"]}
            rows = j["families"][f]["rows"]
            changes = [{"test": r["test"], "published_calibrated": P[r["test"]]["decision_calibrated"],
                        "converged_calibrated": r["decision_calibrated"],
                        "holm_published": P[r["test"]]["holm_calibrated"], "holm_converged": r["holm_calibrated"]}
                       for r in rows if r["decision_calibrated"] != P[r["test"]]["decision_calibrated"]]
            named = {r["test"]: {"holm_published": P[r["test"]]["holm_calibrated"], "holm_converged": r["holm_calibrated"],
                                 "p_method": r["p_method"], "significant": r["decision_calibrated"]}
                     for r in rows if r["test"] in NAMED}
            fam_out[f] = {"n_significant_published": pub["families"][f]["n_significant_calibrated"],
                          "n_significant_converged": j["families"][f]["n_significant_calibrated"],
                          "changes_vs_w12": changes, "named": named, "mc": mc(j["families"][f])}
            print(f"[w49] {name} {f}: {fam_out[f]['n_significant_published']} -> "
                  f"{fam_out[f]['n_significant_converged']} changes {[c['test'] for c in changes]} named {named}",
                  flush=True)
        res["passes"][name] = {"G2_mismatches": g2_bad, "G2_passes": not g2_bad, "n_arrays_substituted": len(consumed),
                               "nuisance": j["calibration"]["parameters"], "families": fam_out}
    ok = g1["passes"] and all(v["G2_passes"] for v in res["passes"].values())
    res["status"] = "COMPLETE" if ok else "HOLD: G2 failed"
    os.makedirs(OUT, exist_ok=True)
    for name in ("gate", "primary", "secondary"):
        shutil.copy(os.path.join(work, name, "calibrated_recheck.json"), os.path.join(OUT, f"{name}_calibrated_recheck.json"))
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"[w49] wrote {out_json} | {res['status']}", flush=True)


if __name__ == "__main__":
    main()
