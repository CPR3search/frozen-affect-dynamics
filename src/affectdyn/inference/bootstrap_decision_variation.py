"""Bootstrap draw variation of decisions near 0.05.

Usage:  python -m affectdyn.inference.bootstrap_decision_variation
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                    # Run on the CPU.
import argparse, json, multiprocessing as mp, time, zlib

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.inference.peak_cluster_inference import perm_test, bootstrap as w7_bootstrap, SEED as W7_SEED, NPERM as W7_NPERM
from affectdyn.inference.pain_direction_cluster_inference import paired_bootstrap as w9_paired
from affectdyn.inference.bootstrap_detection_power import ClipSet
from affectdyn.inference.calibrated_clustered_auc import wilson
from affectdyn.provenance import run_provenance

OUT = "runs/bootstrap_decision_variation"
SEED = 20260924
W9_SEED = 20260911
NBOOT = 10_000
NBOOT_BIG = 100_000
N_STREAMS = 200
N_BINOM = 20_000
N_WORKERS = 16
_FAMS = {}
CHUNK = 2500
ALPHA = 0.05
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-18 — W35-W37 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-18 — W35-W37 CRITERIA AMENDMENT"
ADJACENT = {"contrasts24": ["K= 2: CLIP vs videomae", "K= 8: CLIP vs videomae"],
            "aw2_shape18": ["aw2_image:clip_K16"],
            "f5": ["K4:ordered_minus_shuffled"]}
F5 = ("K2:ordered_minus_shuffled", "K4:ordered_minus_shuffled", "K8:ordered_minus_shuffled",
      "K16:ordered_minus_shuffled", "K4:ordered_minus_static")


def mult(draw, G):
    """Multiplicity of each subject in each resample row."""
    b = draw.shape[0]
    return np.bincount((draw + (np.arange(b) * G)[:, None]).ravel(), minlength=b * G).reshape(b, G).astype(float)


def fast_vals(cs, C, draws):
    """Pooled statistic of each resample: C is counts(s) for a single AUC or counts(a) - counts(b) for a gap."""
    W = mult(draws, cs.G)
    den = (W @ cs.P) * (W @ cs.N)
    ok = den > 0
    return ((W @ C) * W).sum(1)[ok] / den[ok], int((~ok).sum())


def p_from(v, ref):
    p = 2.0 * min(float((v <= ref).mean()), float((v >= ref).mean()))
    return float(min(1.0, max(p, 1.0 / len(v))))




def slow_vals(test, members, draws):
    """The original resample-and-recompute statistic, for the same resample indices."""
    out = []
    for row in draws:
        i = np.concatenate([members[g] for g in row])
        y = test["y"][i]
        if y.min() == y.max():
            continue
        out.append(_auc(test["a"][i], y) - (_auc(test["b"][i], y) if test["b"] is not None else 0.0))
    return np.asarray(out)


def _boot_p(t, rng, nboot):
    if t["b"] is not None:
        return w9_paired(t["a"], t["b"], t["y"], t["g"], "subject", rng, nboot)["p_boot"]
    return w7_bootstrap(t["a"], t["y"], t["g"], "subject", rng, nboot)["p_boot"]


def _stream_job(job):
    fam, s = job
    return [_boot_p(t, np.random.default_rng([SEED, zlib.crc32(f"{fam}:{t['key']}".encode()), s]), NBOOT)
            for t in _FAMS[fam]]


def _big_job(job):
    fam, j = job
    t = _FAMS[fam][j]
    return _boot_p(t, np.random.default_rng([SEED, zlib.crc32(f"{fam}:{t['key']}".encode()), 10 ** 6]), NBOOT_BIG)


def family_contrasts():
    Z = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    keep = Z["keep"].astype(bool)
    y, g = Z["y"][keep].astype(np.int64), Z["subject"][keep]
    W9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))["contrasts"]
    tests = [{"key": r["label"], "a": Z[r["a"]][keep], "b": Z[r["b"]][keep], "y": y, "g": g, "ref": 0.0,
              "pub_raw": r["subject_boot"]["p_boot"], "pub_ci": r["subject_boot"]["ci"],
              "pub_holm": r["holm"]["boot_subject"]} for r in W9]
    return tests


def family_f5():
    Z = np.load("runs/pain_order_cluster_bootstrap/pain_controls_cluster_oof.npz")
    keep = Z["keep"].astype(bool)
    y, g = Z["y"][keep].astype(np.int64), Z["subject"][keep]
    W11 = {r["test"]: r for r in json.load(open("runs/pain_order_cluster_bootstrap/pain_controls_cluster.json"))["tests"]}
    W30 = {r["test"]: r for r in json.load(open("runs/pain_order_holm_family/pain_control_family.json"))["rows"]}
    tests = []
    for name in F5:
        k = name.split(":")[0][1:]
        b = Z["static"][keep] if name.endswith("static") else Z[f"shuf_K{k}"][keep]
        tests.append({"key": name, "a": Z[f"ord_K{k}"][keep], "b": b, "y": y, "g": g, "ref": 0.0,
                      "pub_raw": W11[name]["subject_boot"]["p_boot"], "pub_ci": W11[name]["subject_boot"]["ci"],
                      "pub_holm": W30[name]["holm_F5_bootstrap"],
                      "pub_holm_calibrated": W30[name]["holm_F5_calibrated"]})
    return tests


def family_aw2():
    Z = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    W7 = json.load(open("runs/peak_cluster_inference/cluster.json"))
    pub = {(c["corpus"], c["cell"]): c for c in W7["cells"]}
    corpora = []
    for corpus, tag, maskkey in (("unbc", "unbc", "keep"), ("aw2_image", "aw2img", "mask"), ("aw2_video", "aw2vid", "mask")):
        y, g, m = Z[f"{tag}__y"], Z[f"{tag}__subject"], Z[f"{tag}__{maskkey}"].astype(bool)
        names = [f[len(tag) + 2:] for f in Z.files if f.startswith(tag + "__")
                 and f[len(tag) + 2:] not in ("subject", "y", "keep", "mask", "idx")]
        corpora.append((corpus, tag, y, g, m, names))
    tests = []
    for corpus, tag, y, g, m, names in corpora:
        if corpus == "unbc":
            continue
        for n in names:
            s = Z[f"{tag}__{n}"]
            keep = m & ~np.isnan(s)
            c = pub[(corpus, n)]
            tests.append({"key": f"{corpus}:{n}", "a": s[keep], "b": None, "y": y[keep].astype(np.int64),
                          "g": g[keep], "ref": 0.5, "pub_raw": c["subject"]["p_boot"],
                          "pub_ci": c["subject"]["ci"], "pub_holm": c["holm_pooled"]["boot_subject"]})
    return tests, corpora, Z, pub


def replay_w7(corpora, Z, pub, log):
    rng = np.random.default_rng(W7_SEED)
    out = {}
    for corpus, tag, y, g, m, names in corpora:
        for n in names:
            s = Z[f"{tag}__{n}"]
            keep = m & ~np.isnan(s)
            sk, yk, gk = s[keep], y[keep].astype(np.int64), g[keep]
            perm_test(sk, yk, gk, "clip", rng, W7_NPERM)
            perm_test(sk, yk, gk, "subject", rng, W7_NPERM)
            w7_bootstrap(sk, yk, gk, "clip", rng, NBOOT)
            out[(corpus, n)] = w7_bootstrap(sk, yk, gk, "subject", rng, NBOOT)
        log(f"  replayed {corpus}: {len(names)} tests")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/mc_error.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/mc_error.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"W36 | streams {N_STREAMS} x B={NBOOT}, one run at B={NBOOT_BIG} | seed {SEED}")
    fams = {"contrasts24": family_contrasts(), "f5": family_f5()}
    aw2_tests, corpora, Z7, pub7 = family_aw2()
    fams["aw2_shape18"] = aw2_tests
    res = {"provenance": run_provenance(seeds=[SEED, W9_SEED, W7_SEED], device="cpu"),
           "criteria_heading": CRITERIA_HEADING, "amendment_heading": AMENDMENT_HEADING,
           "nboot": NBOOT, "nboot_big": NBOOT_BIG, "n_streams": N_STREAMS, "families": {}}

    gates = {}
    for fam in ("contrasts24", "f5"):
        rows, ok = [], True
        for t in fams[fam]:
            b = w9_paired(t["a"], t["b"], t["y"], t["g"], "subject",
                          np.random.default_rng([W9_SEED, zlib.crc32((t["key"] + ":subject").encode())]), NBOOT)
            same = (b["p_boot"] == t["pub_raw"]) and (b["ci"] == t["pub_ci"])
            ok &= same
            rows.append({"key": t["key"], "p": b["p_boot"], "pub": t["pub_raw"], "exact": same})
        adj = holm([r["p"] for r in rows])
        holm_ok = all(abs(a - t["pub_holm"]) <= 1e-12 for a, t in zip(adj, fams[fam]))
        gates[fam] = {"all_raw_exact": bool(ok), "holm_exact": bool(holm_ok), "passes": bool(ok and holm_ok),
                      "n_tests": len(rows)}
        log(f"gate {fam}: {gates[fam]}")
    log("replaying W7's whole call order (54 tests, 2 permutation tests and 2 bootstraps each) ...")
    rep = replay_w7(corpora, Z7, pub7, log)
    raw_ok = all(rep[k]["p_boot"] == c["subject"]["p_boot"] and rep[k]["ci"] == c["subject"]["ci"]
                 for k, c in pub7.items() if k in rep)
    aw2_keys = [(t["key"].split(":")[0], t["key"].split(":")[1]) for t in aw2_tests]
    adj = holm([rep[k]["p_boot"] for k in aw2_keys])
    holm_ok = all(abs(a - t["pub_holm"]) <= 1e-12 for a, t in zip(adj, aw2_tests))
    gates["aw2_shape18"] = {"all_54_raw_exact": bool(raw_ok), "holm_exact": bool(holm_ok),
                            "passes": bool(raw_ok and holm_ok), "n_tests": len(aw2_tests)}
    log(f"gate aw2_shape18: {gates['aw2_shape18']}")

    vrng = np.random.default_rng([SEED, 7])
    fast_dev = {}
    for fam, tests in fams.items():
        worst = 0.0
        for t in tests[:3] + [x for x in tests if x["key"] in ADJACENT[fam]]:
            cs = ClipSet(t["key"], t["y"], t["g"], len(tests))
            members = [np.flatnonzero(cs.inv == k) for k in range(cs.G)]
            C = cs.counts(t["a"]) - (cs.counts(t["b"]) if t["b"] is not None else 0.0)
            draws = vrng.integers(0, cs.G, size=(400, cs.G))
            fv, _ = fast_vals(cs, C, draws)
            sv = slow_vals(t, members, draws)
            worst = max(worst, float(np.max(np.abs(fv - sv))))
        fast_dev[fam] = worst
    gates["fast_bootstrap"] = {"max_abs_deviation": fast_dev, "tolerance": 1e-9,
                               "passes": bool(all(v <= 1e-9 for v in fast_dev.values()))}
    log(f"gate fast bootstrap: {gates['fast_bootstrap']}")
    res["gates"] = gates
    held = [f for f in fams if not gates[f]["passes"]]
    gates["fast_bootstrap"]["use"] = ("not used: disqualified on tied data, so every stream calls the original "
                                      "function" if not gates["fast_bootstrap"]["passes"] else "not used")
    if held:
        log(f"HOLD families: {held}")

    # Use the reference functions to preserve their tie handling in every stream.
    global _FAMS
    _FAMS = {f: fams[f] for f in fams if f not in held}
    jobs = [(f, s) for f in _FAMS for s in range(N_STREAMS)]
    big_jobs = [(f, j) for f in _FAMS for j in range(len(_FAMS[f]))]
    t1 = time.time()
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        stream_p = pool.map(_stream_job, jobs, chunksize=4)
        big_p = pool.map(_big_job, big_jobs)
    log(f"  {len(jobs)} family-streams and {len(big_jobs)} runs at B={NBOOT_BIG} in {time.time() - t1:.0f}s "
        f"on {N_WORKERS} workers")
    by_fam = {f: np.empty((N_STREAMS, len(_FAMS[f]))) for f in _FAMS}
    for (f, s), ps in zip(jobs, stream_p):
        by_fam[f][s] = ps
    big_by = {f: [None] * len(_FAMS[f]) for f in _FAMS}
    for (f, j), pv in zip(big_jobs, big_p):
        big_by[f][j] = pv
    for fam, tests in fams.items():
        if fam in held:
            res["families"][fam] = {"status": "HOLD: exact replay failed"}
            continue
        adj_streams = np.array([holm(list(r)) for r in by_fam[fam]])
        adj_big = holm(big_by[fam])
        comp = "bootstrap_component_" if fam == "f5" else ""
        logged_dec = [t["pub_holm"] < ALPHA for t in tests]

        brng = np.random.default_rng([SEED, zlib.crc32(("binomial:" + fam).encode())])
        q = np.array([0.0 if t["pub_raw"] <= 1.0 / NBOOT else t["pub_raw"] / 2.0 for t in tests])
        cnt = brng.binomial(NBOOT, q[None, :].repeat(N_BINOM, 0))
        pb = np.clip(np.maximum(2.0 * cnt / NBOOT, 1.0 / NBOOT), None, 1.0)
        adj_binom = np.array([holm(list(r)) for r in pb])

        rows = []
        for j, t in enumerate(tests):
            if t["key"] not in ADJACENT[fam]:
                continue
            a = adj_streams[:, j]
            flips = int(((a < ALPHA) != logged_dec[j]).sum())
            lo, hi = wilson(flips, N_STREAMS)
            cls = "draw-dependent" if lo >= 0.05 else "stable" if hi < 0.05 else "unresolved"
            ab = adj_binom[:, j]
            row = {"test": t["key"], "published_adjusted": t["pub_holm"], "published_raw": t["pub_raw"],
                   comp + "logged_decision_significant": bool(logged_dec[j]),
                   "streams_adjusted_median": float(np.median(a)),
                   "streams_adjusted_2.5_97.5": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                   "streams_adjusted_min_max": [float(a.min()), float(a.max())],
                   comp + "flip_share": flips / N_STREAMS, comp + "flip_share_wilson95": [lo, hi],
                   comp + "classification": cls, "adjusted_at_B100000": float(adj_big[j]),
                   "binomial_supplement_2.5_97.5": [float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))],
                   "binomial_supplement_flip_share": float(((ab < ALPHA) != logged_dec[j]).mean())}
            if fam == "f5":
                cal = t["pub_holm_calibrated"]
                joint_logged = bool(logged_dec[j] and cal < ALPHA)
                joint_flips = int((((a < ALPHA) & (cal < ALPHA)) != joint_logged).sum())
                row.update({"published_adjusted_calibrated": cal, "joint_logged_decision_significant": joint_logged,
                            "joint_flip_share": joint_flips / N_STREAMS,
                            "joint_flip_share_wilson95": list(wilson(joint_flips, N_STREAMS)),
                            "note": ("the published F5 decision needs BOTH adjusted values below 0.05; the calibrated "
                                     f"value is fixed at {cal:.4f}, so the flips above concern the bootstrap "
                                     "component only and the joint decision cannot change")})
            rows.append(row)
            log(f"  {fam} {t['key']}: published {t['pub_holm']:.4f} | streams median {row['streams_adjusted_median']:.4f} "
                f"[{row['streams_adjusted_2.5_97.5'][0]:.4f}, {row['streams_adjusted_2.5_97.5'][1]:.4f}] | "
                f"{comp}flips {flips}/{N_STREAMS} Wilson [{lo:.3f}, {hi:.3f}] -> {cls} | B=100k {adj_big[j]:.4f}")
        res["families"][fam] = {"status": "COMPLETE", "n_tests": len(tests), "threshold_adjacent": rows}

    res["status"] = "COMPLETE" if not held else f"PARTIAL: held {held}"
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | {res['status']}")
    logf.close()


if __name__ == "__main__":
    main()
