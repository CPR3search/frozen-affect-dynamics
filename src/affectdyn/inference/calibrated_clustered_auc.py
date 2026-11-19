"""Clustered AUC test with simulated null p values for the primary families.

Usage:  python -m affectdyn.inference.calibrated_clustered_auc

Article Eqs. (2)-(4): placements computes half-tie placements; Clustered forms cluster residual
sums; cov computes their covariance; z_single, z_paired and z_between form chance, paired and
independent-corpus statistics. Design.components and Design.var implement batched counterparts for
continuous simulated scores. main.p_used selects normal-tail or empirical-null p values using the
calibration checks; z statistics alone are not the final reported tests.
"""
from __future__ import annotations
import os

N_THREADS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import argparse, datetime, json, math, subprocess, sys, time

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.inference.permutation_detection_limits import _probit
from affectdyn.inference.bootstrap_detection_power import ClipSet

from affectdyn.paths import REPO_STR as REPO
ALPHA = 0.05
SEED = 20260912
N_NULL = 100000
N_NULL_BOOT = 20000
B = 10000
BATCH = 256
ICC = 0.3
THETA_SHAPE_PAIRED = 0.6
TOLERANCE = 1.5
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-12 — W12 PRE-DECLARED CRITERIA"


# Compute clustered AUC covariance.

def placements(s, y):
    """For each positive the share of negatives below it, for each negative the share of positives above it, ties counting one half."""
    pos, neg = s[y == 1], s[y == 0]
    M, N = len(pos), len(neg)
    ns, ps = np.sort(neg), np.sort(pos)
    lo, hi = np.searchsorted(ns, pos, "left"), np.searchsorted(ns, pos, "right")
    v10 = (lo + 0.5 * (hi - lo)) / N
    lo, hi = np.searchsorted(ps, neg, "left"), np.searchsorted(ps, neg, "right")
    v01 = (M - hi + 0.5 * (hi - lo)) / M
    return v10, v01


class Clustered:
    """Pooled AUC and per-cluster placement residual sums for the article's clustered covariance.
    Requires both classes and at least two clusters.
    """

    def __init__(self, s, y, groups):
        y = np.asarray(y).astype(np.int64)
        codes = np.unique(groups, return_inverse=True)[1]
        K = int(codes.max()) + 1
        v10, v01 = placements(np.asarray(s, dtype=np.float64), y)
        self.M, self.N, self.K = len(v10), len(v01), K
        self.auc = float(v10.mean())
        cp, cn = codes[y == 1], codes[y == 0]
        self.d10 = np.bincount(cp, weights=v10, minlength=K) - np.bincount(cp, minlength=K) * self.auc
        self.d01 = np.bincount(cn, weights=v01, minlength=K) - np.bincount(cn, minlength=K) * self.auc


def cov(a, b):
    """Covariance of two AUCs computed on the same clips and clusters (variance when a is b)."""
    c = a.K / (a.K - 1)
    s10 = c * float(a.d10 @ b.d10) / a.M
    s01 = c * float(a.d01 @ b.d01) / a.N
    s11 = c * 0.5 * (float(a.d10 @ b.d01) + float(b.d10 @ a.d01))
    return s10 / a.M + s01 / a.N + 2.0 * s11 / (a.M * a.N)


def p_from_z(z):
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def z_single(s, y, g):
    a = Clustered(s, y, g)
    v = cov(a, a)
    z = (a.auc - 0.5) / math.sqrt(v) if v > 0 else 0.0
    return a.auc, z


def z_paired(sa, sb, y, g):
    a, b = Clustered(sa, y, g), Clustered(sb, y, g)
    v = cov(a, a) + cov(b, b) - 2.0 * cov(a, b)
    d = a.auc - b.auc
    return d, (d / math.sqrt(v) if v > 0 else 0.0)


def z_between(su, yu, gu, sw, yw, gw):
    """Return AUC(w) - AUC(u) and its z statistic for independent corpora."""
    u, w = Clustered(su, yu, gu), Clustered(sw, yw, gw)
    d = w.auc - u.auc
    return d, d / math.sqrt(cov(u, u) + cov(w, w))


def naive_cov(sa, sb, y, g):
    y = np.asarray(y).astype(np.int64)
    uniq = np.unique(g)
    K = len(uniq)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    M, N = len(pos), len(neg)

    def comps(s):
        psi = (s[pos][:, None] > s[neg][None, :]).astype(float) + 0.5 * (s[pos][:, None] == s[neg][None, :])
        auc = psi.mean()
        v10, v01 = psi.mean(1), psi.mean(0)
        d10 = np.array([v10[g[pos] == k].sum() - (g[pos] == k).sum() * auc for k in uniq])
        d01 = np.array([v01[g[neg] == k].sum() - (g[neg] == k).sum() * auc for k in uniq])
        return d10, d01

    (a10, a01), (b10, b01) = comps(sa), comps(sb)
    S10 = K / ((K - 1) * M) * sum(a10[k] * b10[k] for k in range(K))
    S01 = K / ((K - 1) * N) * sum(a01[k] * b01[k] for k in range(K))
    S11 = K / (K - 1) * 0.5 * (sum(a10[k] * b01[k] for k in range(K)) + sum(b10[k] * a01[k] for k in range(K)))
    return S10 / M + S01 / N + 2.0 * S11 / (M * N)


class Design:
    """Label and cluster layout of one clip set, prepared for batched z-statistics."""

    def __init__(self, name, y, groups, m):
        self.name, self.m = name, m
        self.y = np.asarray(y).astype(np.int64)
        self.codes = np.unique(groups, return_inverse=True)[1]
        self.K = int(self.codes.max()) + 1
        self.pos, self.neg = np.flatnonzero(self.y == 1), np.flatnonzero(self.y == 0)
        self.M, self.N = len(self.pos), len(self.neg)
        self.Op = np.zeros((self.M, self.K)); self.Op[np.arange(self.M), self.codes[self.pos]] = 1
        self.On = np.zeros((self.N, self.K)); self.On[np.arange(self.N), self.codes[self.neg]] = 1
        self.mk, self.nk = self.Op.sum(0), self.On.sum(0)
        self.c = self.K / (self.K - 1)

    def draw(self, rng, b, theta, e_noise=None, b_noise=None):
        n = len(self.y)
        mu = math.sqrt(2.0) * _probit(theta)
        e = rng.standard_normal((b, n)) if e_noise is None else e_noise
        u = rng.standard_normal((b, self.K)) if b_noise is None else b_noise
        return mu * self.y[None, :] + math.sqrt(1 - ICC) * e + math.sqrt(ICC) * u[:, self.codes]

    def components(self, s):
        """Batched AUC and centred cluster sums; s is [batch, n] with continuous scores.

        Ties are not supported here; observed scores use placements instead.
        """
        r_all = np.argsort(np.argsort(s, axis=1), axis=1) + 1
        r_pos = np.argsort(np.argsort(s[:, self.pos], axis=1), axis=1) + 1
        r_neg = np.argsort(np.argsort(s[:, self.neg], axis=1), axis=1) + 1
        v10 = (r_all[:, self.pos] - r_pos) / self.N
        v01 = (self.M - (r_all[:, self.neg] - r_neg)) / self.M
        auc = v10.mean(1)
        d10 = v10 @ self.Op - auc[:, None] * self.mk
        d01 = v01 @ self.On - auc[:, None] * self.nk
        return auc, d10, d01

    def var(self, A, Bc=None):
        _, a10, a01 = A
        _, b10, b01 = A if Bc is None else Bc
        return (self.c * (a10 * b10).sum(1) / self.M ** 2 + self.c * (a01 * b01).sum(1) / self.N ** 2
                + self.c * ((a10 * b01).sum(1) + (b10 * a01).sum(1)) / (self.M * self.N))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rates(zs, m, label):
    p = np.array([p_from_z(z) for z in zs]) if len(zs) < 2000 else np.vectorize(p_from_z)(zs)
    n = len(zs)
    k05, km = int((p <= ALPHA).sum()), int((p <= ALPHA / m).sum())
    upper = wilson(km, n)[1]
    return {"test": label, "n_studies": n, "rate_at_0.05": k05 / n, "wilson_0.05": wilson(k05, n),
            "rate_at_alpha_over_m": km / n, "wilson_alpha_over_m": wilson(km, n), "alpha_over_m": ALPHA / m,
            "calibrated": bool(upper <= TOLERANCE * ALPHA / m)}


def sim_single(d, theta, rng, n):
    zs, aucs, vs = np.empty(n), np.empty(n), np.empty(n)
    for i in range(0, n, BATCH):
        b = min(BATCH, n - i)
        A = d.components(d.draw(rng, b, theta))
        v = d.var(A)
        zs[i:i + b], aucs[i:i + b], vs[i:i + b] = (A[0] - 0.5) / np.sqrt(v), A[0], v
    return zs, aucs, vs


def sim_paired(d, theta, rho, rng, n):
    zs = np.empty(n)
    for i in range(0, n, BATCH):
        b = min(BATCH, len(zs) - i)
        e1, e2 = rng.standard_normal((b, len(d.y))), rng.standard_normal((b, len(d.y)))
        u1, u2 = rng.standard_normal((b, d.K)), rng.standard_normal((b, d.K))
        sa = d.draw(rng, b, theta, e1, u1)
        sb = d.draw(rng, b, theta, rho * e1 + math.sqrt(1 - rho ** 2) * e2, rho * u1 + math.sqrt(1 - rho ** 2) * u2)
        A, Bc = d.components(sa), d.components(sb)
        v = d.var(A) + d.var(Bc) - 2 * d.var(A, Bc)
        zs[i:i + b] = (A[0] - Bc[0]) / np.sqrt(v)
    return zs


def sim_paired_bootstrap(cs, d, theta, rho, rng, n, nboot):
    ps = np.empty(n)
    G = cs.G
    for i in range(n):
        e1, e2 = rng.standard_normal((1, len(d.y))), rng.standard_normal((1, len(d.y)))
        u1, u2 = rng.standard_normal((1, d.K)), rng.standard_normal((1, d.K))
        sa = d.draw(rng, 1, theta, e1, u1)[0]
        sb = d.draw(rng, 1, theta, rho * e1 + math.sqrt(1 - rho ** 2) * e2, rho * u1 + math.sqrt(1 - rho ** 2) * u2)[0]
        D = cs.counts(sa) - cs.counts(sb)
        vals = []
        for j in range(0, nboot, 2500):
            bb = min(2500, nboot - j)
            draw = rng.integers(0, G, size=(bb, G))
            W = np.bincount((draw + (np.arange(bb) * G)[:, None]).ravel(), minlength=bb * G).reshape(bb, G).astype(float)
            den = (W @ cs.P) * (W @ cs.N)
            ok = den > 0
            vals.append(((W @ D) * W).sum(1)[ok] / den[ok])
        v = np.concatenate(vals)
        ps[i] = min(1.0, max(2.0 * min(float((v <= 0).mean()), float((v >= 0).mean())), 1.0 / len(v)))
    return ps


def within_class_corr(a, b, y):
    ra = np.concatenate([a[y == c] - a[y == c].mean() for c in (0, 1)])
    rb = np.concatenate([b[y == c] - b[y == c].mean() for c in (0, 1)])
    return float(np.corrcoef(ra, rb)[0, 1])


def criteria_verbatim(heading):
    log = os.path.join(REPO, "RESEARCH_LOG-D3-W1.md")
    if not os.path.exists(log):   # Allow the optional criteria file to be absent.
        return None
    lines = open(log, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(heading))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    return "\n".join(lines[start:end]).strip()


def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip()
    except Exception:                                                          # noqa: BLE001
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/calibrated_clustered_auc")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit(f"run from the repository root {REPO}")
    od = os.path.abspath(args.out_dir)
    for forbidden in ("runs/pain_encoder_maps", "runs/affwild_clips_features"):
        f = os.path.join(REPO, forbidden)
        if od == f or od.startswith(f + os.sep):
            raise SystemExit(f"refusing to write under {forbidden}")
    os.makedirs(od, exist_ok=True)
    out_json, out_log = os.path.join(od, "calibrated_recheck.json"), os.path.join(od, "calibrated_recheck.log")
    if not args.force and os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    n_null, n_boot_studies, nboot = (N_NULL, N_NULL_BOOT, B)
    rng = np.random.default_rng(SEED)
    criteria = criteria_verbatim(CRITERIA_HEADING)
    logf = open(out_log, "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"W12 calibration recheck | numpy {np.__version__} threads {N_THREADS} niceness {os.nice(0)} "
        f"| null {n_null} bootstrap studies {n_boot_studies} B {nboot} smoke=False")

    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    C7 = json.load(open("runs/peak_cluster_inference/cluster.json"))
    W9 = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    C9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))
    W8 = np.load("runs/affwild_peak_order_controls/order_controls_oof.npz")
    C8 = json.load(open("runs/affwild_peak_order_controls/order_controls.json"))
    W11 = np.load("runs/pain_order_cluster_bootstrap/pain_controls_cluster_oof.npz")
    C11 = json.load(open("runs/pain_order_cluster_bootstrap/pain_controls_cluster.json"))
    lab = np.load("runs/affwild_clips_features/labels.npz")
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")

    def corpus_arrays(prefix, mask_key):
        return W7[f"{prefix}__y"].astype(np.int64), W7[f"{prefix}__subject"], W7[f"{prefix}__{mask_key}"].astype(bool)

    uy, ug, ukeep = corpus_arrays("unbc", "keep")
    iy, ig, imask = corpus_arrays("aw2img", "mask")
    vy, vg, vmask = corpus_arrays("aw2vid", "mask")

    g_rng = np.random.default_rng([SEED, 1])
    worst_cov = 0.0
    for _ in range(20):
        n, K = 60, 7
        y = (g_rng.random(n) < 0.3).astype(np.int64); y[:2] = [0, 1]
        g = g_rng.integers(0, K, n)
        sa = np.round(g_rng.standard_normal(n), 1); sb = np.round(sa * 0.6 + g_rng.standard_normal(n) * 0.8, 1)
        worst_cov = max(worst_cov, abs(cov(Clustered(sa, y, g), Clustered(sb, y, g)) - naive_cov(sa, sb, y, g)))
    def midrank_auc(s, y):
        order = np.argsort(s, kind="stable")
        uniq, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
        r = np.empty(len(s)); r[order] = (np.cumsum(cnt) - cnt + (cnt + 1) / 2.0)[inv]
        n1 = int(y.sum()); n0 = len(y) - n1
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    # Validate against midrank AUC because tied pairs receive half credit.
    worst_mid, worst_stage, worst_where = 0.0, 0.0, None
    for c in C7["cells"]:
        key = {"unbc": "unbc", "aw2_image": "aw2img", "aw2_video": "aw2vid"}[c["corpus"]]
        y, g, keep = {"unbc": (uy, ug, ukeep), "aw2img": (iy, ig, imask), "aw2vid": (vy, vg, vmask)}[key]
        s = W7[f"{key}__{c['cell']}"]
        k = keep & ~np.isnan(s)
        s, yk = s[k].astype(np.float64), y[k]
        a = Clustered(s, yk, np.zeros(len(yk))).auc
        worst_mid = max(worst_mid, abs(a - midrank_auc(s, yk)))
        if abs(a - _auc(s, yk)) > worst_stage:
            worst_stage, worst_where = abs(a - _auc(s, yk)), f"{c['corpus']}:{c['cell']}"
    gate = {"i_covariance_vs_pairwise": float(worst_cov), "ii_auc_vs_midrank_auc": float(worst_mid),
            "largest_difference_from_stage_auc": float(worst_stage), "where": worst_where,
            "note": "Gate (ii) uses midrank AUC because _auc breaks ties by sort order.",
            "passes": bool(worst_cov < 1e-9 and worst_mid < 1e-9)}
    log(f"pipeline gate: {gate}")
    out = {"provenance": {"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                          "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                          "python": sys.version.split()[0], "numpy": np.__version__, "seed": SEED,
                          "n_null": n_null, "n_null_bootstrap": n_boot_studies, "B": nboot, "icc": ICC,
                          "tolerance": TOLERANCE, "smoke": False,
                          "method": "Obuchowski (1997) clustered AUC variance, subjects as clusters"},
           "criteria_verbatim": criteria, "pipeline_gate": gate, "status": None}
    if not gate["passes"]:
        out["status"] = "HOLD: pipeline gate failed"
        json.dump(out, open(out_json, "w"), indent=2)
        log("pipeline gate FAILED"); return

    d_unbc = Design("unbc_shape", uy[ukeep], ug[ukeep], 36)
    d_full = Design("aw2_full", iy[imask], ig[imask], 18)
    d_sub = Design("aw2_subset", vy[vmask], vg[vmask], 18)
    dkeep = W9["keep"].astype(bool)
    d_dir = Design("unbc_direction", W9["y"][dkeep], W9["subject"][dkeep], 24)
    cells9 = [n for n in W9.files if n not in ("subject", "y", "keep")]
    theta_dir = float(np.median([_auc(W9[n][dkeep], W9["y"][dkeep].astype(int)) for n in cells9]))
    rho_dir = float(np.median([within_class_corr(W9[c["a"]][dkeep], W9[c["b"]][dkeep], W9["y"][dkeep])
                               for c in C9["contrasts"]]))
    wkeep = W11["keep"].astype(bool)
    rho_unbc = within_class_corr(W11["ord_K4"][wkeep], W11["shuf_K4"][wkeep], W11["y"][wkeep])
    ay = (lab["traj"] == 2).astype(np.int64)
    shuf_d = np.mean([W8[f"dinov2_K2__shuffled_r{r}"] for r in range(3)], 0)
    rho_full = within_class_corr(W8["dinov2_K2__ordered"], shuf_d, ay)
    shuf_v = np.mean([W8[f"vjepa_K16__shuffled_r{r}"] for r in range(3)], 0)
    rho_sub = within_class_corr(W8["vjepa_K16__ordered"], shuf_v, ay[idx])
    log(f"theta_dir {theta_dir:.4f} rho_dir {rho_dir:.4f} | rho ordered-shuffled: unbc {rho_unbc:.4f} "
        f"full {rho_full:.4f} subset {rho_sub:.4f}")

    cal = {"parameters": {"theta_direction": theta_dir, "rho_direction": rho_dir, "theta_shape_paired": THETA_SHAPE_PAIRED,
                          "rho_unbc_shape": rho_unbc, "rho_aw2_full": rho_full, "rho_aw2_subset": rho_sub}}
    nulls = {}
    single06 = {}
    for d in (d_unbc, d_full, d_sub):
        z0, _, _ = sim_single(d, 0.5, rng, n_null)
        nulls[("single", d.name)] = z0
        cal[f"single_{d.name}"] = rates(z0, d.m, "AUC against 0.5")
        log(f"  single {d.name}: {cal[f'single_{d.name}']}")
        _, a6, v6 = sim_single(d, THETA_SHAPE_PAIRED, rng, n_null)
        single06[d.name] = (a6, v6)
    for d, rho in ((d_unbc, rho_unbc), (d_full, rho_full), (d_sub, rho_sub)):
        z0 = sim_paired(d, THETA_SHAPE_PAIRED, rho, rng, n_null)
        nulls[("paired", d.name)] = z0
        cal[f"paired_{d.name}"] = rates(z0, {"unbc_shape": 5, "aw2_full": 4, "aw2_subset": 4}[d.name], "paired difference")
        log(f"  paired {d.name}: {cal[f'paired_{d.name}']}")
    z0 = sim_paired(d_dir, theta_dir, rho_dir, rng, n_null)
    nulls[("paired", "unbc_direction")] = z0
    cal["paired_unbc_direction"] = rates(z0, 24, "paired difference")
    log(f"  paired unbc_direction: {cal['paired_unbc_direction']}")
    cs_dir = ClipSet("unbc_direction", W9["y"][dkeep], W9["subject"][dkeep], 24)
    pb = sim_paired_bootstrap(cs_dir, d_dir, theta_dir, rho_dir, rng, n_boot_studies, nboot)
    k05, km = int((pb <= ALPHA).sum()), int((pb <= ALPHA / 24).sum())
    cal["paired_bootstrap_unbc_direction"] = {"n_studies": n_boot_studies, "B": nboot, "rate_at_0.05": k05 / n_boot_studies,
                                              "wilson_0.05": wilson(k05, n_boot_studies),
                                              "rate_at_alpha_over_m": km / n_boot_studies,
                                              "wilson_alpha_over_m": wilson(km, n_boot_studies), "alpha_over_m": ALPHA / 24}
    log(f"  paired bootstrap unbc_direction: {cal['paired_bootstrap_unbc_direction']}")
    for dw_name in ("aw2_full", "aw2_subset"):
        au, vu = single06["unbc_shape"]; aw, vw = single06[dw_name]
        zb = (aw - au) / np.sqrt(vu + vw)
        zb = zb - 0.0                                             # Both arms use theta = 0.6, giving zero null difference.
        nulls[("between", dw_name)] = zb
        cal[f"between_unbc_{dw_name}"] = rates(zb, 18, "difference between corpora")
        log(f"  between unbc/{dw_name}: {cal[f'between_unbc_{dw_name}']}")
    w10 = json.load(open("runs/bootstrap_detection_power/bootstrap_power.json"))["clip_sets"]
    cal["w10_bootstrap_rates_with_intervals"] = {
        name: {"rate_at_0.05": e["calibration"]["rate_at_0.05"],
               "wilson_0.05": wilson(round(e["calibration"]["rate_at_0.05"] * e["calibration"]["n_sim"]), e["calibration"]["n_sim"]),
               "rate_at_alpha_over_m": e["calibration"]["rate_at_alpha_over_m"],
               "wilson_alpha_over_m": wilson(round(e["calibration"]["rate_at_alpha_over_m"] * e["calibration"]["n_sim"]), e["calibration"]["n_sim"])}
        for name, e in w10.items()}
    out["calibration"] = cal

    def p_used(z, test, dname):
        key = f"{test}_{dname}" if test != "between" else f"between_unbc_{dname}"
        if cal[key]["calibrated"]:
            return p_from_z(z), "clustered z"
        null = nulls[(test, dname)]
        return float((1 + (np.abs(null) >= abs(z)).sum()) / (1 + len(null))), "simulation-calibrated"

    families = {}

    def finish(name, rows, boot_key):
        for r, a in zip(rows, holm([r["p_calibrated"] for r in rows])):
            r["holm_calibrated"] = float(a)
            r["decision_calibrated"] = bool(a < ALPHA)
            r["changed"] = r["decision_calibrated"] != r["decision_bootstrap"]
        families[name] = {"n": len(rows), "n_significant_bootstrap": sum(r["decision_bootstrap"] for r in rows),
                          "n_significant_calibrated": sum(r["decision_calibrated"] for r in rows),
                          "changed": [r["test"] for r in rows if r["changed"]], "rows": rows}
        log(f"  {name}: bootstrap {families[name]['n_significant_bootstrap']} significant, calibrated "
            f"{families[name]['n_significant_calibrated']}, changed {families[name]['changed']}")

    for fam, corpora in (("unbc_shape_36", ("unbc",)), ("aw2_shape_18", ("aw2_image", "aw2_video"))):
        rows = []
        for c in C7["cells"]:
            if c["corpus"] not in corpora:
                continue
            key = {"unbc": "unbc", "aw2_image": "aw2img", "aw2_video": "aw2vid"}[c["corpus"]]
            y, g, keep = {"unbc": (uy, ug, ukeep), "aw2img": (iy, ig, imask), "aw2vid": (vy, vg, vmask)}[key]
            dname = {"unbc": "unbc_shape", "aw2img": "aw2_full", "aw2vid": "aw2_subset"}[key]
            s = W7[f"{key}__{c['cell']}"]; k = keep & ~np.isnan(s)
            auc, z = z_single(s[k], y[k], g[k])
            p, how = p_used(z, "single", dname)
            rows.append({"test": f"{c['corpus']}:{c['cell']}", "auc": auc, "z": z, "p_calibrated": p, "p_method": how,
                         "holm_bootstrap": c["holm_pooled"]["boot_subject"],
                         "decision_bootstrap": bool(c["holm_pooled"]["boot_subject"] < ALPHA)})
        finish(fam, rows, "boot_subject")

    rows = []
    for pr in C7["cross_corpus"]:
        cell = pr["cell"]
        su = W7[f"unbc__{cell}"]; ku = ukeep & ~np.isnan(su)
        if f"aw2img__{cell}" in W7.files:
            sw, yw, gw, kw, dname = W7[f"aw2img__{cell}"], iy, ig, imask, "aw2_full"
        else:
            sw, yw, gw, kw, dname = W7[f"aw2vid__{cell}"], vy, vg, vmask, "aw2_subset"
        kw = kw & ~np.isnan(sw)
        diff, z = z_between(su[ku], uy[ku], ug[ku], sw[kw], yw[kw], gw[kw])
        p, how = p_used(z, "between", dname)
        rows.append({"test": f"cross_corpus:{cell}", "diff": diff, "z": z, "p_calibrated": p, "p_method": how,
                     "holm_bootstrap": pr["p_holm"], "decision_bootstrap": bool(pr["p_holm"] < ALPHA)})
    finish("cross_corpus_18", rows, "p_holm")

    rows = []
    for c in C9["contrasts"]:
        diff, z = z_paired(W9[c["a"]][dkeep], W9[c["b"]][dkeep], W9["y"][dkeep], W9["subject"][dkeep])
        p, how = p_used(z, "paired", "unbc_direction")
        rows.append({"test": c["label"], "diff": diff, "z": z, "p_calibrated": p, "p_method": how,
                     "holm_bootstrap": c["holm"]["boot_subject"], "decision_bootstrap": bool(c["holm"]["boot_subject"] < ALPHA)})
    finish("direction_contrasts_24", rows, "boot_subject")

    rows = []
    for name, (so, sc, y, g, dname) in {
            "vjepa_K16:shuffled": (W8["vjepa_K16__ordered"], shuf_v, ay[idx], lab["subject"][idx], "aw2_subset"),
            "vjepa_K16:static": (W8["vjepa_K16__ordered"], W8["vjepa__static"], ay[idx], lab["subject"][idx], "aw2_subset"),
            "dinov2_K2:shuffled": (W8["dinov2_K2__ordered"], shuf_d, ay, lab["subject"], "aw2_full"),
            "dinov2_K2:static": (W8["dinov2_K2__ordered"], W8["dinov2__static"], ay, lab["subject"], "aw2_full")}.items():
        k = ~np.isnan(so) & ~np.isnan(sc)
        diff, z = z_paired(so[k], sc[k], y[k], g[k])
        p, how = p_used(z, "paired", dname)
        t = C8["tests"][name]
        rows.append({"test": name, "diff": diff, "z": z, "p_calibrated": p, "p_method": how,
                     "holm_bootstrap": t["p_holm"], "decision_bootstrap": bool(t["p_holm"] < ALPHA)})
    finish("w8_order_controls_4", rows, "p_holm")

    rows = []
    for t in C11["tests"]:
        k_ = t["K"]
        other = W11[f"shuf_K{k_}"] if t["test"].endswith("shuffled") else W11["static"]
        diff, z = z_paired(W11[f"ord_K{k_}"][wkeep], other[wkeep], W11["y"][wkeep], W11["subject"][wkeep])
        p, how = p_used(z, "paired", "unbc_shape")
        rows.append({"test": t["test"], "role": t["role"], "diff": diff, "z": z, "p_calibrated": p, "p_method": how,
                     "p_bootstrap": t["subject_boot"]["p_boot"], "decision_bootstrap": bool(t["subject_boot"]["p_boot"] < ALPHA),
                     "decision_calibrated": bool(p < ALPHA)})
        rows[-1]["changed"] = rows[-1]["decision_calibrated"] != rows[-1]["decision_bootstrap"]
    families["w11_exploratory_5"] = {"n": len(rows), "changed": [r["test"] for r in rows if r["changed"]], "rows": rows}
    log(f"  w11_exploratory_5: changed {families['w11_exploratory_5']['changed']}")

    changed = {f: v["changed"] for f, v in families.items() if v["changed"]}
    out["families"] = families
    out["summary"] = {"n_primary_decisions": sum(v["n"] for f, v in families.items() if f != "w11_exploratory_5"),
                      "changed": changed, "robust": not any(v for f, v in changed.items() if f != "w11_exploratory_5")}
    out["status"] = "COMPLETE"
    json.dump(out, open(out_json, "w"), indent=2)
    log(f"status {out['status']} | changed {changed}")
    log(f"wrote {out_json}")
    logf.close()


if __name__ == "__main__":
    main()
