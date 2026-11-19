"""Power of the subject bootstrap for the peak-task families.

Usage:  python -m affectdyn.inference.bootstrap_detection_power
"""
from __future__ import annotations
import os

N_THREADS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import argparse, datetime, json, subprocess, sys, time

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.permutation_detection_limits import _probit

from affectdyn.paths import REPO_STR as REPO
ALPHA = 0.05
POWER = 0.80
B = 10000
N_SIM = 500                        # Simulate this many studies per bisection step.
N_NULL_SIM = 2000                  # Simulate null studies at theta = 0.5.
STEPS = 10                         # Bisect theta over [0.5, 0.95].
ICC_PRIMARY, ICC_SECONDARY = 0.3, 0.0
SEED = 20260911
CHUNK = 2500
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-11 — W10 PRE-DECLARED CRITERIA"


class ClipSet:
    """Labels and subject groups of one corpus's clip set, with the index layout the fast AUC needs."""

    def __init__(self, name, y, groups, m):
        self.name, self.m = name, m
        self.y = np.asarray(y).astype(np.int64)
        uniq, self.inv = np.unique(groups, return_inverse=True)
        self.G = len(uniq)
        self.pos = np.flatnonzero(self.y == 1)
        neg = np.flatnonzero(self.y == 0)
        self.neg_sorted = neg[np.argsort(self.inv[neg], kind="stable")]
        gneg = self.inv[self.neg_sorted]
        self.neg_starts = np.flatnonzero(np.r_[True, gneg[1:] != gneg[:-1]])
        self.neg_subjects = gneg[self.neg_starts]
        self.P = np.bincount(self.inv[self.pos], minlength=self.G).astype(np.float64)
        self.N = np.bincount(self.inv[neg], minlength=self.G).astype(np.float64)
        self.n1, self.n0 = len(self.pos), len(neg)
        self.members = [np.flatnonzero(self.inv == g) for g in range(self.G)]

    def describe(self):
        return {"n_clips": int(len(self.y)), "n_positive": int(self.n1), "n_subjects": int(self.G),
                "n_subjects_with_a_positive": int((self.P > 0).sum()), "holm_family": self.m,
                "alpha_over_m": ALPHA / self.m}

    def counts(self, s):
        """C[g, h]: pairs (positive clip of subject g, negative clip of subject h) with s_pos > s_neg."""
        cmp = (s[self.pos][:, None] > s[self.neg_sorted][None, :]).astype(np.float32)
        blocks = np.add.reduceat(cmp, self.neg_starts, axis=1).astype(np.float64)
        rows = np.zeros((self.G, blocks.shape[1]))
        np.add.at(rows, self.inv[self.pos], blocks)
        C = np.zeros((self.G, self.G))
        C[:, self.neg_subjects] = rows
        return C


def boot_aucs(cs, C, rng, nboot):
    """Pooled AUC of nboot subject resamples, NaN where a resample holds one class only."""
    out = np.empty(nboot)
    G = cs.G
    for i in range(0, nboot, CHUNK):
        b = min(CHUNK, nboot - i)
        draw = rng.integers(0, G, size=(b, G))
        W = np.bincount((draw + (np.arange(b) * G)[:, None]).ravel(),
                        minlength=b * G).reshape(b, G).astype(np.float64)
        num = np.einsum("bj,bj->b", W @ C, W)
        den = (W @ cs.P) * (W @ cs.N)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i:i + b] = np.where(den > 0, num / den, np.nan)
    return out


def p_two_sided(v):
    """As stage_w7_cluster.bootstrap: twice the smaller share on either side of 0.5, floored at 1/B."""
    v = v[~np.isnan(v)]
    p = 2.0 * min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return float(min(1.0, max(p, 1.0 / len(v))))


def simulate(cs, theta, icc, rng):
    """stage_w7_mde.power_at's model: binormal separation giving P(pos > neg) = theta, subject effect."""
    mu = np.sqrt(2.0) * _probit(theta)
    s = mu * cs.y + rng.standard_normal(len(cs.y)) * np.sqrt(max(1e-9, 1.0 - icc))
    if icc > 0:
        s = s + (rng.standard_normal(cs.G) * np.sqrt(icc))[cs.inv]
    return s


def study(cs, theta, icc, rng, nboot):
    C = cs.counts(simulate(cs, theta, icc, rng))
    return C.sum() / (cs.n1 * cs.n0), p_two_sided(boot_aucs(cs, C, rng, nboot))


def power(cs, theta, icc, rng, nsim, nboot):
    rej = 0
    for _ in range(nsim):
        auc, p = study(cs, theta, icc, rng, nboot)
        rej += int(auc > 0.5 and p <= ALPHA / cs.m)
    return rej / nsim


def mde_boot(cs, icc, rng, nsim, nboot, steps, log):
    lo, hi, trace = 0.5, 0.95, []
    for _ in range(steps):
        mid = (lo + hi) / 2
        pw = power(cs, mid, icc, rng, nsim, nboot)
        trace.append({"theta": round(mid, 5), "power": pw})
        log(f"    {cs.name:17s} icc {icc:.1f} theta {mid:.4f} power {pw:.3f}")
        if pw < POWER:
            lo = mid
        else:
            hi = mid
    return {"icc": icc, "mde_auc_at_80pc_power": round(hi, 4), "bracket": [round(lo, 4), round(hi, 4)],
            "n_sim_per_step": nsim, "trace": trace}


def calibration(cs, icc, rng, nsim, nboot):
    r05 = rholm = rpos = 0
    for _ in range(nsim):
        auc, p = study(cs, 0.5, icc, rng, nboot)
        r05 += int(p <= ALPHA)
        rholm += int(p <= ALPHA / cs.m)
        rpos += int(auc > 0.5 and p <= ALPHA / cs.m)
    rate05, rateh = r05 / nsim, rholm / nsim
    return {"icc": icc, "n_sim": nsim, "rate_at_0.05": rate05, "rate_at_alpha_over_m": rateh,
            "rate_positive_direction_at_alpha_over_m": rpos / nsim, "alpha_over_m": ALPHA / cs.m,
            "threshold_0.05": 0.075, "threshold_alpha_over_m": 3 * ALPHA / cs.m,
            "flag_anticonservative": bool(rate05 > 0.075 or rateh > 3 * ALPHA / cs.m)}


def gate_formula(cs, rng, n=200):
    """Largest |fast AUC - concatenated AUC| over n random subject resamples of one simulated study."""
    s = rng.standard_normal(len(cs.y))
    C = cs.counts(s)
    worst, used = 0.0, 0
    for _ in range(n):
        draw = rng.integers(0, cs.G, cs.G)
        i = np.concatenate([cs.members[g] for g in draw])
        if cs.y[i].min() == cs.y[i].max():
            continue
        W = np.bincount(draw, minlength=cs.G).astype(np.float64)
        fast = (W @ C @ W) / ((W @ cs.P) * (W @ cs.N))
        worst = max(worst, abs(_auc(s[i], cs.y[i]) - fast)); used += 1
    return {"max_abs_diff": float(worst), "resamples": used}


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
    ap.add_argument("--out-dir", default="runs/bootstrap_detection_power")
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
    out_json, out_log = os.path.join(od, "bootstrap_power.json"), os.path.join(od, "bootstrap_power.log")
    if not args.force and os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")

    nsim, nnull, nboot, steps = (N_SIM, N_NULL_SIM, B, STEPS)
    rng = np.random.default_rng(SEED)
    criteria = criteria_verbatim(CRITERIA_HEADING)
    logf = open(out_log, "w"); t0 = time.time()

    def log(msg=""):
        line = f"[{time.time() - t0:8.1f}s] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"W10 subject-bootstrap power | numpy {np.__version__} threads {N_THREADS} niceness {os.nice(0)} "
        f"| B {nboot} n_sim {nsim} null {nnull} steps {steps} smoke=False")

    u = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    ukeep = u["traj"] != -1
    sets = [ClipSet("unbc", u["traj"][ukeep] == 2, u["subject"][ukeep], 36)]
    a = np.load("runs/affwild_clips_features/labels.npz")
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    sets.append(ClipSet("aff_wild2_subset", a["traj"][idx] == 2, a["subject"][idx], 18))
    sets.append(ClipSet("aff_wild2_full", a["traj"] == 2, a["subject"], 18))
    for cs in sets:
        log(f"  {cs.name}: {cs.describe()}")

    gate_i = {cs.name: gate_formula(cs, rng) for cs in sets}
    z = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    keep = z["unbc__keep"].astype(bool)
    real = ClipSet("unbc_clip_K4_real", z["unbc__y"][keep], z["unbc__subject"][keep], 36)
    v = boot_aucs(real, real.counts(z["unbc__clip_K4"][keep]), np.random.default_rng([SEED, 7]), B)
    v_ok = v[~np.isnan(v)]
    ref = next(c for c in json.load(open("runs/peak_cluster_inference/cluster.json"))["cells"]
               if c["corpus"] == "unbc" and c["cell"] == "clip_K4")["subject"]
    lo, hi = np.percentile(v_ok, [2.5, 97.5])
    gate_ii = {"ci": [float(lo), float(hi)], "ci_reference": ref["ci"], "p": p_two_sided(v),
               "p_reference": ref["p_boot"],
               "max_abs_ci_delta": float(max(abs(lo - ref["ci"][0]), abs(hi - ref["ci"][1]))),
               "abs_p_delta": float(abs(p_two_sided(v) - ref["p_boot"]))}
    gate = {"i_fast_auc_equals_concatenated": gate_i, "ii_reproduces_w7_subject_bootstrap": gate_ii}
    gate["passes"] = bool(all(g["max_abs_diff"] < 1e-9 for g in gate_i.values())
                          and gate_ii["max_abs_ci_delta"] <= 0.01 and gate_ii["abs_p_delta"] <= 0.005)
    log(f"pipeline gate: {json.dumps(gate)}")

    mde_perm = json.load(open("runs/peak_cluster_inference/mde.json"))["corpora"]
    out = {"provenance": {"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                          "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                          "python": sys.version.split()[0], "numpy": np.__version__, "threads": N_THREADS,
                          "niceness": os.nice(0), "seed": SEED, "B": nboot, "n_sim_per_step": nsim,
                          "n_null_sim": nnull, "bisection_steps": steps, "smoke": False,
                          "model": "stage_w7_mde.power_at generative model; test as stage_w7_cluster.bootstrap"},
           "criteria_verbatim": criteria, "pipeline_gate": gate, "status": None, "clip_sets": {}}

    def save(status):
        out["status"] = status
        json.dump(out, open(out_json, "w"), indent=2)

    if not gate["passes"]:
        save("HOLD: pipeline gate failed, no result is read")
        log("pipeline gate FAILED; stopping")
        return

    for cs in sets:
        entry = cs.describe()
        w7 = mde_perm.get({"unbc": "unbc", "aff_wild2_full": "aff_wild2"}.get(cs.name, ""), {})
        entry["mde_label_permutation_w7"] = w7.get("results")
        log(f"  {cs.name}: calibration at theta 0.5")
        entry["calibration"] = calibration(cs, ICC_PRIMARY, rng, nnull, nboot)
        log(f"    {entry['calibration']}")
        for icc in (ICC_PRIMARY, ICC_SECONDARY):
            log(f"  {cs.name}: MDE_boot, icc {icc}")
            entry[f"mde_boot_icc{icc}"] = mde_boot(cs, icc, rng, nsim, nboot, steps, log)
        out["clip_sets"][cs.name] = entry
        save(f"RUNNING: {cs.name} done")
        log(f"  {cs.name} MDE_boot icc0.3 {entry['mde_boot_icc0.3']['mde_auc_at_80pc_power']} "
            f"icc0 {entry['mde_boot_icc0.0']['mde_auc_at_80pc_power']} | calibration {entry['calibration']}")
    save("COMPLETE")
    log(f"wrote {out_json}")
    logf.close()


if __name__ == "__main__":
    main()
