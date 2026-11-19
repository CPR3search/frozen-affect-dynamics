"""Order controls for the Aff-Wild2 peak-task detections.

Usage:  python -m affectdyn.analyses.peak.affwild_peak_order_controls perms
        $CUDA_PY -m affectdyn.analyses.peak.affwild_peak_order_controls extract [--out-dir DIR]
        CUDA_VISIBLE_DEVICES="" python -m affectdyn.analyses.peak.affwild_peak_order_controls probe
"""
from __future__ import annotations
import os, sys

PROBE_THREADS = 4
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE == "probe":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # Disable CUDA before importing torch.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(PROBE_THREADS)

import argparse, datetime, hashlib, json, subprocess, time, warnings, zlib

import numpy as np

from affectdyn.paths import REPO_STR as REPO

T = 16
N_CLIPS = 13276
N_SHUF = 3
PERM_SEED = 800                   # Seed replicate r with PERM_SEED + r.
NBOOT = 10000
SEED = 20260911
ALPHA = 0.05
TOL = 0.005
COS_MIN = 0.9999
N_REPRO = 32
DINO_KS = (2, 4, 8, 16)
GATE_CELLS = ("vjepa_K16", "dinov2_K2")
PRIMARY = ("vjepa_K16:shuffled", "vjepa_K16:static", "dinov2_K2:shuffled", "dinov2_K2:static")
MEM_FRAC = 0.35                   # Limit allocation to this fraction of the 24 GB device.
OTHER_MIB = 200                   # Treat processes exceeding this allocation as competing GPU jobs.
LOW_FREE_MIB = 2048
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-11 — W8 PRE-DECLARED CRITERIA"


def safe_dir(path):
    od = os.path.abspath(path)
    for forbidden in ("runs/pain_encoder_maps", "runs/affwild_clips_features"):
        f = os.path.join(REPO, forbidden)
        if od == f or od.startswith(f + os.sep):
            raise SystemExit(f"refusing to write under {forbidden}: the manuscript is validated against it")
    os.makedirs(od, exist_ok=True)
    return od


class Log:
    def __init__(self, path):
        self.f = open(path, "a"); self.t0 = time.time()

    def __call__(self, m=""):
        line = f"[{time.time() - self.t0:7.1f}s] {m}"
        print(line, flush=True); self.f.write(line + "\n"); self.f.flush()


def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip()
    except Exception:                                                          # noqa: BLE001
        return "unavailable"


def base_provenance():
    return {"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__,
            "argv": sys.argv, "niceness": os.nice(0)}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def save_npz_atomic(path, compressed=True, **arrays):
    tmp = path[:-4] + ".tmp.npz"
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def load_perms(perms_dir):
    path = os.path.join(perms_dir, "perms.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run the perms mode first")
    perms = np.load(path)["perms"]
    if perms.shape != (N_SHUF, N_CLIPS, T):
        raise SystemExit(f"{path} has shape {perms.shape}, expected {(N_SHUF, N_CLIPS, T)}")
    return perms


def criteria_verbatim():
    log = os.path.join(REPO, "RESEARCH_LOG-D3-W1.md")
    if not os.path.exists(log):   # Allow the optional criteria file to be absent.
        return None
    lines = open(log, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(CRITERIA_HEADING))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    return "\n".join(lines[start:end]).strip()


def cmd_perms(args):
    od = safe_dir(args.out_dir)
    path = os.path.join(od, "perms.npz")
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} exists; refusing to overwrite without --force")
    n = len(np.load("runs/affwild_clips_features/labels.npz")["subject"])
    if n != N_CLIPS:
        raise SystemExit(f"labels carry {n} clips, expected {N_CLIPS}")
    ident, rev = np.arange(T), np.arange(T)[::-1]
    perms = np.empty((N_SHUF, n, T), np.int64)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    stats = []
    for r in range(N_SHUF):
        rng = np.random.default_rng(PERM_SEED + r)            # Create one generator per replicate.
        P = np.stack([rng.permutation(T) for _ in range(n)])
        hit = (P == ident).all(1)
        P[hit] = rev
        perms[r] = P
        fixed = (P[:, 0] == 0) & (P[:, -1] == T - 1)
        swapped = (P[:, 0] == T - 1) & (P[:, -1] == 0)
        stats.append({"replicate": r, "seed": PERM_SEED + r, "identity_rows_reversed": int(hit.sum()),
                      "endpoints_unchanged_all": int(fixed.sum()), "endpoints_swapped_all": int(swapped.sum()),
                      "endpoints_unchanged_vjepa_subset": int(fixed[idx].sum()),
                      "endpoints_swapped_vjepa_subset": int(swapped[idx].sum())})
    np.savez_compressed(path, perms=perms, seeds=np.array([PERM_SEED + r for r in range(N_SHUF)]))
    meta = {"provenance": base_provenance(), "shape": list(perms.shape), "sha": sha(perms),
            "rule": ("one generator per replicate, created once and drawn per clip in clip order; any "
                     "identity row is replaced by the reversal"),
            "replicates": stats}
    json.dump(meta, open(os.path.join(od, "perms.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


def gpu_state():
    def q(fields):
        return subprocess.run(["nvidia-smi", fields, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    name, used, total = [s.strip() for s in q("--query-gpu=name,memory.used,memory.total")[0].split(",")]
    apps = []
    for line in q("--query-compute-apps=pid,process_name,used_memory"):
        parts = [s.strip() for s in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit():
            apps.append({"pid": int(parts[0]), "name": parts[1], "used_mib": int(parts[2])})
    return {"name": name, "used_mib": int(used), "total_mib": int(total), "apps": apps}


def cmd_extract(args):
    import torch
    od = safe_dir(args.out_dir)
    log = Log(os.path.join(od, "extract.log"))
    out_npz = os.path.join(od, "vjepa_shuffled_full.npz")
    ck = os.path.join(od, "vjepa_shuffled_full.partial.npz")
    out_json = os.path.join(od, "extract.json")
    if os.path.exists(out_npz) and not args.force:
        raise SystemExit(f"{out_npz} exists; refusing to overwrite without --force")
    if not torch.cuda.is_available():
        raise SystemExit("extract needs a CUDA device; run it under a CUDA build of torch")
    g0 = gpu_state()
    busy = [a for a in g0["apps"] if a["used_mib"] > OTHER_MIB]
    if busy:
        raise SystemExit(f"refusing to start: another process holds GPU memory {busy}")
    torch.cuda.set_per_process_memory_fraction(MEM_FRAC, 0)
    torch.set_num_threads(2)
    import transformers
    from affectdyn.data.affwild_valence_clips import build_clips
    from affectdyn.features.affwild_video_features import _loader

    log(f"W8 extract | torch {torch.__version__} cuda {torch.version.cuda} transformers "
        f"{transformers.__version__} | {g0['name']} {g0['used_mib']}/{g0['total_mib']} MiB before start")
    perms = load_perms(args.perms_dir)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    lab = np.load("runs/affwild_clips_features/labels.npz")
    clips = build_clips()
    if len(clips) != N_CLIPS:
        raise SystemExit(f"build_clips gave {len(clips)} clips, expected {N_CLIPS}")
    bad = [int(i) for i in idx if str(clips[i]["subject"]) != str(lab["subject"][i])
           or int(clips[i]["traj"]) != int(lab["traj"][i]) or len(clips[i]["paths"]) != T]
    if bad:
        raise SystemExit(f"{len(bad)} subset clips disagree with runs/affwild_clips_features/labels.npz, e.g. {bad[:5]}")
    sub = [clips[i] for i in idx]
    n = len(sub)
    P = perms[:, idx]                                          # Selected permutations have shape [R, 2500, 16].
    cached = np.load("runs/affwild_clips_features/vjepa_windows.npz")["full"]
    D = cached.shape[1]

    fwd, prep, nf = _loader("vjepa")
    if nf != T:
        raise SystemExit(f"V-JEPA-2 loader reports {nf} frames, expected {T}")
    ref = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--vjepa2-vitl-fpc64-256/refs/main")
    revision = open(ref).read().strip() if os.path.exists(ref) else "unknown"

    def frames(c):                     # Match the video extraction's ordered 16-frame input.
        return np.stack([prep(p) for p in c["paths"]])

    def emb(x):
        return fwd(torch.from_numpy(np.ascontiguousarray(x)).to("cuda")).float().cpu().numpy()

    nr = min(N_REPRO, n)
    R = np.stack([emb(frames(sub[j])) for j in range(nr)])
    C = cached[:nr]
    cos = (R * C).sum(1) / (np.linalg.norm(R, axis=1) * np.linalg.norm(C, axis=1))
    repro = {"n_clips": int(nr), "min_cosine": float(cos.min()), "median_cosine": float(np.median(cos)),
             "max_abs_diff": float(np.abs(R - C).max()),
             "max_rel_l2": float((np.linalg.norm(R - C, axis=1) / np.linalg.norm(C, axis=1)).max()),
             "threshold_min_cosine": COS_MIN, "passes": bool(cos.min() >= COS_MIN)}
    log(f"reproduction on {nr} ordered clips: {repro}")
    json.dump({"provenance": base_provenance(), "reproduction": repro}, open(os.path.join(od, "extract_repro.json"), "w"), indent=2)
    if not repro["passes"]:
        log("pipeline gate (ii) FAILED: re-extracted embeddings do not match the cache; no shuffled extraction")
        raise SystemExit(2)

    perms_sha = sha(perms)
    full_shuf = np.full((N_SHUF, len(sub), D), np.nan, np.float32); start = 0
    if os.path.exists(ck):
        z = np.load(ck)
        if str(z["perms_sha"]) != perms_sha:
            raise SystemExit(f"{ck} was built from a different permutation table; remove it to restart")
        full_shuf, start = z["full_shuf"], int(z["n_done"])
        log(f"resuming from checkpoint at clip {start}")
    t0 = time.time()
    for j in range(start, n):
        x = frames(sub[j])
        for r in range(N_SHUF):
            full_shuf[r, j] = emb(x[P[r, j]])
        done = j + 1
        if done % 100 == 0 or done == n:
            el = time.time() - t0
            log(f"{done}/{n}  {el:.0f}s, ~{el / (done - start) * (n - done) / 60:.1f} min left, "
                f"peak alloc {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
        if done % 250 == 0 or done == n:
            save_npz_atomic(ck, compressed=False, full_shuf=full_shuf, n_done=np.array(done),
                            perms_sha=np.array(perms_sha))
            st = gpu_state()
            others = [a for a in st["apps"] if a["pid"] != os.getpid() and a["used_mib"] > OTHER_MIB]
            if others:
                log(f"WARNING another GPU process appeared: {others}")
                if st["total_mib"] - st["used_mib"] < LOW_FREE_MIB:
                    log(f"free GPU memory below {LOW_FREE_MIB} MiB; checkpoint saved at {done}, exiting")
                    raise SystemExit(3)

    res = full_shuf[:, :n]
    if np.isnan(res).any():
        raise SystemExit("NaN left in the shuffled embeddings")
    save_npz_atomic(out_npz, full_shuf=res, idx=idx[:n], perms_subset=P[:, :n], perms_sha=np.array(perms_sha),
                    n_done=np.array(n))
    meta = {"provenance": {**base_provenance(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                           "transformers": transformers.__version__, "gpu": g0["name"],
                           "hf_revision_vjepa2": revision, "memory_fraction_cap": MEM_FRAC,
                           "peak_alloc_mib": float(torch.cuda.max_memory_allocated() / 2**20)},
            "model": "facebook/vjepa2-vitl-fpc64-256, last_hidden_state mean over tokens (stage_w3_video._loader)",
            "input": "the 16 subset frames at 256 px, each clip's frames reindexed by perms[r, subset index]",
            "n_clips": int(n), "limit": None, "complete": bool(n == len(sub)), "replicates": N_SHUF,
            "perms_sha": perms_sha, "full_shuf_sha": sha(res), "seconds": round(time.time() - t0, 1),
            "reproduction": repro}
    json.dump(meta, open(out_json, "w"), indent=2)
    if os.path.exists(ck):
        os.remove(ck)
    log(f"wrote {out_npz} and {out_json}")


def paired_gap(so, sc, y, groups, rng, nboot, members, auc):
    """AUC(ordered) - AUC(control), whole subjects resampled, the same resample serving both."""
    G = len(members)
    vals = []
    for _ in range(nboot):
        i = np.concatenate([members[g] for g in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(auc(so[i], y[i]) - auc(sc[i], y[i]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return {"gap": float(auc(so, y) - auc(sc, y)), "ci": [float(lo), float(hi)],
            "p_boot": float(min(1.0, max(p, 1.0 / len(v)))), "resamples_skipped": int(nboot - len(v))}


def cmd_probe(args):
    from affectdyn.inference.peak_cluster_inference import torch_cpu_only, bootstrap, _members
    from affectdyn.features.unbc_pretraining_comparison import _auc
    from affectdyn.features.temporal_aggregation import image_readout, video_readout, N_REP
    from affectdyn.inference.aggregation_permutation_tests import holm
    from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof, N_SEED

    od = safe_dir(args.out_dir)
    torch = torch_cpu_only()
    torch.set_num_threads(PROBE_THREADS)
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    log = Log(os.path.join(od, "order_controls.log"))
    out_json = os.path.join(od, "order_controls.json")
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")

    ks = DINO_KS
    reps = N_SHUF
    seeds_img = [100 + r for r in range(N_SEED)]
    seeds_vid = [300 + r for r in range(N_REP)]
    nboot = NBOOT
    perms = load_perms(args.perms_dir)
    meta = {"perms_sha": sha(perms), "seeds_img": seeds_img, "seeds_vid": seeds_vid, "reps": reps, "smoke": False}
    log(f"W8 probe | torch {torch.__version__} cuda_build={torch.version.cuda} threads={PROBE_THREADS} "
        f"niceness={os.nice(0)} | {meta}")

    sc_path = os.path.join(od, "order_controls_oof.npz")
    S = {}
    if os.path.exists(sc_path):
        z = np.load(sc_path)
        if json.loads(str(z["__meta"])) != meta:
            raise SystemExit(f"{sc_path} was fitted under different settings; remove it to refit")
        S = {k: z[k] for k in z.files if k != "__meta"}
        log(f"loaded {len(S)} cached score vectors")

    def fit(key, build, y, assign, mask, seeds):
        if key in S:
            return S[key]
        t = time.time(); X = build()
        S[key] = np.nanmean([oof(X, y, assign, mask, sd, "cpu") for sd in seeds], axis=0).astype(np.float32)
        save_npz_atomic(sc_path, __meta=np.array(json.dumps(meta)), **S)
        log(f"  fit {key:28s} dim {X.shape[1]:5d}  {time.time() - t:6.1f}s")
        return S[key]

    # Evaluate DINOv2 on all clips using the image-probe procedure.
    lab = np.load("runs/affwild_clips_features/labels.npz")
    a_sub, a_traj = lab["subject"], lab["traj"]
    a_y = (a_traj == 2).astype(int); a_mask = a_traj != -1
    a_assign = grouped_folds(a_sub)
    fr = np.load("runs/affwild_clips_features/img_frames.npz")["dinov2"]
    if fr.shape[:2] != (len(a_sub), T):
        raise SystemExit(f"dinov2 frames {fr.shape} do not match {len(a_sub)} labels")
    rows = np.arange(len(a_sub))[:, None]
    for k in ks:
        fit(f"dinov2_K{k}__ordered", lambda k=k: image_readout(fr, k), a_y, a_assign, a_mask, seeds_img)
    fit("dinov2__static", lambda: fr.mean(1), a_y, a_assign, a_mask, seeds_img)
    for r in range(reps):
        shuffled = {}
        for k in ks:
            def build(k=k, r=r):
                if "f" not in shuffled:
                    shuffled["f"] = fr[rows, perms[r]]
                return image_readout(shuffled["f"], k)
            fit(f"dinov2_K{k}__shuffled_r{r}", build, a_y, a_assign, a_mask, seeds_img)
        shuffled.clear()
    del fr

    # Evaluate V-JEPA-2 on 2,500 clips using the video-probe procedure.
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    v_sub, v_traj = a_sub[idx], a_traj[idx]
    v_y = (v_traj == 2).astype(int); v_mask = v_traj != -1
    v_assign = grouped_folds(v_sub)
    z = np.load("runs/affwild_clips_features/vjepa_windows.npz"); full, W = z["full"], z["K16_up"]
    fit("vjepa_K16__ordered", lambda: video_readout(full, W), v_y, v_assign, v_mask, seeds_vid)
    fit("vjepa__static", lambda: W.mean(1), v_y, v_assign, v_mask, seeds_vid)
    fit("vjepa__full_only", lambda: full, v_y, v_assign, v_mask, seeds_vid)

    ext_npz = os.path.join(args.extract_dir, "vjepa_shuffled_full.npz")
    ext_json = os.path.join(args.extract_dir, "extract.json")
    extraction = None
    if os.path.exists(ext_npz) and os.path.exists(ext_json):
        e, ej = np.load(ext_npz), json.load(open(ext_json))
        P = perms[:, idx]
        ok = (ej.get("complete") and e["full_shuf"].shape == (N_SHUF, len(idx), full.shape[1])
              and np.array_equal(e["idx"], idx) and np.array_equal(e["perms_subset"], P)
              and str(e["perms_sha"]) == meta["perms_sha"] and not np.isnan(e["full_shuf"]).any())
        if not ok:
            raise SystemExit(f"{ext_npz} is incomplete or does not match this permutation table and subset")
        extraction = {"full_shuf": e["full_shuf"], "meta": ej}
    if extraction is not None:
        vrows = np.arange(len(idx))
        for r in range(reps):
            fit(f"vjepa_K16__shuffled_r{r}",
                lambda r=r: np.concatenate([extraction["full_shuf"][r], W[vrows, P[r, :, -1]] - W[vrows, P[r, :, 0]]], 1),
                v_y, v_assign, v_mask, seeds_vid)
    else:
        log(f"  V-JEPA-2 shuffled extraction not found at {ext_npz}; its shuffled control is pending")

    data = {"dinov2": (a_y, a_sub, a_mask), "vjepa": (v_y, v_sub, v_mask)}

    def rng_for(name):
        return np.random.default_rng([SEED, zlib.crc32(name.encode())])

    def prepared(enc, *vecs):
        y, g, mask = data[enc]
        keep = mask.copy()
        for v in vecs:
            keep &= ~np.isnan(v)
        return keep, y[keep].astype(np.int64), g[keep]

    ordered, tests = {}, {}
    cells = [(f"dinov2_K{k}", "dinov2", "dinov2__static") for k in ks] + [("vjepa_K16", "vjepa", "vjepa__static")]
    for cell, enc, static_key in cells:
        so = S[f"{cell}__ordered"]
        keep, y, g = prepared(enc, so)
        ordered[cell] = {"auc": float(_auc(so[keep], y)), "n": int(len(y)), "n_pos": int(y.sum()),
                         "n_subjects": int(len(np.unique(g))),
                         "subject_bootstrap": bootstrap(so[keep], y, g, "subject", rng_for(f"{cell}:ordered"), nboot)}
        log(f"  {cell:11s} ordered AUC {ordered[cell]['auc']:.4f}  CI {ordered[cell]['subject_bootstrap']['ci']}")
        controls = {"static": S[static_key]}
        sh = [S.get(f"{cell}__shuffled_r{r}") for r in range(reps)]
        if all(s is not None for s in sh):
            controls["shuffled"] = np.mean(sh, axis=0)
        if enc == "vjepa":
            controls["full_only (descriptive)"] = S["vjepa__full_only"]
        for ctrl, sc in controls.items():
            name = f"{cell}:{ctrl}"
            keep, y, g = prepared(enc, so, sc)
            gap = paired_gap(so[keep], sc[keep], y, g, rng_for(name), nboot, _members(g), _auc)
            tests[name] = {"cell": cell, "control": ctrl, "auc_ordered": float(_auc(so[keep], y)),
                           "auc_control": float(_auc(sc[keep], y)), "n": int(len(y)),
                           "control_vs_chance": bootstrap(sc[keep], y, g, "subject", rng_for(name + ":chance"), nboot),
                           **gap,
                           "role": ("primary" if name in PRIMARY else
                                    "descriptive" if "descriptive" in ctrl else "exploratory, uncorrected")}
            if ctrl == "shuffled":
                tests[name]["shuffled_replicate_aucs"] = [float(_auc(s[keep], y)) for s in sh]
            log(f"  {name:36s} ordered {tests[name]['auc_ordered']:.4f} control {tests[name]['auc_control']:.4f} "
                f"gap {gap['gap']:+.4f} CI {gap['ci'][0]:+.4f} to {gap['ci'][1]:+.4f} p {gap['p_boot']:.4f}")

    ref = {c["cell"]: c["auc"] for c in json.load(open("runs/peak_cluster_inference/cluster.json"))["cells"]
           if c["corpus"] in ("aw2_image", "aw2_video")}
    gate_i = {c: {"auc": ordered[c]["auc"], "reference_w7": float(ref[c]), "abs_delta": abs(ordered[c]["auc"] - ref[c]),
                  "passes": bool(abs(ordered[c]["auc"] - ref[c]) <= TOL)} for c in GATE_CELLS if c in ordered}
    gate_ii = extraction["meta"]["reproduction"] if extraction else None
    gate = {"tolerance_auc": TOL, "i_ordered_auc_reproduces_w7": gate_i, "ii_vjepa_reextraction_matches_cache": gate_ii,
            "passes": bool(len(gate_i) == len(GATE_CELLS) and all(v["passes"] for v in gate_i.values())
                           and gate_ii is not None and gate_ii.get("passes"))}
    exploratory_repro = {f"dinov2_K{k}": {"auc": ordered[f"dinov2_K{k}"]["auc"], "reference_w7": float(ref[f"dinov2_K{k}"])}
                         for k in ks}

    have = [n for n in PRIMARY if n in tests]
    verdicts, status = {}, "PENDING"
    if len(have) == len(PRIMARY):
        for n, a in zip(PRIMARY, holm([tests[n]["p_boot"] for n in PRIMARY])):
            tests[n]["p_holm"] = float(a)
            tests[n]["significant_positive"] = bool(tests[n]["gap"] > 0 and a < ALPHA)
            tests[n]["significant_negative"] = bool(tests[n]["gap"] < 0 and a < ALPHA)
        if not gate["passes"]:
            status = "HOLD: pipeline gate failed, no verdict is read"
        else:
            status = "COMPLETE"
            for enc in ("vjepa_K16", "dinov2_K2"):
                s_sig = tests[f"{enc}:shuffled"]["significant_positive"]
                a_sig = tests[f"{enc}:static"]["significant_positive"]
                verdicts[enc] = ("TEMPORAL BEYOND APPEARANCE" if s_sig and a_sig else
                                 "ORDER-DEPENDENT ONLY" if s_sig else "NOT SHOWN TO REST ON ORDER")

    out = {"provenance": {**base_provenance(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                          "cuda_available": torch.cuda.is_available(), "threads": PROBE_THREADS, "nboot": nboot,
                          "seed": SEED, **meta,
                          "extraction": ({k: v for k, v in extraction["meta"].items() if k != "reproduction"}
                                         if extraction else None)},
           "criteria_verbatim": criteria_verbatim(),
           "status": status, "pipeline_gate": gate, "verdicts": verdicts,
           "primary_family": list(PRIMARY), "ordered": ordered, "tests": tests,
           "exploratory_reproduction": exploratory_repro}
    path = out_json if status == "COMPLETE" else os.path.join(od, "order_controls.partial.json")
    json.dump(out, open(path, "w"), indent=2)
    log(f"status {status} | gate {gate['passes']} | verdicts {verdicts} | wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("perms", "extract", "probe"))
    ap.add_argument("--out-dir", default="runs/affwild_peak_order_controls")
    ap.add_argument("--perms-dir", default="runs/affwild_peak_order_controls")
    ap.add_argument("--extract-dir", default="runs/affwild_peak_order_controls")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit(f"run from the repository root {REPO}")
    {"perms": cmd_perms, "extract": cmd_extract, "probe": cmd_probe}[args.mode](args)


if __name__ == "__main__":
    main()
