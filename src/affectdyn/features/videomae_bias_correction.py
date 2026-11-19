"""VideoMAE extraction with its attention biases restored, and VideoMAEv2 alongside."""
from __future__ import annotations
import json, os, sys, time

import numpy as np

from affectdyn.data.unbc_clip_pipeline import build_clips, T, _prep224
from affectdyn.features.temporal_aggregation import _device, _window_orders, KS, video_readout, N_REP
from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.inference.aggregation_permutation_tests import perm_test
from affectdyn.analyses.peak.pain_peak_encoder_map import label_perm_test
from affectdyn.features.videomae_loader import load_videomae, load_videomae_v2
from affectdyn.provenance import run_provenance, audit_loso

OUT = "runs/temporal_aggregation"


def _extract(tag, model_fn, layout, clips, cache):
    import torch
    if os.path.exists(cache):
        print(f"[{tag}] cache hit {cache}", flush=True)
        return cache
    dev = _device()
    m = model_fn(device=dev)

    def emb(order):
        x = torch.from_numpy(np.stack([_prep224(clips_paths[i]) for i in order])).to(dev)
        with torch.no_grad():
            if layout == "BTCHW":
                return m(pixel_values=x.unsqueeze(0)).last_hidden_state.mean(1).squeeze(0) \
                        .float().cpu().numpy()
            o = m(x.permute(1, 0, 2, 3).unsqueeze(0))
            t = o if hasattr(o, "shape") else o.last_hidden_state
            return t.squeeze(0).float().cpu().numpy()

    store = {f"K{k}_up": [] for k in KS}
    full = []
    t0 = time.time()
    for j, c in enumerate(clips):
        clips_paths = c["paths"]
        full.append(emb(list(range(T))))
        for k in KS:
            store[f"K{k}_up"].append(np.stack([emb(o) for o in _window_orders(k, upsample_to=T)]))
        if j % 100 == 0:
            print(f"[{tag}] {j + 1}/{len(clips)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, full=np.stack(full), **{k: np.stack(v) for k, v in store.items()})
    print(f"[{tag}] wrote {cache} ({time.time() - t0:.0f}s)", flush=True)
    return cache


def main():
    import torch
    logf = open(f"{OUT}/videomae_correction.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    clips = build_clips()
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub, traj, direction = lab["subject"], lab["traj"], lab["direction"]
    folds = _loso_folds(sub)
    audit_loso(sub, folds)

    log(f"VideoMAE correction and VideoMAEv2 addition | {len(clips)} UNBC clips, "
        f"{len(set(sub.tolist()))} subjects, K sweep {KS}")

    _extract("videomae-fixed", load_videomae, "BTCHW", clips, f"{OUT}/videomae_windows_fixed.npz")
    _extract("videomaev2", load_videomae_v2, "BCTHW", clips, f"{OUT}/videomaev2_windows.npz")

    caches = {"videomae (as submitted, biases zeroed)": f"{OUT}/videomae_windows.npz",
              "videomae (biases restored)": f"{OUT}/videomae_windows_fixed.npz",
              "videomae-v2": f"{OUT}/videomaev2_windows.npz"}
    tasks = {"direction": ((direction == 1).astype(int), direction != -1),
             "shape": ((traj == 2).astype(int), traj != -1)}

    def score(X, y, mask):
        acc = None
        for rep in range(N_REP):
            torch.manual_seed(1000 + rep)
            s = _oof_scores(X, y, sub, mask, folds)
            acc = s if acc is None else acc + s
        return acc / N_REP

    res = {"provenance": run_provenance(seeds=[1000 + r for r in range(N_REP)])}
    for task, (y, mask) in tasks.items():
        log(f"\n=== {task} ===")
        log(f"  {'variant':38s} " + "".join(f"K={k:<10d}" for k in KS))
        scores = {}
        for name, path in caches.items():
            if not os.path.exists(path):
                log(f"  {name:38s} (cache absent)"); continue
            z = np.load(path)
            row, scores[name] = f"  {name:38s} ", {}
            for k in KS:
                key = f"K{k}_up" if f"K{k}_up" in z.files else f"K{k}_native"
                if key not in z.files:
                    row += " " * 12; continue
                s = score(video_readout(z["full"], z[key]), y, mask)
                keep = mask & ~np.isnan(s)
                scores[name][k] = (s, keep)
                row += f"{_auc(s[keep], y[keep]):.3f}      "
            log(row)

        a, b = "videomae (as submitted, biases zeroed)", "videomae (biases restored)"
        if a in scores and b in scores:
            log(f"\n  correction effect, {task}:")
            for k in KS:
                if k in scores[a] and k in scores[b]:
                    keep = scores[a][k][1] & scores[b][k][1]
                    gap, p = perm_test(scores[b][k][0][keep], scores[a][k][0][keep], y[keep])
                    log(f"    K={k:2d}  restored minus zeroed = {gap:+.3f}   p={p:.4f}")
                    res.setdefault(f"{task}_correction", {})[f"K{k}"] = {"gap": gap, "p": p}
    json.dump(res, open(f"{OUT}/videomae_correction.json", "w"), indent=2)
    log(f"\nwrote {OUT}/videomae_correction.json")
    log("Whatever the size of the shift, it is disclosed rather than silently applied: the")
    log("uncorrected version is in a journal's records.")
    logf.close()


if __name__ == "__main__":
    main()
