"""Encoder map across pretraining objectives on UNBC; provides the AUC helper."""
from __future__ import annotations
import os, sys
import numpy as np

from affectdyn.data.unbc_clip_pipeline import build_clips, T

NORMS = {  # Store each encoder's normalization mean and standard deviation for 224px inputs.
    "dinov2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip":   ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
    "siglip": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


def _prep(path, mean, std):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB").resize((224, 224)), np.float32) / 255.0
    return ((a - np.array(mean, np.float32)) / np.array(std, np.float32)).transpose(2, 0, 1)


def extract(clips, out_dir):
    import torch
    from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "crosspretrain_feats.npz")
    if os.path.exists(cache):
        return cache
    torch.set_num_threads(8)
    encs = {
        "dinov2": AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float32).eval(),
        "clip": CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", dtype=torch.float32).eval(),
        "siglip": SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", dtype=torch.float32).eval(),
    }
    rng = np.random.default_rng(0)
    store = {e: {"mean": [], "tord": [], "tshuf": []} for e in encs}
    for j, c in enumerate(clips):
        sh = rng.permutation(T)
        for e, m in encs.items():
            mean, std = NORMS[e]
            x = torch.from_numpy(np.stack([_prep(p, mean, std) for p in c["paths"]]))
            with torch.no_grad():
                o = m(pixel_values=x)
                f = (o.pooler_output if getattr(o, "pooler_output", None) is not None
                     else o.last_hidden_state.mean(1)).numpy()       # The feature array has shape [T, D].
            store[e]["mean"].append(f.mean(0))
            store[e]["tord"].append(np.concatenate([f.mean(0), f[-1] - f[0]]))      # Append an order-sensitive endpoint difference.
            fs = f[sh]
            store[e]["tshuf"].append(np.concatenate([fs.mean(0), fs[-1] - fs[0]]))
        if j % 50 == 0:
            print(f"[s1b] {j+1}/{len(clips)} clips", flush=True)
    out = {}
    for e in encs:
        for k in store[e]:
            out[f"{e}_{k}"] = np.stack(store[e][k])
    np.savez_compressed(cache, **out)
    return cache


def _auc(s, y):
    """Historical ordinal-rank AUC used by the fixed-score analyses. Ties receive distinct ranks in
    NumPy argsort order, rather than half credit. See analyses.sensitivity.auc_tie_audit and the
    record runs/auc_tie_holm_decisions/tie_decisions.json for the effect of this convention.
    """
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _loso(X, y, sub, subs, mask, kind):
    import torch, torch.nn as nn
    out = []
    for st in subs:
        tr = (sub != st) & mask; te = (sub == st) & mask
        if te.sum() < (3 if kind == "corr" else 1) or len(set(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        reg = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-3)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        for _ in range(300):
            o = reg(xt).squeeze(-1)
            loss = (nn.functional.binary_cross_entropy_with_logits(o, yt) if kind == "auc"
                    else nn.functional.mse_loss(o, yt))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pr = reg(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
        if kind == "auc":
            v = _auc(pr, y[te])
        else:
            yy = y[te]
            v = (np.corrcoef(np.argsort(np.argsort(pr)), np.argsort(np.argsort(yy)))[0, 1]
                 if np.std(pr) > 0 and np.std(yy) > 0 else np.nan)
        if not np.isnan(v):
            out.append(v)
    return np.mean(out), np.std(out)


def probe(cache, vjepa_cache="runs/unbc_clip_pipeline/stage1_feats.npz"):
    cp = np.load(cache, allow_pickle=False)
    vj = np.load(vjepa_cache, allow_pickle=False)
    sub = vj["subject"]; subs = sorted(set(sub.tolist()))
    direction = (vj["direction"] == 1).astype(int); dmask = vj["direction"] != -1
    slope = vj["slope"]; amask = np.ones(len(slope), bool)
    # Map feature-set names to arrays and video flags.
    sets = {
        "V-JEPA-2 ordered": vj["vord"], "V-JEPA-2 shuffled": vj["vshuf"],
        "DINOv2 static": cp["dinov2_mean"], "DINOv2 temporal": cp["dinov2_tord"], "DINOv2 t-shuf": cp["dinov2_tshuf"],
        "CLIP static": cp["clip_mean"], "CLIP temporal": cp["clip_tord"], "CLIP t-shuf": cp["clip_tshuf"],
        "SigLIP static": cp["siglip_mean"], "SigLIP temporal": cp["siglip_tord"], "SigLIP t-shuf": cp["siglip_tshuf"],
    }
    print("\n=== Cross-pretraining map: P1 direction (AUC) | P4 velocity (Spearman) ===")
    for nm, X in sets.items():
        da, ds = _loso(X, direction, sub, subs, dmask, "auc")
        va, vs = _loso(X, slope, sub, subs, amask, "corr")
        print(f"  {nm:20s}  dir {da:.3f}±{ds:.3f}   vel {va:.3f}±{vs:.3f}")


if __name__ == "__main__":
    out = "runs/unbc_pretraining_comparison"
    clips = build_clips()
    print(f"[s1b] {len(clips)} clips (aligned to Stage-1)", flush=True)
    cache = extract(clips, out)
    probe(cache)
