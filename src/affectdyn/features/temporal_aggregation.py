"""Form-matched temporal aggregation for image and video encoders, and the aggregation sweep."""
from __future__ import annotations
import json, os, sys, time
import numpy as np

from affectdyn.data.unbc_clip_pipeline import build_clips, T, _prep224, _prep256
from affectdyn.features.unbc_pretraining_comparison import NORMS, _prep as _prep_img_norm, _auc
from affectdyn.provenance import run_provenance, audit_loso

OUT = "runs/temporal_aggregation"
KS = (2, 4, 8, 16)                     # Video interfaces constrain native window granularities.
N_REP = 5                              # Repeat each cell's probe with distinct seeds.
IMG_ENCODERS = ("clip", "dinov2", "siglip")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"



def extract_image_frames(clips, cache=f"{OUT}/img_frames.npz", bs=64):
    """Per-frame features [N, T, D] for each image encoder."""
    if os.path.exists(cache):
        return cache
    import torch
    from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    dev = _device()
    encs = {
        "dinov2": AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float32),
        "clip": CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", dtype=torch.float32),
        "siglip": SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", dtype=torch.float32),
    }
    encs = {k: v.eval().to(dev) for k, v in encs.items()}
    store = {e: [] for e in encs}
    t0 = time.time()
    for j, c in enumerate(clips):
        for e, m in encs.items():
            mean, std = NORMS[e]
            x = torch.from_numpy(np.stack([_prep_img_norm(p, mean, std) for p in c["paths"]])).to(dev)
            with torch.no_grad():
                outs = []
                for i in range(0, len(x), bs):
                    o = m(pixel_values=x[i:i + bs])
                    f = (o.pooler_output if getattr(o, "pooler_output", None) is not None
                         else o.last_hidden_state.mean(1))
                    outs.append(f.float().cpu().numpy())
            store[e].append(np.concatenate(outs, 0))                    # Store frame features with shape [T, D].
        if j % 50 == 0:
            print(f"[g0r/img] {j + 1}/{len(clips)} clips  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, **{e: np.stack(v) for e, v in store.items()})
    return cache


def _window_orders(k, upsample_to=None):
    """K contiguous windows over T frames, each optionally repeat-upsampled to a fixed length."""
    w = T // k
    orders = []
    for i in range(k):
        base = list(range(i * w, (i + 1) * w))
        if upsample_to:
            rep = upsample_to // len(base)
            base = [f for f in base for _ in range(rep)]
        orders.append(base)
    return orders


def extract_vjepa(clips, cache=f"{OUT}/vjepa_windows.npz"):
    if os.path.exists(cache):
        return cache
    import torch
    from transformers import VJEPA2Model
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    dev = _device()
    m = VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32).eval().to(dev)

    def emb(paths, order):
        x = torch.from_numpy(np.stack([_prep256(paths[i]) for i in order])).unsqueeze(0).to(dev)
        with torch.no_grad():
            return m(pixel_values_videos=x).last_hidden_state.mean(1).squeeze(0).float().cpu().numpy()

    ks_native = [k for k in KS if T // k >= 2]     # Tubelets require at least two frames per native window.
    ks_up = list(KS)                               # Upsampling supports every requested granularity.
    # At K=16, repeat each window's single frame to fill the encoder input.
    store = {f"K{k}_native": [] for k in ks_native}
    store.update({f"K{k}_up": [] for k in ks_up})
    full = []
    t0 = time.time()
    for j, c in enumerate(clips):
        full.append(emb(c["paths"], list(range(T))))
        for k in ks_native:
            store[f"K{k}_native"].append(np.stack([emb(c["paths"], o) for o in _window_orders(k)]))
        for k in ks_up:
            store[f"K{k}_up"].append(np.stack([emb(c["paths"], o)
                                               for o in _window_orders(k, upsample_to=T)]))
        if j % 25 == 0:
            print(f"[g0r/vjepa] {j + 1}/{len(clips)} clips  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, full=np.stack(full), **{k: np.stack(v) for k, v in store.items()})
    return cache


def extract_videomae(clips, cache=f"{OUT}/videomae_windows.npz"):
    """VideoMAE window embeddings."""
    if os.path.exists(cache):
        return cache
    import torch
    from affectdyn.features.videomae_loader import load_videomae
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    dev = _device()
    m = load_videomae(device=dev)

    def emb(paths, order):
        x = torch.from_numpy(np.stack([_prep224(paths[i]) for i in order])).unsqueeze(0).to(dev)
        with torch.no_grad():
            return m(pixel_values=x).last_hidden_state.mean(1).squeeze(0).float().cpu().numpy()

    ks = [k for k in KS if T // k >= 1]
    store = {f"K{k}_up": [] for k in ks}
    full = []
    t0 = time.time()
    for j, c in enumerate(clips):
        full.append(emb(c["paths"], list(range(T))))
        for k in ks:
            store[f"K{k}_up"].append(np.stack([emb(c["paths"], o)
                                               for o in _window_orders(k, upsample_to=T)]))
        if j % 25 == 0:
            print(f"[g0r/videomae] {j + 1}/{len(clips)} clips  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, full=np.stack(full), **{k: np.stack(v) for k, v in store.items()})
    return cache



def image_readout(frames, k):
    """Article Eq. (1), image case: frames [N,T,D] -> [N,2D], concatenating the clip mean and
    last-minus-first window means. k must divide T.
    """
    n, t, d = frames.shape
    w = frames.reshape(n, k, t // k, d).mean(2)                 # Window features have shape [N, K, D].
    return np.concatenate([frames.mean(1), w[:, -1] - w[:, 0]], 1)


def video_readout(full, windows):
    """Article Eq. (1), video case: concatenate full-clip embeddings [N,D] and last-minus-first window
    embeddings [N,K,D] to return [N,2D]. Window extraction and frame repetition happen upstream.
    """
    return np.concatenate([full, windows[:, -1] - windows[:, 0]], 1)



def loso_auc(X, y, sub, mask, rep):
    """Legacy aggregation-sweep statistic: mean of subject-specific test AUCs, excluding subjects
    without both classes. This is not the article's pooled out-of-fold AUC. For pooled scoring, use
    inference.loso_permutation_tests._oof_scores, average logits across seeded fits, and evaluate
    the combined eligible clips.
    """
    import torch, torch.nn as nn
    torch.manual_seed(1000 + rep)
    subs = sorted(set(sub.tolist()))
    out = []
    for st in subs:
        tr = (sub != st) & mask; te = (sub == st) & mask
        if te.sum() < 1 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        reg = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-3)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        for _ in range(300):
            loss = nn.functional.binary_cross_entropy_with_logits(reg(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pr = reg(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
        v = _auc(pr, y[te])
        if not np.isnan(v):
            out.append(v)
    return float(np.mean(out))


def cell(X, y, sub, mask, name, log):
    """Summarize the legacy mean-subject AUC across seeded fits."""
    vals = [loso_auc(X, y, sub, mask, r) for r in range(N_REP)]
    m, s = float(np.mean(vals)), float(np.std(vals))
    log(f"  {name:34s} AUC {m:.3f} +- {s:.3f}  (over {N_REP} seeded repeats)")
    return {"mean": m, "std": s, "reps": vals}



def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/run.log", "w")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    clips = build_clips()
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    audit_loso(lab["subject"])
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int)
    mask = lab["direction"] != -1
    assert len(clips) == len(sub), f"clip set drifted: {len(clips)} vs {len(sub)}"
    log(f"GATE G0-R readout-fairness sweep | {len(clips)} clips, {mask.sum()} direction-labelled, "
        f"{len(set(sub.tolist()))} subjects | device {_device()}")
    log(f"GO = (a) no video model exceeds best image at densest K "
        f"AND (b) image advantage survives reciprocal coarsening.")

    img_cache = extract_image_frames(clips)
    vj_cache = extract_vjepa(clips)
    vm_cache = extract_videomae(clips)
    IMG = np.load(img_cache); VJ = np.load(vj_cache); VM = np.load(vm_cache)

    res = {"provenance": run_provenance(seeds=[1000 + r for r in range(N_REP)]),
           "K_sweep": {}, "n_rep": N_REP, "n_clips": int(len(clips)),
           "n_direction": int(mask.sum())}

    log("\n--- image encoders, K windows (K=16 is the submitted readout, K=2 the reciprocal control)")
    for e in IMG_ENCODERS:
        for k in KS:
            res["K_sweep"][f"{e}_K{k}"] = cell(image_readout(IMG[e], k), y, sub, mask,
                                               f"{e} K={k}", log)

    log("\n--- V-JEPA-2, K windows (K=2 native is the submitted readout)")
    for key in sorted([k for k in VJ.files if k != "full"]):
        res["K_sweep"][f"vjepa_{key}"] = cell(video_readout(VJ["full"], VJ[key]), y, sub, mask,
                                              f"vjepa {key}", log)

    log("\n--- VideoMAE, K windows (K=2 is the submitted readout)")
    for key in sorted([k for k in VM.files if k != "full"]):
        res["K_sweep"][f"videomae_{key}"] = cell(video_readout(VM["full"], VM[key]), y, sub, mask,
                                                f"videomae {key}", log)

    # Check consistency against reference values.
    log("\n--- consistency with the submitted manuscript (expect agreement to ~0.02 AUC)")
    for name, key, published in (("CLIP K=16", "clip_K16", 0.911), ("DINOv2 K=16", "dinov2_K16", 0.908),
                                 ("SigLIP K=16", "siglip_K16", None),
                                 ("V-JEPA-2 K=2", "vjepa_K2_native", 0.814),
                                 ("VideoMAE K=2", "videomae_K2_up", 0.863)):
        if key in res["K_sweep"] and published is not None:
            got = res["K_sweep"][key]["mean"]
            flag = "ok" if abs(got - published) <= 0.03 else "DRIFT"
            log(f"  {name:16s} published {published:.3f}  reproduced {got:.3f}  [{flag}]")
            res.setdefault("consistency", {})[key] = {"published": published, "reproduced": got,
                                                      "status": flag}

    best_img_dense = max((res["K_sweep"][f"{e}_K16"]["mean"], e) for e in IMG_ENCODERS)
    vid_dense = []
    for fam, Z in (("vjepa", VJ), ("videomae", VM)):
        ks = [int(k.split("_")[0][1:]) for k in Z.files if k != "full"]
        kmax = max(ks)
        for key in [f"{fam}_K{kmax}_native", f"{fam}_K{kmax}_up"]:
            if key in res["K_sweep"]:
                vid_dense.append((res["K_sweep"][key]["mean"], key))
    best_vid_dense = max(vid_dense)
    crit_a = best_vid_dense[0] > best_img_dense[0]

    coarse = []
    for k in KS:
        vk_img = max(res["K_sweep"][f"{e}_K{k}"]["mean"] for e in IMG_ENCODERS)
        vk_vid = [res["K_sweep"][key]["mean"] for key in res["K_sweep"]
                  if key.startswith(("vjepa_", "videomae_")) and f"_K{k}_" in key]
        if vk_vid:
            coarse.append((k, vk_img, max(vk_vid), vk_img > max(vk_vid)))
    crit_b = all(c[3] for c in coarse)

    log("\n=== PRE-DECLARED CRITERIA ===")
    log(f"  best image at densest K=16 : {best_img_dense[1]} {best_img_dense[0]:.3f}")
    log(f"  best video at densest K    : {best_vid_dense[1]} {best_vid_dense[0]:.3f}")
    log(f"  (a) a video model exceeds the best image encoder at densest K : {crit_a}")
    for k, i, v, ok in coarse:
        log(f"      matched K={k:2d}: image {i:.3f} vs video {v:.3f} -> image ahead: {ok}")
    log(f"  (b) image advantage survives reciprocal coarsening at every matched K : {crit_b}")
    verdict = "GO (headline holds)" if (not crit_a and crit_b) else "NO-GO (headline is readout-dependent)"
    log(f"  VERDICT: {verdict}")

    res["criteria"] = {"a_video_exceeds_image_at_densest": bool(crit_a),
                       "b_image_survives_coarsening": bool(crit_b),
                       "best_image_dense": {"encoder": best_img_dense[1], "auc": best_img_dense[0]},
                       "best_video_dense": {"readout": best_vid_dense[1], "auc": best_vid_dense[0]},
                       "matched_K": [{"K": k, "image": i, "video": v, "image_ahead": ok}
                                     for k, i, v, ok in coarse],
                       "verdict": verdict}
    json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
    log(f"\nwrote {OUT}/metrics.json")
    logf.close()


if __name__ == "__main__":
    main()
