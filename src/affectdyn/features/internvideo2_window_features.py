"""InternVideo2 feature extraction at every granularity."""
from __future__ import annotations
import os, sys, time, types
import numpy as np

from affectdyn.data.unbc_clip_pipeline import build_clips, T, _prep224
from affectdyn.features.temporal_aggregation import _device, KS
from affectdyn.features.video_window_features import _orders, _full_order, OUT

IV2_REPO = "_local/InternVideo/InternVideo2/multi_modality"
IV2_COMMIT = "3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3"
CKPT_REPO = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
CKPT_FILE = "InternVideo2-stage2_1b-224p-f4.pt"
NFRAMES = 4                      # The checkpoint expects four frames.


def _stub_flash_attn():
    """Satisfy internvideo2.py's module-scope flash-attn import without building flash-attn."""
    def _refuse(*a, **k):
        raise RuntimeError("flash-attn stub called: the model was built with use_flash_attn=False, "
                           "so this path should be unreachable. Do not trust these features.")
    for name, attrs in {
        "flash_attn": {},
        "flash_attn.flash_attn_interface": {"flash_attn_varlen_qkvpacked_func": _refuse,
                                            "flash_attn_qkvpacked_func": _refuse},
        "flash_attn.bert_padding": {"unpad_input": _refuse, "pad_input": _refuse,
                                    "index_first_axis": _refuse},
        "flash_attn.modules": {},
        "flash_attn.modules.mlp": {"FusedMLP": _refuse},
        "flash_attn.ops": {},
        "flash_attn.ops.rms_norm": {"DropoutAddRMSNorm": _refuse, "rms_norm": _refuse},
    }.items():
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m


def _checkpoint_path():
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")     # Use standard checkpoint downloads.
    return hf_hub_download(CKPT_REPO, CKPT_FILE)


def _load():
    """Build InternVideo2-1B and load its vision-encoder weights, refusing a partial load."""
    import torch
    _stub_flash_attn()
    repo = os.path.abspath(IV2_REPO)
    if not os.path.isdir(repo):
        raise FileNotFoundError(
            f"{repo} missing. Sparse-clone it:\n"
            f"  git clone --filter=blob:none --sparse "
            f"https://github.com/OpenGVLab/InternVideo.git _local/InternVideo\n"
            f"  cd _local/InternVideo && git sparse-checkout set InternVideo2/multi_modality && git checkout {IV2_COMMIT}")
    # Import the vision module directly to avoid unrelated multimodal dependencies.
    import importlib.util
    base = os.path.join(repo, "models", "backbones", "internvideo2")
    pkg = types.ModuleType("iv2pkg")
    pkg.__path__ = [base]
    sys.modules["iv2pkg"] = pkg
    spec = importlib.util.spec_from_file_location("iv2pkg.internvideo2",
                                                  os.path.join(base, "internvideo2.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["iv2pkg.internvideo2"] = mod
    spec.loader.exec_module(mod)
    PretrainInternVideo2 = mod.PretrainInternVideo2

    model = PretrainInternVideo2(
        in_chans=3, img_size=224, patch_size=14,
        embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11,
        clip_embed_dim=768, attn_pool_num_heads=16, qkv_bias=False,
        drop_path_rate=0.0,                      # Disable stochastic depth for inference.
        init_values=0.00001, qk_normalization=True,
        use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        fused_mlp_heuristic=1, layerscale_no_force_fp32=False,
        num_frames=NFRAMES, tubelet_size=1,
        sep_pos_embed=False, sep_image_video_pos_embed=True,
        use_checkpoint=False, checkpoint_num=0,
        clip_teacher_embed_dim=3200, clip_teacher_final_dim=768,
        clip_norm_type="l2", clip_return_layer=6, clip_student_return_interval=1,
    )

    sd = torch.load(_checkpoint_path(), map_location="cpu", weights_only=False)
    for key in ("module", "model", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    vis = {}
    for k, v in sd.items():
        if k.startswith("vision_encoder."):
            vis[k[len("vision_encoder."):]] = v
    if not vis:                                   # Support checkpoints with unprefixed vision parameters.
        vis = sd
    info = model.load_state_dict(vis, strict=False)
    # Exclude unused training heads from missing-parameter checks.
    essential = [k for k in info.missing_keys
                 if not k.startswith(("clip_decoder", "final_clip_decoder", "clip_pos_embed",
                                      "clip_img_pos_embed", "mask_token"))]
    if essential:
        raise RuntimeError(
            f"InternVideo2 loaded with {len(essential)} missing encoder parameters, "
            f"e.g. {essential[:5]}. Refusing to extract features from a partly random encoder.")
    print(f"[w2/internvideo2] loaded {CKPT_FILE} | commit {IV2_COMMIT} | "
          f"{len(vis)} tensors, {len(info.missing_keys)} missing "
          f"({len(essential)} essential), {len(info.unexpected_keys)} unexpected", flush=True)
    return model.eval().to(_device())


def extract(clips, cache=f"{OUT}/internvideo2_windows.npz"):
    import torch
    if os.path.exists(cache):
        print(f"[w2/internvideo2] cache hit {cache}")
        return cache
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    m = _load()
    dev = _device()

    def emb(paths, order):
        x = torch.from_numpy(np.stack([_prep224(paths[i]) for i in order]))    # Input frames have shape [F, C, H, W].
        x = x.permute(1, 0, 2, 3).unsqueeze(0).to(dev)                         # The model input has shape [1, C, F, H, W].
        with torch.no_grad():
            out = m(x, mask=None, use_image=False)
        x_pool_vis = out[1] if isinstance(out, (tuple, list)) else out
        return x_pool_vis.squeeze(0).float().cpu().numpy()

    store = {f"K{k}_up": [] for k in KS}
    full = []
    t0 = time.time()
    for j, c in enumerate(clips):
        full.append(emb(c["paths"], _full_order(NFRAMES)))
        for k in KS:
            store[f"K{k}_up"].append(np.stack([emb(c["paths"], o) for o in _orders(k, NFRAMES)]))
        if j % 25 == 0:
            print(f"[w2/internvideo2] {j + 1}/{len(clips)} clips  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, full=np.stack(full), **{k: np.stack(v) for k, v in store.items()})
    print(f"[w2/internvideo2] wrote {cache}")
    return cache


if __name__ == "__main__":
    clips = build_clips()
    print(f"[w2/internvideo2] {len(clips)} clips | device {_device()}", flush=True)
    extract(clips)
