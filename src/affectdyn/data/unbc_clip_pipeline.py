"""UNBC clip construction, label rules and the base linear probe."""
from __future__ import annotations
import json, os, re, sys
import numpy as np

DATA = "data/unbc"
T = 16
MAXPER = 30


def build_clips():
    """Build capped 16-frame UNBC clips with an eight-frame stride. Exclude flat-neutral clips before
    applying the subject cap. direction is 1=onset, 0=offset, -1=ineligible, with inclusive +/-2
    PSPI margins. traj is 2=peaked, 0=rising non-peaked, 1=falling non-peaked, -1=ineligible. A peak
    takes precedence and requires the first maximum at frames 4-13 inclusive, at least 1 PSPI above
    both endpoints; otherwise a non-peaked label requires absolute endpoint change >=1. Binary peak-
    task analyses use traj==2 and exclude traj==-1.
    """
    recs = json.load(open(os.path.join(DATA, "manifest.json")))
    by = {}
    for r in recs:
        m = re.search(r"(\d+)$", r["frame"]); fn = int(m.group(1)) if m else 0
        path = os.path.join(DATA, "aligned_112", r["crop"].split("aligned_112/")[1])
        by.setdefault((r["subject"], r["seq"]), []).append((fn, path, float(r["pspi"])))
    clips = []; per = {}
    for (subj, seq), fr in sorted(by.items()):
        fr.sort()
        # Historical extraction boundary: the stop is exclusive at length-T, so a window starting exactly there is omitted, including the sole window of a T-frame sequence. Retained to reproduce the released clip selection.
        for i in range(0, len(fr) - T, T // 2):
            win = fr[i:i + T]
            pspi = np.array([w[2] for w in win], np.float32)
            if pspi.max() < 1 and pspi.std() < 0.5:
                continue   # Skip flat-neutral windows.
            if per.get(subj, 0) >= MAXPER:
                continue
            clips.append({"subject": subj, "paths": [w[1] for w in win], "pspi": pspi})
            per[subj] = per.get(subj, 0) + 1
    for c in clips:
        p = c["pspi"]; d = p[-1] - p[0]; a = int(np.argmax(p))
        c["direction"] = 1 if d >= 2 else (0 if d <= -2 else -1)   # Exclude windows labelled -1.
        c["slope"] = float(d)
        peaked = 3 <= a <= 12 and p[a] >= p[0] + 1 and p[a] >= p[-1] + 1
        c["traj"] = 2 if peaked else (0 if d >= 1 else (1 if d <= -1 else -1))   # Encode rises as 0, falls as 1, and peaks as 2.
    return clips


def _prep256(path):
    from PIL import Image
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
    a = np.asarray(Image.open(path).convert("RGB").resize((256, 256)), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def _prep224(path):
    from PIL import Image
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
    a = np.asarray(Image.open(path).convert("RGB").resize((224, 224)), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def extract(clips, out_dir):
    import torch, torchvision
    from transformers import VJEPA2Model
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "stage1_feats.npz")
    if os.path.exists(cache):
        return cache
    torch.set_num_threads(8)
    vj = VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32).eval()
    res = torchvision.models.resnet18(weights="IMAGENET1K_V1"); res.fc = torch.nn.Identity(); res.eval()

    def vjepa(paths, order):
        x = torch.from_numpy(np.stack([_prep256(paths[i]) for i in order])).unsqueeze(0)
        with torch.no_grad():
            return vj(pixel_values_videos=x).last_hidden_state.mean(1).squeeze(0).numpy()

    Vo, Vs, Vr, St, dirc, traj, slope, sub = [], [], [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for j, c in enumerate(clips):
        o = list(range(T)); sh = list(rng.permutation(T)); rv = list(range(T - 1, -1, -1))
        Vo.append(vjepa(c["paths"], o)); Vs.append(vjepa(c["paths"], sh))
        Vr.append(vjepa(c["paths"], rv) if c["direction"] != -1 else np.zeros(1024, np.float32))
        with torch.no_grad():
            fr = res(torch.from_numpy(np.stack([_prep224(p) for p in c["paths"]]))).numpy()
        St.append(fr.mean(0))   # Mean pooling produces an order-invariant representation.
        dirc.append(c["direction"]); traj.append(c["traj"]); slope.append(c["slope"]); sub.append(c["subject"])
        if j % 25 == 0:
            print(f"[s1] {j+1}/{len(clips)} clips", flush=True)
    np.savez_compressed(cache, vord=np.stack(Vo), vshuf=np.stack(Vs), vrev=np.stack(Vr),
                        static=np.stack(St), direction=np.array(dirc), traj=np.array(traj),
                        slope=np.array(slope), subject=np.array(sub))
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


def _fit_logits(X, y, classes=1):
    import torch, torch.nn as nn
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    clf = nn.Linear(X.shape[1], classes)
    opt = torch.optim.Adam(clf.parameters(), 1e-2, weight_decay=1e-3)
    yt = torch.tensor(y, dtype=torch.float32 if classes == 1 else torch.long)
    for _ in range(300):
        out = clf(xt)
        loss = (nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), yt) if classes == 1
                else nn.functional.cross_entropy(out, yt))
        opt.zero_grad(); loss.backward(); opt.step()
    return clf, mu, sd


def probe(cache):
    import torch
    d = np.load(cache, allow_pickle=False)
    subs = sorted(set(d["subject"].tolist())); SUB = d["subject"]
    feats = {"V-JEPA ordered": d["vord"], "V-JEPA shuffled": d["vshuf"], "static ResNet": d["static"]}

    def loso_auc(X, y, mask):
        a = []
        for st in subs:
            tr = (SUB != st) & mask; te = (SUB == st) & mask
            if te.sum() == 0 or len(set(y[tr])) < 2 or len(set(y[te])) < 1:
                continue
            clf, mu, sd = _fit_logits(X[tr], y[tr], 1)
            with torch.no_grad():
                sc = clf(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
            v = _auc(sc, y[te])
            if not np.isnan(v): a.append(v)
        return np.mean(a), np.std(a), len(a)

    def loso_macrof1(X, y, mask, C):
        f1s = []
        for st in subs:
            tr = (SUB != st) & mask; te = (SUB == st) & mask
            if te.sum() == 0 or len(set(y[tr])) < C:
                continue
            clf, mu, sd = _fit_logits(X[tr], y[tr], C)
            with torch.no_grad():
                pr = clf(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).argmax(1).numpy()
            yt = y[te]; fs = []
            for c in range(C):
                tp = ((pr == c) & (yt == c)).sum(); fp = ((pr == c) & (yt != c)).sum(); fn = ((pr != c) & (yt == c)).sum()
                p_ = tp / (tp + fp) if tp + fp else 0; r_ = tp / (tp + fn) if tp + fn else 0
                fs.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0)
            f1s.append(np.mean(fs))
        return np.mean(f1s), np.std(f1s), len(f1s)

    def loso_corr(X, y, mask):
        cs = []
        for st in subs:
            tr = (SUB != st) & mask; te = (SUB == st) & mask
            if te.sum() < 3:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            import torch.nn as nn
            xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.float32)
            reg = nn.Linear(X.shape[1], 1); opt = torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-3)
            for _ in range(300):
                loss = nn.functional.mse_loss(reg(xt).squeeze(-1), yt); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pr = reg(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
            yy = y[te]
            if np.std(pr) > 0 and np.std(yy) > 0:
                rp = np.argsort(np.argsort(pr)); ry = np.argsort(np.argsort(yy))
                cs.append(np.corrcoef(rp, ry)[0, 1])
        return np.mean(cs), np.std(cs), len(cs)

    print("\n=== P1 direction (onset vs offset), AUC ===")
    dm = d["direction"] != -1
    for nm, X in feats.items():
        m, s, n = loso_auc(X, (d["direction"] == 1).astype(int), dm)
        print(f"  {nm:16s} AUC {m:.3f} ± {s:.3f} (n={n})")
    # A direction probe trained on ordered clips should invert on reversed clips.
    a_rev = []
    for st in subs:
        tr = (SUB != st) & dm; te = (SUB == st) & dm
        y = (d["direction"] == 1).astype(int)
        if te.sum() == 0 or len(set(y[tr])) < 2 or len(set(y[te])) < 1:
            continue
        clf, mu, sd = _fit_logits(d["vord"][tr], y[tr], 1)
        with torch.no_grad():
            sc = clf(torch.tensor((d["vrev"][te] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
        v = _auc(sc, y[te])
        if not np.isnan(v): a_rev.append(v)
    print(f"  C2 reversal: ordered-probe on REVERSED clips AUC {np.mean(a_rev):.3f} (expect <0.5 if direction encoded)")

    print("=== P2 trajectory class (rise/fall/peak), macro-F1 ===")
    tm = d["traj"] != -1
    for nm in ("V-JEPA ordered", "V-JEPA shuffled", "static ResNet"):
        m, s, n = loso_macrof1(feats[nm], d["traj"], tm, 3)
        print(f"  {nm:16s} macroF1 {m:.3f} ± {s:.3f} (n={n})")
    print("=== P4 velocity (Delta-PSPI), Spearman ===")
    allm = np.ones(len(d["slope"]), bool)
    for nm in ("V-JEPA ordered", "V-JEPA shuffled", "static ResNet"):
        m, s, n = loso_corr(feats[nm], d["slope"], allm)
        print(f"  {nm:16s} rho {m:.3f} ± {s:.3f} (n={n})")


if __name__ == "__main__":
    out = "runs/unbc_clip_pipeline"
    clips = build_clips()
    nd = sum(1 for c in clips if c["direction"] != -1); nt = sum(1 for c in clips if c["traj"] != -1)
    print(f"[s1] {len(clips)} clips | direction-labeled {nd} | traj-labeled {nt} | "
          f"{len(set(c['subject'] for c in clips))} subjects", flush=True)
    cache = extract(clips, out)
    probe(cache)
