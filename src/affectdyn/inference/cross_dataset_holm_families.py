"""Correction families for the cross-dataset differences.

Usage:  python3 -m affectdyn.inference.cross_dataset_holm_families            # writes runs/peak_cluster_inference/family_harmonization.json
        python3 -m affectdyn.inference.cross_dataset_holm_families --print    # also prints the report
"""
from __future__ import annotations
import argparse, datetime, json, os, subprocess, sys

OUT = "runs/peak_cluster_inference/family_harmonization.json"
IMAGE_ENCODERS = ("clip", "dinov2", "siglip")


def _git(*args, default="unavailable"):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10).stdout.strip() or default
    except Exception:                                                  # noqa: BLE001
        return default


def provenance():
    """Stdlib-only counterpart of src/provenance.run_provenance; no numpy or torch here."""
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", default="")),
        "python": sys.version.split()[0],
        "inputs": ["runs/pain_encoder_maps/shape.json", "runs/affwild_clips_features/probe.json", "runs/affwild_clips_features/video_probe.json"],
        "note": "Re-corrects logged p values only; no probe is refitted and no feature is read.",
    }


def holm(pvals):
    """Local Holm adjustment; the shared implementation is
    inference.aggregation_permutation_tests.holm. Family membership and size are specified by the
    caller.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def significant(labels, aucs, adj, alpha=0.05):
    return [{"test": labels[i], "auc": round(aucs[i], 4), "p_holm": round(adj[i], 6)}
            for i in range(len(labels)) if adj[i] < alpha]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()

    unbc = json.load(open("runs/pain_encoder_maps/shape.json"))["rows"]
    aw_img = json.load(open("runs/affwild_clips_features/probe.json"))["shape"]["rows"]
    aw_vid = json.load(open("runs/affwild_clips_features/video_probe.json"))["shape"]

    u_lab = [r["cell"] for r in unbc]
    u_auc = [r["auc"] for r in unbc]
    u_p = [r["p"] for r in unbc]
    a_lab = ([f"{r['encoder']}_K{r['K']}" for r in aw_img] + [f"{r['encoder']}_K16" for r in aw_vid])
    a_auc = [r["auc"] for r in aw_img] + [r["auc"] for r in aw_vid]
    a_p = [r["p"] for r in aw_img] + [r["p"] for r in aw_vid]

    checks = {}
    for name, rows in (("unbc_36", unbc), ("aw2_image_12", aw_img), ("aw2_video_6", aw_vid)):
        mine = holm([r["p"] for r in rows])
        checks[name] = max(abs(a - r["p_holm"]) for a, r in zip(mine, rows))

    out = {"provenance": provenance(),
           "holm_reproduces_logged_p_holm": {k: v for k, v in checks.items()},
           "alpha": 0.05, "scenarios": {}}

    pub_u = holm(u_p)
    pub_ai, pub_av = holm([r["p"] for r in aw_img]), holm([r["p"] for r in aw_vid])
    pub_a_lab = a_lab
    pub_a_adj = pub_ai + pub_av
    out["scenarios"]["as_published"] = {
        "convention": "UNBC pooled (36); Aff-Wild2 split by modality (12 image, 6 video)",
        "unbc_significant": significant(u_lab, u_auc, pub_u),
        "aff_wild2_significant": significant(pub_a_lab, a_auc, pub_a_adj),
    }

    # Pool modalities into one family per corpus.
    harm_a = holm(a_p)
    out["scenarios"]["both_pooled"] = {
        "convention": "one family per corpus per property (UNBC 36, Aff-Wild2 18)",
        "unbc_significant": significant(u_lab, u_auc, pub_u),
        "aff_wild2_significant": significant(a_lab, a_auc, harm_a),
    }

    # Split each corpus's families by modality.
    ui = [i for i, l in enumerate(u_lab) if l.rsplit("_K", 1)[0] in IMAGE_ENCODERS]
    uv = [i for i, l in enumerate(u_lab) if l.rsplit("_K", 1)[0] not in IMAGE_ENCODERS]
    adj_ui, adj_uv = holm([u_p[i] for i in ui]), holm([u_p[i] for i in uv])
    unbc_split = (significant([u_lab[i] for i in ui], [u_auc[i] for i in ui], adj_ui)
                  + significant([u_lab[i] for i in uv], [u_auc[i] for i in uv], adj_uv))
    out["scenarios"]["both_split_by_modality"] = {
        "convention": "split by modality on BOTH corpora (UNBC 12/24, Aff-Wild2 12/6)",
        "unbc_significant": unbc_split,
        "aff_wild2_significant": significant(pub_a_lab, a_auc, pub_a_adj),
    }

    out["finding"] = {
        "aff_wild2_positives_are_robust": (
            "Pooling the Aff-Wild2 tests into one family of 18 leaves every positive standing; the "
            "raw p values are small enough that the family size does not decide them."),
        "unbc_non_detection_is_convention_dependent": (
            "Under the modality-split convention Aff-Wild2 actually received, CLIP reaches "
            "significance on UNBC pain at K=4 and K=8, so the claim that trajectory shape is "
            "undecodable from every encoder on pain does not survive a matched convention."),
        "consequence": (
            "The dissociation holds under a pooled-per-corpus convention and weakens under a "
            "modality-split one. Whichever is chosen must be applied to BOTH corpora and stated."),
        "reporting_gap": (
            "CLIP on Aff-Wild2 is significant at AUC 0.440, below chance rather than above it. It is "
            "excluded from the manuscript's count of three, defensibly, but the exclusion is not "
            "stated and a significant below-chance result carries its own reporting obligation."),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")

    if args.show:
        for name, sc in out["scenarios"].items():
            print(f"\n--- {name}: {sc['convention']}")
            print(f"    UNBC significant      : {len(sc['unbc_significant'])} "
                  f"{[s['test'] for s in sc['unbc_significant']]}")
            print(f"    Aff-Wild2 significant : {len(sc['aff_wild2_significant'])} "
                  f"{[s['test'] for s in sc['aff_wild2_significant']]}")


if __name__ == "__main__":
    main()
