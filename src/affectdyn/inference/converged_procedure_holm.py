"""Holm correction for every procedure under the converged fit."""
import json, os, sys

from affectdyn.inference.aggregation_permutation_tests import holm                                       # noqa: E402
from affectdyn.provenance import run_provenance                                # noqa: E402

def main():
    d = json.load(open("runs/converged_probe_fits/convergence.json"))
    out = {"provenance": run_provenance(device="cpu"), "source": "runs/converged_probe_fits/convergence.json", "families": {}}
    ok = True
    for label, key in (("published", "F2F3_published"), ("converged", "F2F3_converged")):
        cells = d["M3"][key]
        for fam in ("unbc", "aw2"):
            fc = [c for c in cells if c["family"] == fam]
            res = {c["cell"]: {"auc": c["auc"]} for c in fc}
            for proc in ("p_perm_clip", "p_perm_subject", "p_boot_clip", "p_boot_subject"):
                for c, a in zip(fc, holm([c[proc] for c in fc])):
                    res[c["cell"]][proc] = c[proc]; res[c["cell"]]["holm_" + proc] = float(a)
            for c in fc:
                ok &= res[c["cell"]]["holm_p_boot_subject"] == c["holm_boot_subject"]
            out["families"][f"{label}:{fam}"] = res
    out["gate_subject_bootstrap_holm_reproduces_w46"] = bool(ok)
    json.dump(out, open("runs/converged_probe_fits/holm_procedures.json", "w"), indent=1)
    x = out["families"]["converged:unbc"]["xclip_K8"]
    print("gate", ok, "| xclip_K8 converged", {k: round(v, 5) for k, v in x.items()})


if __name__ == "__main__":
    main()
