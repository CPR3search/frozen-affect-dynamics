"""Final summary of the pipeline simulation, including the generator verification.

Usage: python -m affectdyn.simulations.pipeline_simulation_summary   (after the --run stage has finished)
"""
from __future__ import annotations
import json, os, sys

from affectdyn.simulations.pipeline_simulation_resampled import OUT, LIMIT, KS, ALPHA, wilson1, summarize_regime   # noqa: E402

R = 1500


def main():
    ver = json.load(open(f"{OUT}/calibration_verify.json"))
    base = json.load(open(f"{OUT}/calibration.json"))
    if base["R"] != R:
        raise SystemExit(f"calibration.json has R={base['R']}, expected {R}")
    recs = [json.loads(l) for l in open(f"{OUT}/calibration_studies.jsonl")]
    verified = bool(ver["verified"])
    ev = {}
    for sc in ("A", "B"):
        rr = [r for r in recs if r["scenario"] == sc and "fits" in r and r["index"] < R]
        for fit in ("fixed", "converged"):
            n = len(rr); k = sum(r["fits"][fit]["event"] for r in rr)
            lo, hi = wilson1(k, n, "lower"), wilson1(k, n, "upper")
            out = "MET" if hi <= LIMIT else "EXCEEDED" if lo > LIMIT else "INCONCLUSIVE"
            old = base["events"][f"{sc}_{fit}"]
            if (n, k) != (old["n"], old["k"]):
                raise SystemExit(f"{sc}_{fit}: recount {k}/{n} differs from calibration.json {old['k']}/{old['n']}")
            row = {"n": n, "k": k, "rate": k / n, "wilson1_lower": lo, "wilson1_upper": hi, "criterion_outcome": out,
                   "per_K_rate_at_0.05": {f"K{kk}": sum(r["fits"][fit][f"K{kk}"]["p"] < ALPHA for r in rr) / n
                                          for kk in KS}}
            row["outcome"] = out if (verified or sc == "A") else "DESCRIPTIVE (off the operating regime)"
            ev[f"{sc}_{fit}"] = row
    deciding = [v["outcome"] for kk, v in ev.items() if verified or kk.startswith("A_")]
    if base["failures"]["gate_fail"]:
        deciding.append("INCONCLUSIVE")
    if verified:
        level = ("met the scoped simulation criterion" if all(o == "MET" for o in deciding)
                 else "not met" if any(o == "EXCEEDED" for o in deciding) else "INCONCLUSIVE")
    else:
        level = ("not met" if any(o == "EXCEEDED" for o in deciding)
                 else "INCONCLUSIVE (generator off the operating regime)")
    res = {"source": ["calibration.json", "calibration_studies.jsonl", "calibration_verify.json"],
           "verification": {"verified": verified, "distance": ver["distance"], "agg": ver["agg"]},
           "arm_auc_A": summarize_regime([r for r in recs if r["scenario"] == "A" and r["index"] < R]),
           "arm_auc_B": summarize_regime([r for r in recs if r["scenario"] == "B" and r["index"] < R]),
           "events": ev, "failures": base["failures"], "claim_level_unadjusted": base["claim_level"],
           "claim_level": level}
    json.dump(res, open(f"{OUT}/calibration_final.json", "w"), indent=2, default=float)
    for kk, v in ev.items():
        print(f"{kk}: {v['k']}/{v['n']} = {v['rate']:.4f} [{v['wilson1_lower']:.4f}, {v['wilson1_upper']:.4f}] "
              f"-> {v['outcome']}")
    print(f"CLAIM LEVEL: {level} (unadjusted: {base['claim_level']})")


if __name__ == "__main__":
    main()
