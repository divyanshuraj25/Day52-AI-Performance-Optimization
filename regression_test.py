"""regression_test_runner(): runs the full 30-question suite with the production (FIXED) config,
writes a timestamped results file, prints PASS or FAIL with the specific dimension and score that dropped.

    python regression_test.py                # offline
    python regression_test.py --mode openai
Exit code 0 = PASS, 1 = FAIL (CI friendly).
"""
import argparse, json, os, sys
from dataclasses import replace
from datetime import datetime
from rag_pipeline import FIXED
from llm_judge import DIMS
from run_eval import load_dataset, run_suite, tier_breakdown, dim_means, HERE

MIN_THRESHOLDS = {"correctness": 0.70, "groundedness": 0.90, "completeness": 0.70, "refusal_appropriateness": 0.90}
MIN_TIER = {"easy": 0.85, "medium": 0.75, "hard": 0.60}

def regression_test_runner(mode="offline", thresholds=None, tier_thresholds=None, cfg=None):
    thresholds = thresholds or MIN_THRESHOLDS; tier_thresholds = tier_thresholds or MIN_TIER
    cfg = replace(cfg or FIXED, mode=mode)
    rows = run_suite(load_dataset(), cfg)
    dims, tiers = dim_means(rows), tier_breakdown(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"regression_{mode}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "mode": mode, "config": cfg.__dict__, "dimensions": dims, "tiers": tiers, "rows": rows}, f, indent=2)
    failures = [f"{d} dropped to {dims[d]:.3f} (min {thresholds[d]:.2f})" for d in DIMS if dims[d] < thresholds[d]]
    failures += [f"tier '{t}' dropped to {tiers[t]:.3f} (min {tier_thresholds[t]:.2f})" for t in tier_thresholds if tiers[t] < tier_thresholds[t]]
    print(f"Dimensions: {dims}\nTiers: {tiers}\nResults written to {path}")
    if failures:
        print("FAIL: " + "; ".join(failures)); return False
    print("PASS: all dimensions meet minimum thresholds"); return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="offline", choices=["offline", "openai"])
    sys.exit(0 if regression_test_runner(ap.parse_args().mode) else 1)
