"""Aggregate per-experiment summary JSONs under results/alpha_540/ into a
single 7-way comparison table. All 6 baseline experiments are pre-evaluated;
the Path B one is evaluated by tools/eval_7way_pathB.sh.

Usage:
    python tools/aggregate_7way_summary.py \
        --summary-root results/alpha_540 \
        --baseline E0_new_code \
        --output results/alpha_540/summary_7way_pathB.json
"""
import argparse
import json
import os
import sys


# Canonical order for the 7 experiments. The filename mapping handles legacy
# single-letter naming (summary_E4rev.json etc.) as well as the new one
# (summary_pathB_p1_iter2000.json).
EXPERIMENT_ORDER = [
    ("E0_new_code",            "summary_E0new.json"),
    ("E3_cosine",              "summary_E3.json"),
    ("E4_reverse",             "summary_E4rev.json"),
    ("E4_reverse_clamped",     "summary_E4revclamped.json"),
    ("E4_sigmoid_mid",         "summary_E4mid.json"),
    ("E4_sigmoid_mid_clamped", "summary_E4clamped.json"),
    ("pathB_p1_iter2000",      "summary_pathB_p1_iter2000.json"),
]

METRICS = ["FVD", "EPE", "SSIM", "PSNR", "CLIP", "CTC"]
HIGHER_IS_BETTER = {"FVD": False, "EPE": False, "SSIM": True,
                     "PSNR": True, "CLIP": True, "CTC": True}


def load_summary(path, expected_key):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # summary files are {experiment_name: {metric: value}}
    if expected_key in data:
        return data[expected_key]
    # fallback: first (and only) entry
    keys = list(data.keys())
    if len(keys) == 1:
        return data[keys[0]]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-root", default="results/alpha_540")
    ap.add_argument("--baseline", default="E0_new_code")
    ap.add_argument("--output", default="results/alpha_540/summary_7way_pathB.json")
    args = ap.parse_args()

    root = args.summary_root
    results = {}
    for name, fname in EXPERIMENT_ORDER:
        path = os.path.join(root, fname)
        summary = load_summary(path, expected_key=name)
        if summary is None:
            print(f"  [MISSING] {name} — {path} not found or malformed. Skipping.")
            continue
        results[name] = summary

    if not results:
        sys.exit("No summaries could be loaded. Nothing to aggregate.")

    # --- build markdown table ---
    header = "| Experiment | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    lines = [header, sep]
    for name, _ in EXPERIMENT_ORDER:
        if name not in results:
            continue
        vals = [f"{results[name][m]:.4f}" for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    print("\n=== 7-way comparison (absolute) ===\n")
    print("\n".join(lines))

    # --- Δ vs baseline ---
    if args.baseline in results:
        base = results[args.baseline]
        print(f"\n=== Δ vs {args.baseline} ===\n")
        dlines = [header, sep]
        for name, _ in EXPERIMENT_ORDER:
            if name not in results or name == args.baseline:
                continue
            parts = []
            for m in METRICS:
                d = results[name][m] - base[m]
                better = (d > 0) if HIGHER_IS_BETTER[m] else (d < 0)
                mark = "✓" if better else "✗"
                sign = "+" if d >= 0 else ""
                parts.append(f"{sign}{d:.4f} {mark}")
            dlines.append(f"| {name} | " + " | ".join(parts) + " |")
        print("\n".join(dlines))

    # --- ranking per metric (who wins each) ---
    print(f"\n=== best experiment per metric ===\n")
    for m in METRICS:
        if HIGHER_IS_BETTER[m]:
            best_name = max(results, key=lambda n: results[n][m])
        else:
            best_name = min(results, key=lambda n: results[n][m])
        print(f"  {m:6s} (lower is better={'no' if HIGHER_IS_BETTER[m] else 'yes'}) → "
              f"{best_name:25s} = {results[best_name][m]:.4f}")

    # --- Path B specific narrative ---
    if "pathB_p1_iter2000" in results:
        pathB = results["pathB_p1_iter2000"]
        print(f"\n=== Path B narrative ===")
        print(f"  Path B FVD: {pathB['FVD']:.2f}")
        if args.baseline in results:
            dfvd = pathB["FVD"] - results[args.baseline]["FVD"]
            pct = 100.0 * dfvd / results[args.baseline]["FVD"]
            print(f"  vs {args.baseline}: Δ={dfvd:+.2f} ({pct:+.1f}%)")
        if "E4_reverse" in results:
            de = pathB["FVD"] - results["E4_reverse"]["FVD"]
            print(f"  vs E4_reverse (Path A winner): Δ={de:+.2f} — "
                  f"{'learned > handcrafted' if de < 0 else 'handcrafted ≥ learned'}")
        # Decision hint
        fvd = pathB["FVD"]
        if fvd < 400:
            verdict = "Strong → Direction ② probe next"
        elif fvd < 450:
            verdict = "Mild → probe + P2 TimeNoise"
        elif fvd < 500:
            verdict = "Match/Degrade → diagnose gate_net learned schedule"
        else:
            verdict = "Null → Path B training issue, diagnose before ②"
        print(f"  verdict: {verdict}")

    # --- dump aggregated JSON ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"experiments": results, "baseline": args.baseline,
                   "metrics": METRICS}, f, indent=2)
    print(f"\nSaved aggregated results → {args.output}")


if __name__ == "__main__":
    main()
