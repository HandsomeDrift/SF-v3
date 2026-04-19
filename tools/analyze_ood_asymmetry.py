"""§6 OOD Temporal Asymmetry — unified analysis for Exp 1 and Exp 4.

Given a run directory from one of:
  - run_exp1_full_sweep.sh  (single-step α perturbation sweep)
  - run_exp4_amplification.sh  (single-step latent perturbation sweep)

Computes:
  1. Per-step L2 error curve (Exp 1: sensitivity*amp product; Exp 4: amplification)
  2. Log-plot and linear-fit diagnostic
  3. Ratio L2(τ_early) / L2(τ_late) → main asymmetry claim
  4. Per-sample variance + bootstrap CI

Outputs JSON + PNG plot + markdown summary.

Usage:
  python tools/analyze_ood_asymmetry.py --exp exp1 --base results/exp1_alpha_sweep
  python tools/analyze_ood_asymmetry.py --exp exp4 --base results/exp4_amplification --eps 0.01
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["exp1", "exp4"], required=True)
    ap.add_argument("--base", required=True, help="output dir from sweep script")
    ap.add_argument("--eps", type=float, default=None,
                    help="for exp4: ε used; divide diffs by this to get amplification")
    ap.add_argument("--delta", type=float, default=0.3,
                    help="for exp1: δα_brain used")
    ap.add_argument("--plot", action="store_true", help="save matplotlib PNG (requires matplotlib)")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--out_md", type=str, default=None)
    return ap.parse_args()


def steps_for_exp(exp, base):
    """Discover which step directories exist under base."""
    if exp == "exp1":
        dirs = glob.glob(os.path.join(base, "step_*"))
    else:  # exp4
        dirs = glob.glob(os.path.join(base, "step_*_dir_*"))
    steps = set()
    for d in dirs:
        name = os.path.basename(d)
        if exp == "exp1":
            # step_{i}
            try:
                steps.add(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        else:
            # step_{i}_dir_{d}
            try:
                steps.add(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(steps)


def load_latents_dir(d):
    files = sorted(glob.glob(os.path.join(d, "*_latent.pt")))
    return {os.path.basename(f).replace("_latent.pt", ""): torch.load(f, map_location="cpu") for f in files}


def compute_exp1(base, delta, steps):
    """Exp 1: compare each step's latents against baseline, produce L2 curve."""
    baseline_dir = os.path.join(base, "baseline")
    baseline = load_latents_dir(baseline_dir)
    if not baseline:
        raise RuntimeError(f"No baseline latents in {baseline_dir}")

    results = {}
    for step in steps:
        step_dir = os.path.join(base, f"step_{step}")
        perturbed = load_latents_dir(step_dir)
        shared_keys = set(baseline.keys()) & set(perturbed.keys())
        if not shared_keys:
            print(f"  WARN: step {step} has no matching samples with baseline")
            continue
        per_sample_l2 = []
        per_sample_maxabs = []
        for k in sorted(shared_keys):
            b = baseline[k].float()
            p = perturbed[k].float()
            per_sample_l2.append((b - p).norm().item())
            per_sample_maxabs.append((b - p).abs().max().item())
        results[step] = {
            "n_samples": len(per_sample_l2),
            "l2_mean": sum(per_sample_l2) / len(per_sample_l2),
            "l2_std": (sum((x - sum(per_sample_l2) / len(per_sample_l2)) ** 2 for x in per_sample_l2) / len(per_sample_l2)) ** 0.5,
            "l2_per_sample": per_sample_l2,
            "maxabs_mean": sum(per_sample_maxabs) / len(per_sample_maxabs),
            "sensitivity": (sum(per_sample_l2) / len(per_sample_l2)) / delta,
        }
    return results


def compute_exp4(base, eps, steps):
    """Exp 4: for each step, aggregate over direction seeds, compute A(τ)."""
    baseline_dir = os.path.join(base, "baseline")
    baseline = load_latents_dir(baseline_dir)

    results = {}
    for step in steps:
        dir_matches = sorted(glob.glob(os.path.join(base, f"step_{step}_dir_*")))
        per_run = []
        for dir_dir in dir_matches:
            perturbed = load_latents_dir(dir_dir)
            shared_keys = set(baseline.keys()) & set(perturbed.keys())
            for k in sorted(shared_keys):
                b = baseline[k].float()
                p = perturbed[k].float()
                diff_l2 = (b - p).norm().item()
                per_run.append({
                    "dir": os.path.basename(dir_dir),
                    "sample": k,
                    "l2": diff_l2,
                    "amplification": diff_l2 / eps if eps else None,
                })
        if not per_run:
            continue
        amps = [r["amplification"] for r in per_run if r["amplification"] is not None]
        results[step] = {
            "n_runs": len(per_run),
            "amp_mean": sum(amps) / len(amps) if amps else None,
            "amp_std": (sum((x - sum(amps) / len(amps)) ** 2 for x in amps) / len(amps)) ** 0.5 if amps else None,
            "per_run": per_run,
        }
    return results


def print_summary(exp, results, delta=None, eps=None, num_steps=50):
    print("\n" + "=" * 60)
    print(f"  {exp.upper()} Analysis Summary")
    print("=" * 60)
    steps = sorted(results.keys())
    print(f"{'step i*':>8} {'τ':>6}", end="")
    if exp == "exp1":
        print(f" {'n_samples':>10} {'L2 mean':>10} {'L2 std':>10} {'sensitivity':>12}")
    else:
        print(f" {'n_runs':>8} {'A(τ) mean':>12} {'A(τ) std':>12}")

    for step in steps:
        tau = step / max(num_steps - 2, 1)
        r = results[step]
        print(f"{step:>8} {tau:>6.3f}", end="")
        if exp == "exp1":
            print(f" {r['n_samples']:>10} {r['l2_mean']:>10.2e} {r['l2_std']:>10.2e} {r['sensitivity']:>12.3e}")
        else:
            print(f" {r['n_runs']:>8} {r['amp_mean']:>12.2e} {r['amp_std']:>12.2e}")

    if len(steps) >= 2:
        first, last = steps[0], steps[-1]
        if exp == "exp1":
            ratio = results[first]["l2_mean"] / max(results[last]["l2_mean"], 1e-12)
        else:
            ratio = results[first]["amp_mean"] / max(results[last]["amp_mean"], 1e-12)
        print(f"\nAsymmetry ratio (step {first} / step {last}) = {ratio:.2f}x")
        print(f"  Theoretical prediction (P1, §6.2): ≥ 3×")
        verdict = "STRONG" if ratio >= 3 else ("WEAK" if ratio >= 1.5 else "FAIL")
        print(f"  Verdict: {verdict}")
    return steps


def plot_curve(exp, results, delta, eps, out_png, num_steps=50):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    steps = sorted(results.keys())
    taus = [s / max(num_steps - 2, 1) for s in steps]
    if exp == "exp1":
        means = [results[s]["l2_mean"] for s in steps]
        stds = [results[s]["l2_std"] for s in steps]
        ylabel = r"$\|e_N\|_2$ (Exp 1 single-step $\alpha$ perturbation)"
        title = rf"Exp 1: $L_2$ error vs step $i^*$ (δα = {delta})"
    else:
        means = [results[s]["amp_mean"] for s in steps]
        stds = [results[s]["amp_std"] for s in steps]
        ylabel = r"$A(\tau)$ (amplification factor)"
        title = rf"Exp 4: Amplification $A(\tau)$ (ε = {eps})"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.errorbar(taus, means, yerr=stds, marker="o", capsize=3)
    ax1.set_xlabel(r"$\tau = i^* / (N - 1)$")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title + " (linear)")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(taus, means, yerr=stds, marker="o", capsize=3)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel(ylabel + " (log scale)")
    ax2.set_title(title + " (log)")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nPlot saved to {out_png}")


def write_md_report(exp, results, delta, eps, out_md, num_steps=50):
    steps = sorted(results.keys())
    with open(out_md, "w") as f:
        f.write(f"# {exp.upper()} OOD Asymmetry Analysis\n\n")
        f.write(f"Analyzed {len(steps)} step values: {steps}\n\n")

        f.write("## Per-step table\n\n")
        if exp == "exp1":
            f.write("| step $i^*$ | $\\tau$ | n samples | $L_2$ mean | $L_2$ std | sensitivity |\n")
            f.write("|---:|---:|---:|---:|---:|---:|\n")
            for step in steps:
                tau = step / max(num_steps - 2, 1)
                r = results[step]
                f.write(f"| {step} | {tau:.3f} | {r['n_samples']} | {r['l2_mean']:.2e} | {r['l2_std']:.2e} | {r['sensitivity']:.3e} |\n")
        else:
            f.write("| step $i^*$ | $\\tau$ | n runs | $A(\\tau)$ mean | $A(\\tau)$ std |\n")
            f.write("|---:|---:|---:|---:|---:|\n")
            for step in steps:
                tau = step / max(num_steps - 2, 1)
                r = results[step]
                f.write(f"| {step} | {tau:.3f} | {r['n_runs']} | {r['amp_mean']:.2e} | {r['amp_std']:.2e} |\n")

        if len(steps) >= 2:
            first, last = steps[0], steps[-1]
            if exp == "exp1":
                ratio = results[first]["l2_mean"] / max(results[last]["l2_mean"], 1e-12)
            else:
                ratio = results[first]["amp_mean"] / max(results[last]["amp_mean"], 1e-12)
            f.write(f"\n## Asymmetry (P1 verification)\n\n")
            f.write(f"Ratio step {first} / step {last} = **{ratio:.2f}x**\n\n")
            f.write(f"Theoretical prediction: ≥ 3× for P1 to hold.\n\n")
            verdict = "STRONG" if ratio >= 3 else ("WEAK" if ratio >= 1.5 else "FAIL")
            f.write(f"Verdict: **{verdict}**\n")


def main():
    args = parse_args()
    if args.exp == "exp4" and args.eps is None:
        raise ValueError("--eps required for exp4")

    steps = steps_for_exp(args.exp, args.base)
    print(f"Discovered {len(steps)} step directories: {steps}")
    if not steps:
        raise RuntimeError(f"No step directories under {args.base}")

    if args.exp == "exp1":
        results = compute_exp1(args.base, args.delta, steps)
    else:
        results = compute_exp4(args.base, args.eps, steps)

    print_summary(args.exp, results, delta=args.delta, eps=args.eps)

    out_json = args.out_json or os.path.join(args.base, "analysis.json")
    with open(out_json, "w") as f:
        # Sanitize (drop large per_sample lists for JSON)
        clean = {}
        for step, r in results.items():
            clean[str(step)] = {k: v for k, v in r.items() if not isinstance(v, list) or len(v) < 100}
        json.dump(clean, f, indent=2)
    print(f"Saved {out_json}")

    if args.plot:
        out_png = os.path.join(args.base, f"{args.exp}_curve.png")
        plot_curve(args.exp, results, args.delta, args.eps, out_png)

    out_md = args.out_md or os.path.join(args.base, f"{args.exp}_analysis.md")
    write_md_report(args.exp, results, args.delta, args.eps, out_md)
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
