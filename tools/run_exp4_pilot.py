"""§6 Exp 4 — epsilon selection pilot.

Before the full amplification sweep, we need to choose a perturbation
magnitude ε that lies in the linear response region. Too small and
numerical/quantization noise dominates; too large and second-order
terms pollute the amplification estimate.

Protocol (per §E of THEORY_ood_asymmetry_v1.md):
  - Fix i* = 27 (middle step)
  - Sweep ε ∈ {1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0}
  - For each ε, run N_DIRS independent noise directions, compute
    ||x_N^perturbed - x_N^baseline|| / ε
  - Identify linear region: ε range where ratio is stable (not collapsing to 0
    or growing super-linearly)
  - Output recommended ε*

Usage:
    python tools/run_exp4_pilot.py --n_samples 1 --n_dirs 3

This script writes each sub-config, launches sample_brain_va.py via subprocess,
then aggregates results.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import glob

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=1, help="number of samples (use 1 for speed)")
    ap.add_argument("--n_dirs", type=int, default=3, help="random noise directions per epsilon")
    ap.add_argument("--i_star", type=int, default=27, help="step index to perturb at")
    ap.add_argument("--output_base", type=str, default="results/exp4_eps_pilot")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port_base", type=int, default=29950)
    ap.add_argument("--dataset_json", type=str,
                    default="/public/home/maoyaoxin/xxt/datasets/exp1_unit_1sample.json")
    ap.add_argument("--skip_baseline", action="store_true",
                    help="if baseline already run, skip re-running it")
    return ap.parse_args()


EPS_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0]
OVERRIDE_TEMPLATE = """\
model:
  sampler_config:
    params:
      perturb_spec:
        alpha_brain_step: null
        alpha_brain_delta: 0.0
        latent_step: {latent_step}
        latent_eps: {latent_eps}
        latent_seed: {latent_seed}
"""


def run_inference(output_dir, override_path, port, gpu, dataset_json):
    os.makedirs(output_dir, exist_ok=True)
    base_args = [
        "configs/sf_v1/cinebrain_sf_v1_model.yaml",
    ]
    if override_path is not None:
        base_args.append(override_path)
    base_args.append("configs/sf_v1/infer_stage3_v2.yaml")

    cmd = [
        "/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python",
        "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=1",
        f"--master_port={port}",
        "sample_brain_va.py",
        "--base", *base_args,
        "--seed", "42",
        "--jsonpath", dataset_json,
        "--output_dir", output_dir,
        "--save_latents",
    ]
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda-12.4"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = "/public/home/maoyaoxin/xxt/SF-v3"

    log_path = output_dir + ".log"
    print(f"[Run] port={port}, gpu={gpu}, override={override_path}")
    print(f"      log: {log_path}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    t_elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"      FAILED (rc={result.returncode}) after {t_elapsed:.0f}s")
        raise RuntimeError(f"Inference failed (rc={result.returncode}). See {log_path}")
    print(f"      OK in {t_elapsed:.0f}s")


def load_latent(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "*_latent.pt")))
    assert len(files) >= 1, f"No latents in {output_dir}"
    return [torch.load(f, map_location="cpu") for f in files]


def main():
    args = parse_args()
    os.makedirs(args.output_base, exist_ok=True)

    # 1. Baseline
    baseline_dir = os.path.join(args.output_base, "baseline")
    if not args.skip_baseline or not os.path.exists(os.path.join(baseline_dir, "done.flag")):
        run_inference(baseline_dir, None, args.port_base, args.gpu, args.dataset_json)
        open(os.path.join(baseline_dir, "done.flag"), "w").close()

    baseline_latents = load_latent(baseline_dir)
    print(f"\nLoaded {len(baseline_latents)} baseline latents")
    print(f"Shape: {baseline_latents[0].shape}, dtype: {baseline_latents[0].dtype}")

    # 2. Perturbed runs
    port = args.port_base
    results = {}  # eps -> list of (sample, dir, ratio, raw_diff)

    for eps in EPS_VALUES:
        results[eps] = []
        for d in range(args.n_dirs):
            port += 1
            seed_val = int(eps * 1e6) + d * 1000
            out_dir = os.path.join(args.output_base, f"eps_{eps:.0e}_dir_{d}")
            override_path = f"/tmp/exp4_pilot_override_{eps:.0e}_{d}.yaml"
            with open(override_path, "w") as f:
                f.write(OVERRIDE_TEMPLATE.format(
                    latent_step=args.i_star,
                    latent_eps=eps,
                    latent_seed=seed_val,
                ))

            if os.path.exists(os.path.join(out_dir, "done.flag")):
                print(f"Skip eps={eps}, dir={d}: already done")
            else:
                run_inference(out_dir, override_path, port, args.gpu, args.dataset_json)
                open(os.path.join(out_dir, "done.flag"), "w").close()

            pert_latents = load_latent(out_dir)
            for sample_idx, (b, p) in enumerate(zip(baseline_latents, pert_latents)):
                diff = (b.float() - p.float()).norm().item()
                ratio = diff / eps
                results[eps].append({
                    "sample": sample_idx,
                    "dir": d,
                    "diff": diff,
                    "ratio": ratio,
                })
                print(f"  eps={eps:.0e}, dir={d}, sample={sample_idx}: ||Δx|| = {diff:.2e}, ratio = {ratio:.2f}")

    # 3. Save and analyze
    with open(os.path.join(args.output_base, "pilot_results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print("\n=========================================")
    print("  Pilot summary — choose eps in linear region")
    print("=========================================")
    print(f"{'eps':>10} {'mean_ratio':>14} {'std_ratio':>12} {'status':>20}")
    mean_ratios = {}
    for eps in EPS_VALUES:
        ratios = [r["ratio"] for r in results[eps]]
        mean = sum(ratios) / len(ratios)
        std = (sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5
        mean_ratios[eps] = mean
        print(f"{eps:>10.0e} {mean:>14.2f} {std:>12.2f}", end="")
        # rough linearity check
        if 0 < mean < 1e6:
            print(f"{'  OK':>20}")
        else:
            print(f"{'  OUT OF RANGE':>20}")

    # Identify linear region: consecutive epsilons with similar mean_ratio
    print("\nLinear region analysis:")
    linear_eps = []
    for i, eps in enumerate(EPS_VALUES[:-1]):
        r_now, r_next = mean_ratios[eps], mean_ratios[EPS_VALUES[i + 1]]
        ratio_delta = abs(r_now - r_next) / max(r_now, r_next, 1e-9)
        if ratio_delta < 0.2:
            linear_eps.append(eps)
            linear_eps.append(EPS_VALUES[i + 1])
            print(f"  [{eps:.0e}, {EPS_VALUES[i+1]:.0e}]: Δratio = {ratio_delta * 100:.1f}% (LINEAR)")
        else:
            print(f"  [{eps:.0e}, {EPS_VALUES[i+1]:.0e}]: Δratio = {ratio_delta * 100:.1f}%")

    linear_eps = sorted(set(linear_eps))
    if linear_eps:
        eps_star = (linear_eps[0] * linear_eps[-1]) ** 0.5  # geometric mean
        print(f"\nRecommended ε* = {eps_star:.2e} (geometric mean of linear-region endpoints)")
        print(f"Linear region: [{linear_eps[0]:.0e}, {linear_eps[-1]:.0e}]")
    else:
        print("\nNo clear linear region found. Try wider sweep or more samples.")


if __name__ == "__main__":
    main()
