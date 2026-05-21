"""Probe a Path B checkpoint's α(sample, τ) curve.

Purpose: during P1 training we want a fast (~3 min) sanity check at each
saved iter (500 / 1000 / 1500 / 2000) — is gate_net actually learning to
make α depend on τ, or is it still a constant function?

Runs conditioner forward once per sample to populate `_last_slow_feat /
_last_fast_feat`, then re-invokes `gated_fusion(slow_feat, fast_feat, t_emb)`
across a τ grid. No sampler / decoder / VAE needed.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=X \\
    /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \\
        -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29998 \\
        tools/probe_pathB_alpha_curve.py \\
        --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml \\
               configs/sf_v1/infer_stage3_v2.yaml \\
        --ckpt_path ckpts_5b/sf_v3_pathB_p1-<stamp>/<step> \\
        --jsonpath /public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json \\
        --max_samples 50 \\
        --dump_output results/pathB/alpha_curve_step500.json \\
        --seed 42

Output JSON schema:
    {
      "checkpoint": "<ckpt_path>",
      "t_emb_dim": 256,
      "tau_grid": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
      "stats_per_tau": {
        "alpha_key":   [{tau, mean, std, ...}, ...],
        ...
      },
      "stats_per_sample_range": {
        # For each channel, α(τ=τ_min) - α(τ=τ_max) aggregated over samples
        "alpha_key":   {mean, std, min, max},
        ...
      },
      "per_sample": [
        {"video_id": "...", "alphas": {"alpha_key": [val_at_tau0, ..., val_at_tau_last], ...}},
        ...
      ]
    }

Stop-or-go rule for P1:
  - If for ALL four channels, mean |α(τ=0) - α(τ=1)| across samples < 0.01
    at iter 500 → STOP. gate_net is not using t_emb (R1 risk realized).
  - If any channel shows |Δα| ≥ 0.05 and growing with iter → continue.
"""
import os
import sys
import json
import argparse

import torch
import numpy as np

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video_brain import SATVideoDiffusionEngineBrain
from arguments import get_args

from sgm.modules.diffusionmodules.sampling import timestep_embedding


CHANNELS = ["alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"]


def _stats(vals: np.ndarray) -> dict:
    return {
        "mean":   float(vals.mean()),
        "std":    float(vals.std()),
        "min":    float(vals.min()),
        "max":    float(vals.max()),
        "median": float(np.median(vals)),
        "q25":    float(np.quantile(vals, 0.25)),
        "q75":    float(np.quantile(vals, 0.75)),
    }


def probe_main(args, max_samples, dump_output, tau_grid):
    model = get_model(args, SATVideoDiffusionEngineBrain)
    load_checkpoint(model, args)
    model.eval()

    brain_embedder = model.conditioner.embedders[0]
    gated_fusion = brain_embedder.gated_fusion
    t_emb_dim = int(getattr(brain_embedder, "gated_fusion_t_emb_dim", 0))
    if t_emb_dim <= 0:
        raise RuntimeError(
            "Loaded model has gated_fusion_t_emb_dim<=0 — this is not a Path B "
            "model. Probe only makes sense when Path B t_emb path is active."
        )

    data = json.load(open(args.jsonpath))
    data = data[:max_samples]
    print(f"Probing α(sample, τ) for {len(data)} samples from {args.jsonpath}")
    print(f"τ grid: {tau_grid}")

    per_sample = []
    # shape: (n_samples, n_tau) per channel — fill in loop
    alpha_table = {k: np.zeros((len(data), len(tau_grid)), dtype=np.float64) for k in CHANNELS}

    with torch.no_grad():
        for i, item in enumerate(data):
            video_id = os.path.basename(item["video"]).split(".")[0]

            fmri_paths = item["fmri"]
            fmri_list = [torch.from_numpy(np.load(p)).unsqueeze(0) for p in fmri_paths]
            fmri_full = torch.cat(fmri_list, dim=0).unsqueeze(0).cuda()
            fmri = fmri_full[:, :, :8405]
            fmri_auditory = fmri_full[:, :, 8405:]

            eeg_paths = item["eeg"]
            eeg_list = [torch.from_numpy(np.load(p)[:64, :]).unsqueeze(0) for p in eeg_paths]
            eeg = torch.cat(eeg_list, dim=0).unsqueeze(0).cuda()

            batch = {
                "fmri": fmri,
                "fmri_auditory": fmri_auditory,
                "eeg": eeg,
                "num_frames": 33,
            }

            # Populate _last_slow_feat / _last_fast_feat via full conditioner forward
            _c, _uc = model.conditioner.get_unconditional_conditioning(
                batch,
                force_uc_zero_embeddings=["txt"],
            )

            slow_feat = brain_embedder._last_slow_feat
            fast_feat = brain_embedder._last_fast_feat
            if slow_feat is None or fast_feat is None:
                raise RuntimeError(
                    "_last_slow_feat / _last_fast_feat not cached. Did the model "
                    "actually hit the SF-dual-branch gated_fusion path?"
                )
            B = slow_feat.shape[0]

            per_tau_alphas = {k: [] for k in CHANNELS}
            for j, tau in enumerate(tau_grid):
                tau_t = slow_feat.new_full((B,), float(tau))
                t_emb = timestep_embedding(tau_t, dim=t_emb_dim)
                _, alphas = gated_fusion(slow_feat, fast_feat, t_emb=t_emb, tau=tau_t)
                for k in CHANNELS:
                    val = float(alphas[k].float().mean().item())
                    per_tau_alphas[k].append(val)
                    alpha_table[k][i, j] = val

            per_sample.append({
                "video_id": video_id,
                "alphas": per_tau_alphas,
            })

            # Per-sample Δα(τ=min vs τ=max) for quick visibility
            delta_line = "  Δ(τ_min→τ_max): "
            for k in CHANNELS:
                delta = per_tau_alphas[k][-1] - per_tau_alphas[k][0]
                delta_line += f"{k[6:]}={delta:+.4f} "
            print(f"[{i+1}/{len(data)}] {video_id}:{delta_line}")

    stats_per_tau = {k: [] for k in CHANNELS}
    for k in CHANNELS:
        for j, tau in enumerate(tau_grid):
            s = _stats(alpha_table[k][:, j])
            s["tau"] = float(tau)
            stats_per_tau[k].append(s)

    stats_per_sample_range = {}
    for k in CHANNELS:
        # Δα = α(τ=max) - α(τ=min), per sample
        deltas = alpha_table[k][:, -1] - alpha_table[k][:, 0]
        stats_per_sample_range[k] = {
            "mean_abs_delta":   float(np.abs(deltas).mean()),
            "mean_delta":       float(deltas.mean()),
            "std_delta":        float(deltas.std()),
            "min_delta":        float(deltas.min()),
            "max_delta":        float(deltas.max()),
            "frac_gt_0p05":     float((np.abs(deltas) > 0.05).mean()),
            "frac_gt_0p01":     float((np.abs(deltas) > 0.01).mean()),
        }

    if mpu.get_model_parallel_rank() == 0:
        os.makedirs(os.path.dirname(dump_output) or ".", exist_ok=True)
        out = {
            "num_samples": len(per_sample),
            "checkpoint": args.load,
            "t_emb_dim": t_emb_dim,
            "tau_grid": list(map(float, tau_grid)),
            "stats_per_tau": stats_per_tau,
            "stats_per_sample_range": stats_per_sample_range,
            "per_sample": per_sample,
        }
        with open(dump_output, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n=== α(sample, τ) probe — {len(per_sample)} samples → {dump_output} ===")
        print(f"{'channel':13s}  {'τ=min':>10s}  {'τ=max':>10s}  {'mean|Δ|':>10s}  frac|Δ|>0.05")
        for k in CHANNELS:
            tau_min_mean = stats_per_tau[k][0]["mean"]
            tau_max_mean = stats_per_tau[k][-1]["mean"]
            r = stats_per_sample_range[k]
            print(
                f"{k:13s}  {tau_min_mean:10.4f}  {tau_max_mean:10.4f}  "
                f"{r['mean_abs_delta']:10.4f}  {r['frac_gt_0p05']*100:4.0f}%"
            )
        # Headline go/no-go
        all_trivial = all(
            stats_per_sample_range[k]["mean_abs_delta"] < 0.01 for k in CHANNELS
        )
        print()
        if all_trivial:
            print(">>> VERDICT: α(τ) still trivial (mean|Δ|<0.01 on all channels).")
            print(">>> If this is iter 500 or later, consider stopping P1 (R1 risk).")
        else:
            any_strong = any(
                stats_per_sample_range[k]["mean_abs_delta"] >= 0.05 for k in CHANNELS
            )
            if any_strong:
                print(">>> VERDICT: α(τ) has strong signal (at least one channel |Δ|>=0.05). Continue.")
            else:
                print(">>> VERDICT: α(τ) has weak signal (0.01 <= mean|Δ| < 0.05). Monitor further.")


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--ckpt_path", type=str, default=None,
                           help="Override YAML's `load:` with this ckpt dir.")
    py_parser.add_argument("--max_samples", type=int, default=50)
    py_parser.add_argument("--dump_output", type=str,
                           default="results/pathB/alpha_curve_probe.json")
    py_parser.add_argument("--jsonpath", type=str,
                           default="/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json")
    py_parser.add_argument("--tau_grid", type=str,
                           default="0.0,0.1,0.3,0.5,0.7,0.9,1.0")
    known, args_list = py_parser.parse_known_args()

    # Ensure get_args sees --jsonpath (arguments.py expects it).
    if "--jsonpath" not in args_list:
        args_list = args_list + ["--jsonpath", known.jsonpath]

    args = get_args(args_list)
    del args.deepspeed_config
    if known.ckpt_path is not None:
        args.load = known.ckpt_path
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    tau_grid = [float(x) for x in known.tau_grid.split(",")]
    probe_main(
        args,
        max_samples=known.max_samples,
        dump_output=known.dump_output,
        tau_grid=tau_grid,
    )
