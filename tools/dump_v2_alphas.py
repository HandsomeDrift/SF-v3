"""Dump v2 checkpoint learned alpha_base across N samples.

Usage (single GPU, no sampler / decoder — only conditioner forward):

    CUDA_VISIBLE_DEVICES=X \\
    /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \\
        -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29999 \\
        tools/dump_v2_alphas.py \\
        --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/infer_stage3_v2.yaml \\
        --seed 42 \\
        --jsonpath /public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json \\
        --max-samples 50 \\
        --dump-output results/alpha_540/v2_alpha_base_dump.json

Produces JSON with per-sample alphas + stats (mean/std/min/max/median per channel).
Purpose: validate H* (v2 gate_net saturation hypothesis) in DEBUG_direction1_inversion_finding.md.
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


def dump_main(args, max_samples, dump_output):
    model = get_model(args, SATVideoDiffusionEngineBrain)
    load_checkpoint(model, args)
    model.eval()

    data = json.load(open(args.jsonpath))
    data = data[:max_samples]
    print(f"Dumping alphas for {len(data)} samples from {args.jsonpath}")

    alpha_records = []

    with torch.no_grad():
        for i, item in enumerate(data):
            video_id = os.path.basename(item["video"]).split('.')[0]

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

            # Only forward conditioner — _last_alphas will be populated
            _c, _uc = model.conditioner.get_unconditional_conditioning(
                batch,
                force_uc_zero_embeddings=["txt"],
            )

            brain_embedder = model.conditioner.embedders[0]
            alphas = brain_embedder._last_alphas

            record = {
                "video_id": video_id,
                "alpha_key":   float(alphas["alpha_key"].cpu().float().item()),
                "alpha_txt":   float(alphas["alpha_txt"].cpu().float().item()),
                "alpha_mot":   float(alphas["alpha_mot"].cpu().float().item()),
                "alpha_brain": float(alphas["alpha_brain"].cpu().float().item()),
            }
            alpha_records.append(record)
            print(
                f"  [{i+1}/{len(data)}] {video_id}: "
                f"key={record['alpha_key']:.4f} txt={record['alpha_txt']:.4f} "
                f"mot={record['alpha_mot']:.4f} brain={record['alpha_brain']:.4f}"
            )

    # Aggregate stats per channel
    stats = {}
    for key in ["alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"]:
        vals = np.array([r[key] for r in alpha_records], dtype=np.float64)
        stats[key] = {
            "mean":   float(vals.mean()),
            "std":    float(vals.std()),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
            "median": float(np.median(vals)),
            "q25":    float(np.quantile(vals, 0.25)),
            "q75":    float(np.quantile(vals, 0.75)),
        }

    if mpu.get_model_parallel_rank() == 0:
        os.makedirs(os.path.dirname(dump_output) or ".", exist_ok=True)
        output = {
            "num_samples": len(alpha_records),
            "checkpoint": args.load,
            "jsonpath": args.jsonpath,
            "stats": stats,
            "per_sample": alpha_records,
        }
        with open(dump_output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n=== Dumped {len(alpha_records)} samples to {dump_output} ===")
        print("Per-channel stats:")
        for k, s in stats.items():
            sat_lo = (np.array([r[k] for r in alpha_records]) < 0.05).mean()
            sat_hi = (np.array([r[k] for r in alpha_records]) > 0.95).mean()
            print(
                f"  {k:13s} mean={s['mean']:.4f} std={s['std']:.4f} "
                f"[min={s['min']:.4f} q25={s['q25']:.4f} med={s['median']:.4f} "
                f"q75={s['q75']:.4f} max={s['max']:.4f}]  "
                f"sat<0.05={sat_lo*100:.0f}%  sat>0.95={sat_hi*100:.0f}%"
            )


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--max-samples", type=int, default=50)
    py_parser.add_argument(
        "--dump-output",
        type=str,
        default="results/alpha_540/v2_alpha_base_dump.json",
    )
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    dump_main(args, max_samples=known.max_samples, dump_output=known.dump_output)
