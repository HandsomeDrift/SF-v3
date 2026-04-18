"""Aggregate FVD/EPE/SSIM/PSNR/CLIP/CTC across alpha-schedule experiments.

Usage:
    python tools/eval_alpha_schedule.py \
        --gt-jsonpath /public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json \
        --result-dir E0_v2_static=/public/home/maoyaoxin/xxt/SF-v1/CineBrain/results/stage3_v2_sub05 \
        --result-dir E1_linear_mild=results/v3_alpha_E1_linear_mild \
        --result-dir E3_cosine=results/v3_alpha_E3_cosine

Each --result-dir value is `NAME=PATH`. All directories must contain
`{video_id:06d}.mp4` files and will be compared against GT on the same video
ID set. Output is a markdown table written to stdout; results are also saved
to JSON if --output is given.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import imageio.v3 as iio

from local_config import get_paths
from models.eval_metrics import (
    load_clip_model, load_vit_model,
    clip_score_only, ssim_score_only, psnr_score_only,
    clip_temporal_consistency,
    compute_fvd, compute_epe,
)


METRIC_HIGHER_IS_BETTER = {"FVD": False, "EPE": False, "SSIM": True,
                           "PSNR": True, "CLIP": True, "CTC": True}


def load_videos(result_dir, video_ids, n_frames=33):
    videos = []
    missing = []
    for vid in video_ids:
        p = os.path.join(result_dir, f"{str(vid).zfill(6)}.mp4")
        if not os.path.exists(p):
            missing.append(vid)
            continue
        videos.append(iio.imread(p)[:n_frames])
    return (np.stack(videos) if videos else None), missing


def compute_metrics(pred, gt, device, tag="", preloaded=None):
    if preloaded is None:
        vit_proc, vit_mod = load_vit_model(device=device)
        clip_proc, clip_mod = load_clip_model(device=device)
    else:
        (vit_proc, vit_mod), (clip_proc, clip_mod) = preloaded
    out = {}
    t = time.time()
    out["FVD"] = float(compute_fvd(pred, gt, device=device))
    print(f"  [{tag}] FVD: {out['FVD']:.4f} ({time.time()-t:.1f}s)")

    t = time.time()
    epe_mean, _ = compute_epe(pred, gt)
    out["EPE"] = float(epe_mean)
    print(f"  [{tag}] EPE: {epe_mean:.4f} ({time.time()-t:.1f}s)")

    t = time.time()
    ctc_mean, _ = clip_temporal_consistency(pred, device=device,
                                            preloaded=(clip_proc, clip_mod))
    out["CTC"] = float(ctc_mean)
    print(f"  [{tag}] CTC: {ctc_mean:.4f} ({time.time()-t:.1f}s)")

    ssim_l, psnr_l, clip_l = [], [], []
    for fi in range(pred.shape[1]):
        s_mean, _ = ssim_score_only(pred[:, fi], gt[:, fi])
        p_mean, _ = psnr_score_only(pred[:, fi], gt[:, fi])
        c_mean, _ = clip_score_only(pred[:, fi], gt[:, fi], device=device,
                                     preloaded=(clip_proc, clip_mod))
        ssim_l.append(s_mean); psnr_l.append(p_mean); clip_l.append(c_mean)
    out["SSIM"] = float(np.mean(ssim_l))
    out["PSNR"] = float(np.mean(psnr_l))
    out["CLIP"] = float(np.mean(clip_l))
    print(f"  [{tag}] SSIM: {out['SSIM']:.4f} PSNR: {out['PSNR']:.2f} CLIP: {out['CLIP']:.4f}")
    return out


def print_markdown_table(results, baseline_name=None):
    """results: dict[name] -> dict[metric] -> value."""
    names = list(results.keys())
    metrics = ["FVD", "EPE", "SSIM", "PSNR", "CLIP", "CTC"]
    header = "| Experiment | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines = [header, sep]
    for n in names:
        vals = [f"{results[n][m]:.4f}" for m in metrics]
        lines.append(f"| {n} | " + " | ".join(vals) + " |")
    print("\n" + "\n".join(lines))

    if baseline_name is not None and baseline_name in results:
        base = results[baseline_name]
        print(f"\n### Δ vs {baseline_name}\n")
        dheader = "| Experiment | " + " | ".join(metrics) + " |"
        print(dheader); print(sep)
        for n in names:
            if n == baseline_name:
                continue
            parts = []
            for m in metrics:
                d = results[n][m] - base[m]
                better = (d < 0) if not METRIC_HIGHER_IS_BETTER[m] else (d > 0)
                mark = "✓" if better else "✗"
                parts.append(f"{'+' if d >= 0 else ''}{d:.4f} {mark}")
            print(f"| {n} | " + " | ".join(parts) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-jsonpath", required=True,
                    help="e.g. /public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json")
    ap.add_argument("--result-dir", action="append", required=True,
                    help="NAME=PATH format. Can be repeated.")
    ap.add_argument("--baseline", default="E0_v2_static",
                    help="Name (from --result-dir) used as delta baseline.")
    ap.add_argument("--output", default=None,
                    help="Optional JSON dump of results dict.")
    ap.add_argument("--n-frames", type=int, default=33)
    args = ap.parse_args()

    experiments = []
    for entry in args.result_dir:
        if "=" not in entry:
            sys.exit(f"--result-dir must be NAME=PATH, got: {entry}")
        name, path = entry.split("=", 1)
        experiments.append((name, path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_paths()

    items = json.load(open(args.gt_jsonpath))
    video_ids = sorted(int(os.path.basename(d["video"]).split(".")[0]) for d in items)
    print(f"Evaluating {len(video_ids)} videos: {video_ids[0]}-{video_ids[-1]}")

    gt_videos, missing_gt = [], []
    for vid in video_ids:
        p = os.path.join(paths["video_dir"], f"{str(vid).zfill(6)}.mp4")
        if not os.path.exists(p):
            missing_gt.append(vid); continue
        gt_videos.append(iio.imread(p)[:args.n_frames])
    gt = np.stack(gt_videos)
    print(f"GT shape: {gt.shape}, missing: {len(missing_gt)}")

    # Preload metric models once
    preloaded = (load_vit_model(device=device), load_clip_model(device=device))

    results = {}
    for name, path in experiments:
        print("\n" + "=" * 60)
        print(f"  {name}  (dir={path})")
        print("=" * 60)
        pred, missing = load_videos(path, video_ids, n_frames=args.n_frames)
        if pred is None:
            print(f"  [SKIP] no videos found under {path}"); continue
        if missing:
            print(f"  [WARN] {len(missing)} missing videos in {name}")
        results[name] = compute_metrics(pred, gt, device, tag=name, preloaded=preloaded)

    print_markdown_table(results, baseline_name=args.baseline)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
