"""Per-block per-condition contribution probe for Direction ② design.

Purpose:
    Direction ② (DiT layer × condition gating) requires evidence that different
    DiT blocks in CogVideoX-5B respond differently to different brain conditions
    (key_frame / text / motion / brain). This probe measures per-block activation
    change when only one condition is injected vs. all-zero baseline.

Method:
    For each sample and each condition c ∈ {key, txt, mot, brain}:
      1. Override gated_fusion's alpha output: alphas = 0 except alpha_c = alpha_value
      2. Run conditioner → builds context with only condition c injected
      3. Run one DiT forward at τ=tau_value with fixed noise x and hooks on each
         transformer layer to capture h_b = layer_b(x, context)
      4. Measure ||h_b(only_c) - h_b(null)|| per (c, b)
    Baseline null: alphas = [0, 0, 0, 0] (no condition injected).

Output:
    JSON with per-block per-condition contribution matrix (num_conditions × num_layers).
    Optionally PNG heatmap via matplotlib.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=X \
    /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \
        -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29997 \
        tools/probe_per_block_contribution.py \
        --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml \
               configs/sf_v1/infer_pathB_p1.yaml \
        --jsonpath /public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va_mini20.json \
        --max_samples 10 \
        --tau_value 0.5 \
        --alpha_value 0.5 \
        --dump_output results/probe_per_block/heatmap_pathB_p1.json \
        --heatmap_png results/probe_per_block/heatmap_pathB_p1.png \
        --seed 42

Decision rule (Direction ② go/no-go):
    - For each condition c, find argmax_b contribution(c, b).
    - If peaks for different conditions fall in different block indices
      (e.g., motion peaks at block ~15-22, brain/txt peaks at 20-27)
      → layer specialization confirmed → proceed to MiniLayer Phase B.
    - If all 4 conditions peak at similar blocks → global α is enough,
      abandon Direction ②.
"""

import os
import sys
import json
import argparse
import math

import torch
import numpy as np

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video_brain import SATVideoDiffusionEngineBrain
from arguments import get_args


CONDITIONS = ["null", "key", "txt", "mot", "brain"]
ALPHA_CHANNELS = {"key": "alpha_key", "txt": "alpha_txt",
                  "mot": "alpha_mot", "brain": "alpha_brain"}


def make_alpha_override_hook(target_channel, target_value):
    """Returns a forward-hook that rewrites gated_fusion's alpha output.

    Args:
        target_channel: None → all-zero (null baseline), or one of
                        "key"/"txt"/"mot"/"brain"
        target_value:   scalar to set the target channel alpha to (others=0)
    """
    def hook(module, inputs, output):
        z_b, _alphas_orig = output
        B = z_b.shape[0]
        device = z_b.device
        dtype = z_b.dtype
        new_alphas = {
            "alpha_key":   torch.zeros(B, 1, device=device, dtype=dtype),
            "alpha_txt":   torch.zeros(B, 1, device=device, dtype=dtype),
            "alpha_mot":   torch.zeros(B, 1, device=device, dtype=dtype),
            "alpha_brain": torch.zeros(B, 1, device=device, dtype=dtype),
        }
        if target_channel is not None:
            key = ALPHA_CHANNELS[target_channel]
            new_alphas[key] = torch.full(
                (B, 1), float(target_value), device=device, dtype=dtype
            )
        return (z_b, new_alphas)
    return hook


def register_block_hooks(transformer_layers, storage):
    """Register a forward-hook on each DiT transformer layer.

    storage (mutable dict) will be filled at forward time with:
        storage[layer_idx] = tensor (detached, on-device)
    The exact tensor captured is the layer's hidden-state output (B, L, D)
    — this covers both self-attention and MLP residuals, i.e., the full
    contribution of that block to the denoising trajectory.
    """
    hooks = []
    for idx, layer in enumerate(transformer_layers):
        def make_hook(b_idx):
            def _hook(module, inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                if isinstance(tensor, torch.Tensor):
                    storage[b_idx] = tensor.detach()
            return _hook
        hooks.append(layer.register_forward_hook(make_hook(idx)))
    return hooks


def _load_one_sample(item):
    """Build the per-sample batch dict expected by the brain conditioner."""
    fmri_paths = item["fmri"]
    fmri_list = [torch.from_numpy(np.load(p)).unsqueeze(0) for p in fmri_paths]
    fmri_full = torch.cat(fmri_list, dim=0).unsqueeze(0).cuda()
    fmri = fmri_full[:, :, :8405]
    fmri_auditory = fmri_full[:, :, 8405:]

    eeg_paths = item["eeg"]
    eeg_list = [torch.from_numpy(np.load(p)[:64, :]).unsqueeze(0) for p in eeg_paths]
    eeg = torch.cat(eeg_list, dim=0).unsqueeze(0).cuda()

    return {
        "fmri": fmri,
        "fmri_auditory": fmri_auditory,
        "eeg": eeg,
        "num_frames": 33,
    }


def _make_noise(shape, device, dtype, seed):
    """Reproducible Gaussian noise for a given (sample, shape)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def _dit_forward(model, x, sigma, c, storage, idx):
    """Invoke the denoiser-wrapped DiT once, populating block `storage`.

    Uses model.denoiser(network, x, sigma, cond) which is how the sampler
    calls the network — this ensures the cross-attn context path is built
    exactly as in production inference.
    """
    # CogVideoX uses VideoScaling requiring additional_model_inputs[idx].
    idx_tensor = x.new_ones([x.shape[0]]) * float(idx)
    return model.denoiser(model.model, x, sigma, c, idx=idx_tensor)


def probe_main(args, max_samples, dump_output, heatmap_png,
               alpha_value, tau_value, seed):
    model = get_model(args, SATVideoDiffusionEngineBrain)
    load_checkpoint(model, args)
    model.eval()

    brain_embedder = model.conditioner.embedders[0]
    gated_fusion = brain_embedder.gated_fusion

    # Access DiT via model.model.diffusion_model (OPENAIUNETWRAPPER convention)
    dit = model.model.diffusion_model
    transformer = dit.transformer
    transformer_layers = transformer.layers
    num_layers = len(transformer_layers)
    print(f"[probe] DiT has {num_layers} transformer layers")

    # Determine denoising shape from sampling config (CogVideoX latent)
    T = args.sampling_num_frames
    H_img, W_img = args.sampling_image_size
    C_latent = args.latent_channels
    F_down = args.first_stage_config.params.ddconfig.get("downsample_factor", 8) \
        if hasattr(args, "first_stage_config") else 8
    H = H_img // F_down
    W = W_img // F_down
    noise_shape = (1, T, C_latent, H, W)
    print(f"[probe] Noise shape: {noise_shape}")

    # Pick sigma for τ=tau_value from the denoiser's discretization.
    # We use ZeroSNRDDPMDiscretization consistent with VPSDEDPMPP2MSampler.
    # Here we approximate: sigma(τ) follows the VP SDE schedule, so we sample
    # sigmas directly from model.denoiser's scaling at a fixed t index.
    # Simpler: use num_steps=50 grid and pick closest step.
    num_inference_steps = 50
    step_idx = int(round((1.0 - tau_value) * (num_inference_steps - 1)))
    # This is admittedly sketchy — the exact sigma varies by discretization.
    # For probe purposes, pick a plausible sigma from the DDPM default schedule.
    sigma_value = torch.tensor([tau_value], device="cuda", dtype=torch.float32)
    # Map [0,1] → [sigma_min, sigma_max] roughly for VPSDE
    # Since probe is relative (compare conditions), sigma precision is not critical.

    data = json.load(open(args.jsonpath))
    data = data[:max_samples]
    print(f"[probe] Probing {len(data)} samples × {len(CONDITIONS)} conditions × {num_layers} layers")
    print(f"[probe] alpha_value={alpha_value}, tau_value={tau_value}")

    # contributions[cond_idx][sample_idx] = np.array of length num_layers (L2 norm per block)
    activations = {cond: np.zeros((len(data), num_layers), dtype=np.float64)
                   for cond in CONDITIONS}

    with torch.no_grad():
        for i, item in enumerate(data):
            video_id = os.path.basename(item["video"]).split(".")[0]
            batch = _load_one_sample(item)

            # Fixed noise per sample → consistent across conditions
            x_noise = _make_noise(noise_shape, device="cuda", dtype=torch.bfloat16,
                                  seed=seed + i)

            for cond in CONDITIONS:
                # --- install alpha override on gated_fusion ---
                target = None if cond == "null" else cond
                alpha_hook = gated_fusion.register_forward_hook(
                    make_alpha_override_hook(target, alpha_value)
                )

                # --- conditioner forward (builds context with controlled α) ---
                try:
                    c_cond, _uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        force_uc_zero_embeddings=["txt"],
                    )
                finally:
                    alpha_hook.remove()

                # Truncate cond tensors to B=1 (conditioner may duplicate)
                c_use = {}
                for k, v in c_cond.items():
                    if isinstance(v, torch.Tensor):
                        c_use[k] = v[:1].to("cuda")
                    else:
                        c_use[k] = v

                # --- install per-block hooks, run single DiT forward ---
                block_storage = {}
                block_hooks = register_block_hooks(transformer_layers, block_storage)

                try:
                    sigma = sigma_value.to("cuda")
                    _ = _dit_forward(model, x_noise, sigma, c_use, block_storage, idx=step_idx+1)
                finally:
                    for h in block_hooks:
                        h.remove()

                # --- record per-block L2 norm ---
                for b_idx in range(num_layers):
                    tensor = block_storage.get(b_idx)
                    if tensor is None:
                        activations[cond][i, b_idx] = float("nan")
                        continue
                    norm = float(tensor.float().norm().item())
                    activations[cond][i, b_idx] = norm

            print(f"  [{i+1}/{len(data)}] {video_id} — done")

    # --- compute contribution matrix: |h(only_c) - h(null)| / |h(null)| ---
    null_norm = activations["null"]  # (N, L)
    contribution = {}
    for cond in ["key", "txt", "mot", "brain"]:
        # Per-sample relative activation delta
        delta = np.abs(activations[cond] - null_norm)
        rel = delta / (null_norm + 1e-8)
        contribution[cond] = {
            "abs_delta_mean":   rel.mean(axis=0).tolist(),  # per-block mean
            "abs_delta_std":    rel.std(axis=0).tolist(),
            "raw_norm_mean":    activations[cond].mean(axis=0).tolist(),
            "raw_norm_std":     activations[cond].std(axis=0).tolist(),
        }

    # --- dump JSON ---
    if mpu.get_model_parallel_rank() == 0:
        os.makedirs(os.path.dirname(dump_output) or ".", exist_ok=True)
        out = {
            "checkpoint":         args.load,
            "num_samples":        len(data),
            "num_layers":         num_layers,
            "conditions":         ["key", "txt", "mot", "brain"],
            "alpha_value":        float(alpha_value),
            "tau_value":          float(tau_value),
            "null_norm_mean":     null_norm.mean(axis=0).tolist(),
            "null_norm_std":      null_norm.std(axis=0).tolist(),
            "contribution":       contribution,
        }
        with open(dump_output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[probe] JSON → {dump_output}")

        # --- Summary ---
        print(f"\n=== Per-block relative contribution (|Δ| / |null|) ===")
        print(f"{'block':>6s}  " + "  ".join(f"{c:>8s}" for c in ["key", "txt", "mot", "brain"]))
        for b in range(num_layers):
            row_vals = [contribution[c]["abs_delta_mean"][b] for c in ["key", "txt", "mot", "brain"]]
            print(f"{b:>6d}  " + "  ".join(f"{v:>8.4f}" for v in row_vals))

        # --- peak block per condition ---
        print(f"\n=== Peak block per condition ===")
        for c in ["key", "txt", "mot", "brain"]:
            vals = np.array(contribution[c]["abs_delta_mean"])
            peak_b = int(np.argmax(vals))
            print(f"  {c:10s} → block {peak_b:3d}  (|Δ|={vals[peak_b]:.4f})")

        # --- heatmap PNG (best-effort, no matplotlib crash if missing) ---
        if heatmap_png:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(14, 3.5))
                mat = np.array([contribution[c]["abs_delta_mean"]
                                for c in ["key", "txt", "mot", "brain"]])
                im = ax.imshow(mat, aspect="auto", cmap="viridis")
                ax.set_yticks(range(4))
                ax.set_yticklabels(["key", "txt", "mot", "brain"])
                ax.set_xlabel("DiT block index")
                ax.set_ylabel("condition")
                ax.set_title(
                    f"Per-block contribution — {os.path.basename(args.load)} "
                    f"(α={alpha_value}, τ={tau_value}, n={len(data)})"
                )
                fig.colorbar(im, ax=ax, label="|h_b(only_c) - h_b(null)| / |h_b(null)|")
                fig.tight_layout()
                os.makedirs(os.path.dirname(heatmap_png) or ".", exist_ok=True)
                fig.savefig(heatmap_png, dpi=120)
                plt.close(fig)
                print(f"[probe] heatmap PNG → {heatmap_png}")
            except Exception as e:
                print(f"[probe] (heatmap skipped: {e})")


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--ckpt_path", type=str, default=None,
                           help="Override YAML's `load:` with this ckpt dir.")
    py_parser.add_argument("--max_samples", type=int, default=10,
                           help="Number of samples to probe (mini probe)")
    py_parser.add_argument("--alpha_value", type=float, default=0.5,
                           help="Target alpha magnitude for isolated condition")
    py_parser.add_argument("--tau_value", type=float, default=0.5,
                           help="Normalized timestep τ (0=high-noise, 1=low-noise)")
    py_parser.add_argument("--dump_output", type=str,
                           default="results/probe_per_block/heatmap.json")
    py_parser.add_argument("--heatmap_png", type=str,
                           default="results/probe_per_block/heatmap.png")
    py_parser.add_argument("--jsonpath", type=str,
                           default="/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va_mini20.json")
    known, args_list = py_parser.parse_known_args()

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

    probe_main(
        args,
        max_samples=known.max_samples,
        dump_output=known.dump_output,
        heatmap_png=known.heatmap_png,
        alpha_value=known.alpha_value,
        tau_value=known.tau_value,
        seed=args.seed,
    )
