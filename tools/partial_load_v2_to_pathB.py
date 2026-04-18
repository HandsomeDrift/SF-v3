"""Partial-load v2 checkpoint into Path B model shape.

Path B extends CrossModalGatedFusion.gate_net[0] from Linear(hidden, hidden//2)
to Linear(hidden + t_emb_proj_dim, hidden//2) by concatenating a t_emb feature
onto the pooled fusion features. The v2 checkpoint has the smaller shape.

This script:
  1. Walks the v2 checkpoint's model_states file.
  2. Right-pads `conditioner.embedders.0.gated_fusion.gate_net.0.weight` on
     the input dim with zeros of width t_emb_proj_dim.
  3. Leaves `gate_net.0.bias`, `gate_net.2.weight`, `gate_net.2.bias`, and
     t_emb_proj.* untouched (t_emb_proj keys will be missing in v2 ckpt → the
     new model initializes them from scratch with zero-init last layer).
  4. Writes a new DeepSpeed-compatible checkpoint directory.

Forward equivalence at iter 0:
  Padded zero columns × any t_emb_feat = 0. Zero-init t_emb_proj last layer
  → t_emb_feat = 0 regardless of input. Either alone guarantees iter-0 α
  equals v2 α on the same (slow_feat, fast_feat). Both together is belt-and-
  suspenders safety.

Usage:
  python tools/partial_load_v2_to_pathB.py \
      --src_ckpt_dir /path/to/v2_ckpt \
      --dst_ckpt_dir ckpts_5b/sf_v3_pathB_init \
      --t_emb_proj_dim 512
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch


GATE_NET_KEY = "conditioner.embedders.0.gated_fusion.gate_net.0.weight"


def _find_model_state_file(ckpt_dir: Path) -> Path:
    """Locate the DeepSpeed model_states .pt inside a checkpoint directory."""
    latest_file = ckpt_dir / "latest"
    if latest_file.exists():
        step_name = latest_file.read_text().strip()
        step_dir = ckpt_dir / step_name
        if step_dir.is_dir():
            for p in step_dir.glob("mp_rank_*_model_states.pt"):
                return p

    for p in ckpt_dir.rglob("mp_rank_*_model_states.pt"):
        return p

    raise FileNotFoundError(
        f"Could not find DeepSpeed model_states file under {ckpt_dir}. "
        f"Expected structure: {ckpt_dir}/<stepN>/mp_rank_00_model_states.pt"
    )


def _pad_gate_net_weight(weight: torch.Tensor, pad_cols: int) -> torch.Tensor:
    """Right-pad a (out, in) Linear weight with zeros to (out, in + pad_cols)."""
    if weight.dim() != 2:
        raise ValueError(f"Expected 2-D Linear weight, got shape {tuple(weight.shape)}")
    pad = torch.zeros(
        weight.shape[0], pad_cols,
        dtype=weight.dtype, device=weight.device,
    )
    return torch.cat([weight, pad], dim=1)


def convert(src_ckpt_dir: Path, dst_ckpt_dir: Path, t_emb_proj_dim: int) -> None:
    src_state_file = _find_model_state_file(src_ckpt_dir)
    print(f"[partial-load] source state file: {src_state_file}")

    print("[partial-load] loading state dict (cpu) ...")
    state = torch.load(src_state_file, map_location="cpu", weights_only=False)

    model_state = state.get("module") or state.get("model") or state
    found_inline = False
    if isinstance(model_state, dict) and GATE_NET_KEY in model_state:
        found_inline = True
    if not found_inline:
        # Some SAT/DeepSpeed ckpts nest state_dict under different keys.
        for candidate_key in ("model_state_dict", "state_dict"):
            if isinstance(state.get(candidate_key), dict) and GATE_NET_KEY in state[candidate_key]:
                model_state = state[candidate_key]
                found_inline = True
                break

    if not found_inline:
        preview = list(state.keys()) if isinstance(state, dict) else type(state)
        raise KeyError(
            f"'{GATE_NET_KEY}' not found in checkpoint. Top-level keys: {preview}. "
            f"Adjust this script if the ckpt nests state under a different key."
        )

    old_w = model_state[GATE_NET_KEY]
    print(f"[partial-load] '{GATE_NET_KEY}' shape: {tuple(old_w.shape)}")
    new_w = _pad_gate_net_weight(old_w, t_emb_proj_dim)
    model_state[GATE_NET_KEY] = new_w
    print(f"[partial-load] padded to shape: {tuple(new_w.shape)} (added {t_emb_proj_dim} zero cols)")

    dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Preserve DeepSpeed's step-nested layout. Use the same step dir name.
    src_step_dir = src_state_file.parent
    dst_step_dir = dst_ckpt_dir / src_step_dir.name
    dst_step_dir.mkdir(parents=True, exist_ok=True)

    dst_state_file = dst_step_dir / src_state_file.name
    print(f"[partial-load] saving to: {dst_state_file}")
    torch.save(state, dst_state_file)

    # Write a `latest` pointer so DeepSpeed can find the step dir.
    (dst_ckpt_dir / "latest").write_text(src_step_dir.name)

    # Also copy any small metadata files (not optimizer states — those are
    # stale for the new shape; finetune mode should not need them).
    for name in ("latest_checkpointed_iteration.txt",):
        src_meta = src_ckpt_dir / name
        if src_meta.exists():
            shutil.copy2(src_meta, dst_ckpt_dir / name)

    print("[partial-load] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_ckpt_dir", required=True, type=str)
    ap.add_argument("--dst_ckpt_dir", required=True, type=str)
    ap.add_argument("--t_emb_proj_dim", default=512, type=int)
    args = ap.parse_args()

    src = Path(args.src_ckpt_dir).resolve()
    dst = Path(args.dst_ckpt_dir).resolve()
    if not src.is_dir():
        sys.exit(f"Source ckpt dir not found: {src}")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"Destination ckpt dir is not empty: {dst}")

    convert(src, dst, args.t_emb_proj_dim)


if __name__ == "__main__":
    main()
