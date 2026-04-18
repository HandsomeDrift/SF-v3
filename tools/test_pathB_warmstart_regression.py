"""Direction ① Path B — warm-start regression test.

Verifies two critical equivalences that must hold before Path B training:

  (T1) v2 CrossModalGatedFusion (t_emb_dim=0) vs Path B (t_emb_dim>0) with
       gate_net.0.weight right-padded with zeros and t_emb_proj last layer
       zero-init, fed the SAME (slow_feat, fast_feat) and t_emb=None, produces
       BIT-IDENTICAL z_b and α. This is the partial-load warm-start guarantee.

  (T2) Path B with non-trivial t_emb values produces identical α as T1 *only*
       while t_emb_proj last layer is zero-init. Once we manually set t_emb_proj
       last layer to non-zero, α must diverge on the same (slow, fast). This
       proves gate_net's architectural capacity to use the τ input.

Run with:
    python tools/test_pathB_warmstart_regression.py

Expected output: "PASS T1/T2" on both lines. Any mismatch is a bug — do not
start P1 training until both pass.
"""
import sys
from pathlib import Path

import torch

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgm.modules.encoders.gated_fusion import CrossModalGatedFusion  # noqa: E402


HIDDEN_DIM = 2048
T_EMB_DIM = 256
T_EMB_PROJ_DIM = 512
S = 16  # reduced num_spatial for test speed (gated_fusion uses this for pos_embed)


def _build_pair():
    v2 = CrossModalGatedFusion(
        hidden_dim=HIDDEN_DIM, num_layers=1, num_spatial=S, t_emb_dim=0,
    )
    pb = CrossModalGatedFusion(
        hidden_dim=HIDDEN_DIM, num_layers=1, num_spatial=S,
        t_emb_dim=T_EMB_DIM, t_emb_proj_dim=T_EMB_PROJ_DIM,
    )
    v2.eval()
    pb.eval()
    return v2, pb


def _partial_load(v2: CrossModalGatedFusion, pb: CrossModalGatedFusion) -> None:
    """Mirror tools/partial_load_v2_to_pathB.py logic on in-memory state dicts."""
    v2_sd = v2.state_dict()
    pb_sd = pb.state_dict()
    missing_in_pb = []
    for k, v in v2_sd.items():
        if k not in pb_sd:
            missing_in_pb.append(k)
            continue
        target = pb_sd[k]
        if target.shape == v.shape:
            pb_sd[k] = v.clone()
        elif k == "gate_net.0.weight":
            # (out, hidden + t_emb_proj_dim): pad right with zeros.
            pad_cols = target.shape[1] - v.shape[1]
            assert pad_cols == T_EMB_PROJ_DIM, (k, v.shape, target.shape)
            pad = torch.zeros(v.shape[0], pad_cols, dtype=v.dtype)
            pb_sd[k] = torch.cat([v, pad], dim=1)
        else:
            raise RuntimeError(f"Unexpected shape mismatch on {k}: {v.shape} vs {target.shape}")
    if missing_in_pb:
        raise RuntimeError(f"v2 keys missing in Path B: {missing_in_pb}")
    pb.load_state_dict(pb_sd)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_t1():
    torch.manual_seed(42)
    v2, pb = _build_pair()
    _partial_load(v2, pb)

    slow = torch.randn(2, S, HIDDEN_DIM)
    fast = torch.randn(2, S, HIDDEN_DIM)
    with torch.no_grad():
        z_v2, a_v2 = v2(slow, fast)
        z_pb, a_pb = pb(slow, fast, t_emb=None)

    dz = _max_abs(z_v2, z_pb)
    dalphas = {k: _max_abs(a_v2[k], a_pb[k]) for k in a_v2}
    print(f"[T1] max|Δz_b| = {dz:.3e}")
    for k, v in dalphas.items():
        print(f"[T1] max|Δ{k}| = {v:.3e}")

    tol = 1e-6
    if dz > tol or any(v > tol for v in dalphas.values()):
        print("FAIL T1: Path B with t_emb=None + zero-init t_emb_proj diverges from v2.")
        return False
    print("PASS T1: Path B iter-0 matches v2 bit-identical.")
    return True


def run_t2():
    """Verify gate_net has capacity to use t_emb AND no gradient deadlock.

    Post-fix design: t_emb_proj[-1] is small-random-init (not zero) so t_emb_feat
    is non-zero at iter 0. Warm-start equivalence is guaranteed by the zero-pad
    on gate_net[0].weight's t_emb cols (partial_load). α does not depend on τ
    at iter 0 because W_t_emb_cols annihilates t_emb_feat.

    T2 checks:
      (a) With partial-loaded gate_net (zero-padded cols), random t_emb yields
          α identical to v2 — no τ dependency at init.
      (b) If we manually break the zero-pad (simulate gradient moving W_t_emb),
          α shifts with t_emb — confirms gate_net CAN use t_emb.
    """
    torch.manual_seed(42)
    v2, pb = _build_pair()
    _partial_load(v2, pb)
    # Simulate a trained v2 state: gate_net[2] has non-trivial weights so α
    # depends on gate_net[0]'s output (otherwise α is always 0.5 and T2b can't
    # detect any change).
    with torch.no_grad():
        pb.gate_net[2].weight.normal_(std=0.02)
        pb.gate_net[2].bias.normal_(std=0.02)

    slow = torch.randn(2, S, HIDDEN_DIM)
    fast = torch.randn(2, S, HIDDEN_DIM)
    t_emb = torch.randn(2, T_EMB_DIM)

    with torch.no_grad():
        _, a_none = pb(slow, fast, t_emb=None)
        _, a_with_t = pb(slow, fast, t_emb=t_emb)

    dalphas_no_effect = {k: _max_abs(a_none[k], a_with_t[k]) for k in a_none}
    max_no_effect = max(dalphas_no_effect.values())
    print(f"[T2a] max|Δα with random t_emb (zero-padded gate_net cols)| = {max_no_effect:.3e}")
    if max_no_effect > 1e-6:
        print(f"FAIL T2a: zero-padded gate_net cols should absorb any t_emb, got {dalphas_no_effect}")
        return False

    # Simulate gradient having moved W_t_emb_cols — break the zero-pad.
    with torch.no_grad():
        pb.gate_net[0].weight[:, HIDDEN_DIM:].normal_(std=0.02)
        _, a_broken = pb(slow, fast, t_emb=t_emb)
    dalphas_broken = {k: _max_abs(a_none[k], a_broken[k]) for k in a_none}
    max_delta = max(dalphas_broken.values())
    print(f"[T2b] max|Δα with random t_emb (broken zero-pad)| = {max_delta:.3e}")
    if max_delta < 1e-4:
        print("FAIL T2b: breaking zero-pad failed to change α — gate_net not capacity-limited.")
        return False
    print("PASS T2: gate_net responds to t_emb once W_t_emb_cols leaves zero-pad.")
    return True


def main():
    ok1 = run_t1()
    print()
    ok2 = run_t2()
    print()
    if not (ok1 and ok2):
        sys.exit(1)
    print("All regression checks passed. Safe to proceed to P1 training.")


if __name__ == "__main__":
    main()
