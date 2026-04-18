"""Regression check for direction ① path A.

Verifies that (1) the new compute_components+mix_context path matches the
legacy MultiGuidanceAdapter.forward output, and (2) the sampler's
_remix_cond_for_step returns the cond unchanged when alpha_schedule is null
or amp=0.

Does not require a GPU or checkpoint — uses randomly initialized modules and
mock tensors. Run with:
    python tools/test_alpha_remix_regression.py
"""
import sys
import torch

from sgm.modules.encoders.multi_guidance import MultiGuidanceAdapter
from sgm.modules.diffusionmodules.sampling import VPSDEDPMPP2MSampler


def _mock_inputs(B=1, S=226, brain_dim=4096, head_dim=1152, mot_dim=2048, device="cpu"):
    torch.manual_seed(0)
    z_b = torch.randn(B, S, brain_dim, device=device)
    slow_out = {
        "z_key": torch.randn(B, head_dim, device=device),
        "z_txt": torch.randn(B, head_dim, device=device),
    }
    fast_out = {"eeg_pooled_proj": torch.randn(B, mot_dim, device=device)}
    alphas = {
        "alpha_key": torch.rand(B, 1, device=device),
        "alpha_txt": torch.rand(B, 1, device=device),
        "alpha_mot": torch.rand(B, 1, device=device),
        "alpha_brain": torch.rand(B, 1, device=device),
    }
    return z_b, slow_out, fast_out, alphas


def test_compute_mix_matches_forward():
    torch.manual_seed(0)
    adapter = MultiGuidanceAdapter(brain_dim=4096, head_dim=1152, mot_input_dim=2048)
    adapter.eval()
    z_b, slow_out, fast_out, alphas = _mock_inputs()
    with torch.no_grad():
        ref = adapter(z_b, alphas, slow_out, fast_out)
        components = adapter.compute_components(slow_out, fast_out)
        out = adapter.mix_context(z_b, alphas, components)
    diff = (ref - out).abs().max().item()
    assert diff < 1e-6, f"forward vs compute+mix diverged: max|Δ|={diff}"
    print(f"[PASS] compute_components+mix_context == forward  (max|Δ|={diff:.2e})")


def test_remix_null_schedule_passthrough():
    sampler = VPSDEDPMPP2MSampler(
        alpha_schedule=None,
        discretization_config={
            "target": "sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization",
            "params": {"shift_scale": 1.0},
        },
        num_steps=50,
    )
    cond = {"crossattn": torch.randn(1, 226, 4096)}

    class StubEmbedder:
        _last_premix = None
        guidance_adapter = None

    out = sampler._remix_cond_for_step(cond, StubEmbedder(), i=5, num_sigmas=51)
    assert out is cond, "null schedule should return the same cond object"
    print("[PASS] null alpha_schedule returns cond unchanged (identity)")


def test_remix_amp_zero_equivalence():
    torch.manual_seed(0)
    adapter = MultiGuidanceAdapter(brain_dim=4096, head_dim=1152, mot_input_dim=2048)
    adapter.eval()
    z_b, slow_out, fast_out, alphas = _mock_inputs()
    with torch.no_grad():
        components = adapter.compute_components(slow_out, fast_out)
        static_ctx = adapter.mix_context(z_b, alphas, components)

    sampler = VPSDEDPMPP2MSampler(
        alpha_schedule={"type": "cosine", "amp": 0.0},
        discretization_config={
            "target": "sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization",
            "params": {"shift_scale": 1.0},
        },
        num_steps=50,
    )
    cond = {"crossattn": static_ctx.clone()}

    class StubEmbedder:
        pass

    stub = StubEmbedder()
    stub._last_premix = {"z_b": z_b, "components": components, "alphas_base": alphas}
    stub.guidance_adapter = adapter

    # Check across several steps — amp=0 must hold identically regardless of tau.
    with torch.no_grad():
        for i in [0, 10, 25, 40, 49]:
            out = sampler._remix_cond_for_step(cond, stub, i=i, num_sigmas=51)
            diff = (out["crossattn"] - static_ctx).abs().max().item()
            assert diff < 1e-5, f"amp=0 should match static; i={i}, max|Δ|={diff}"
    print("[PASS] amp=0 cosine schedule ≡ v2 static context (all tested steps)")


def test_remix_amp_nonzero_changes():
    torch.manual_seed(0)
    adapter = MultiGuidanceAdapter(brain_dim=4096, head_dim=1152, mot_input_dim=2048)
    adapter.eval()
    z_b, slow_out, fast_out, alphas = _mock_inputs()
    with torch.no_grad():
        components = adapter.compute_components(slow_out, fast_out)
        static_ctx = adapter.mix_context(z_b, alphas, components)

    sampler = VPSDEDPMPP2MSampler(
        alpha_schedule={"type": "linear", "amp": 0.5},
        discretization_config={
            "target": "sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization",
            "params": {"shift_scale": 1.0},
        },
        num_steps=50,
    )
    cond = {"crossattn": static_ctx.clone()}

    class StubEmbedder:
        pass

    stub = StubEmbedder()
    stub._last_premix = {"z_b": z_b, "components": components, "alphas_base": alphas}
    stub.guidance_adapter = adapter

    with torch.no_grad():
        ctx_early = sampler._remix_cond_for_step(cond, stub, i=0, num_sigmas=51)["crossattn"]
        ctx_mid = sampler._remix_cond_for_step(cond, stub, i=25, num_sigmas=51)["crossattn"]
        ctx_late = sampler._remix_cond_for_step(cond, stub, i=49, num_sigmas=51)["crossattn"]
    # Mid-step should be closest to static (τ≈0.5 → schedule ≈ 1.0).
    diff_early = (ctx_early - static_ctx).abs().mean().item()
    diff_mid = (ctx_mid - static_ctx).abs().mean().item()
    diff_late = (ctx_late - static_ctx).abs().mean().item()
    assert diff_mid < diff_early and diff_mid < diff_late, \
        f"expected τ=0.5 closest to static; got early={diff_early:.2e} mid={diff_mid:.2e} late={diff_late:.2e}"
    print(f"[PASS] linear amp=0.5 modulates context  "
          f"(early {diff_early:.2e} > mid {diff_mid:.2e} < late {diff_late:.2e})")


if __name__ == "__main__":
    test_compute_mix_matches_forward()
    test_remix_null_schedule_passthrough()
    test_remix_amp_zero_equivalence()
    test_remix_amp_nonzero_changes()
    print("\nAll regression checks passed.")
