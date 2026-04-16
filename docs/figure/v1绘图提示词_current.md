# CineBrain-SF v1 绘图提示词（当前版本）

> 基于实际实现的架构，更新自初版提示词。主要变化：Fast Branch 从分类头改为蒸馏+时序动态解码器，增加三阶段训练流程图。

---

## Figure 1: 模型总览（Model Overview）

```
Create a clean, publication-quality scientific figure for a brain-computer interface / video generation paper. The figure is a model overview diagram for a framework named "CineBrain-SF v1". Use a modern vector infographic style, white background, neat layout, soft shadows, minimal but clear text, suitable for a NeurIPS / CVPR paper. Make the diagram wide (landscape), and easy to read at single-column width.

Main idea:
The model reconstructs continuous video from synchronized multimodal brain signals (fMRI + EEG) during naturalistic movie viewing. It uses slow-fast disentanglement:
1) a Slow Branch (fMRI-dominant) for semantic and keyframe information,
2) a Fast Branch (EEG-dominant) using feature distillation and a temporal dynamics decoder for motion and temporal changes,
3) a Cross-Modal Gated Fusion module with adaptive gating weights,
4) a Multi-Guidance Adapter that injects multi-channel conditioning into a CogVideoX-5B DiT video diffusion model via cross-attention.

Layout from left to right:

═══════════════════════════════════════════════════

[LEFT] Input Block
- Top-left: a small filmstrip icon representing the naturalistic movie stimulus
- Below: two parallel input streams, clearly separated:
  - fMRI input (blue tone): labeled "fMRI (226 ROIs, 2s TR)" — show a brain icon with blue highlight, an arrow pointing to a box "fMRI Encoder (24-layer Transformer)"
  - EEG input (orange tone): labeled "EEG (64ch × 800 samples, 200Hz)" — show a head icon with orange EEG traces, an arrow pointing to a box "EEG Encoder (12-layer Transformer)"
- Both encoders output (B, 226, 2048) spatial tokens. Show this dimensionality.
- fMRI encoder also outputs a CLS token (B, 1152). Show this with a small branch.

═══════════════════════════════════════════════════

[MIDDLE-TOP] Slow Semantic Branch (blue shaded region)
- Label: "Slow Branch (fMRI-dominant)"
- Input: fMRI spatial tokens (B, 226, 2048)
- Internal components (show as small rounded boxes inside the blue region):
  - "Keyframe Head" → outputs z_key (B, 1152): a global vector representing keyframe semantics
  - "Scene-Text Head" → outputs z_txt (B, 1152): a global vector representing text/caption semantics
- Also passes raw slow_feat (B, 226, 2048) directly to Fusion
- Use a blue-toned region with clean internal layout
- Visual metaphor: slow, stable, semantic — perhaps a "glacier" or "anchor" icon

═══════════════════════════════════════════════════

[MIDDLE-BOTTOM] Fast Motion-Dynamics Branch (orange shaded region)
- Label: "Fast Branch (EEG-dominant)"
- Input: EEG spatial tokens (B, 226, 2048)
- This branch has TWO sub-stages, shown as sequential blocks inside the orange region:

  Sub-stage 1: "P0 — Feature Distillation"
  - "Temporal Attention Pool" → pooled features (B, 2048)
  - "Distillation Projector (Spatial)" → eeg_pooled_proj (B, 2048)
  - "Distillation Projector (CLS)" → eeg_cls_proj (B, 1152)
  - Show a DASHED arrow from fMRI encoder outputs to here, labeled "MSE distillation (teacher: frozen fMRI)"
  - This indicates the EEG learns to approximate fMRI features

  Sub-stage 2: "P1 — Temporal Dynamics Decoder"
  - A cross-attention decoder block:
    - Input: T+1 learnable query tokens (T=9 temporal + 1 global)
    - Cross-attention keys/values: EEG spatial tokens (226)
    - Self-attention with optional causal mask
  - Outputs (show as three small labeled arrows):
    - temporal_tokens (B, 9, 1152): per-frame temporal embeddings representing delta changes
    - global_dyn_token (B, 1152): global dynamics summary
    - flow_traj_pred (B, 9): predicted per-frame motion magnitude
    - dyn_logits (B, 2): coarse dynamic/static classification
  - Also passes raw fast_feat (B, 226, 2048) to Fusion
  - Visual metaphor: fast, dynamic — perhaps a "lightning" or "wave" icon

═══════════════════════════════════════════════════

[CENTER] Cross-Modal Gated Fusion (purple/violet shaded region)
- Label: "Cross-Modal Gated Fusion"
- Inputs: slow_feat (B, 226, 2048) from top and fast_feat (B, 226, 2048) from bottom
- Internal architecture (show compactly):
  - "Concat + Linear Projection" → (B, 226, 2048) hidden
  - Two-stream modality embeddings (slow embed + fast embed)
  - "4-layer Fusion Transformer" processing 2×226 = 452 tokens
  - Output: take first 226 tokens → "Linear → z_b (B, 226, 4096)"
  - "Gating Network" branch from pooled features → 4 sigmoid outputs
- Show 4 gating weights as small gauge/dial icons:
  - α_key (keyframe weight)
  - α_txt (text weight)  
  - α_mot (motion weight)
  - α_brain (brain latent weight)
- These alpha arrows should flow rightward to the Guidance Adapter
- Use purple/violet tone

═══════════════════════════════════════════════════

[RIGHT-CENTER] Multi-Guidance Adapter (teal/green-blue shaded region)
- Label: "Multi-Guidance Adapter"
- Inputs from left:
  - z_b (B, 226, 4096) from Fusion — main latent
  - z_key (B, 1152) from Slow Branch → "Key Proj → 4096" → weighted by α_key
  - z_txt (B, 1152) from Slow Branch → "Txt Proj → 4096" → weighted by α_txt
  - eeg_pooled_proj (B, 2048) from Fast Branch → "Mot Proj → 4096" → weighted by α_mot
  - z_b itself (residual) → weighted by α_brain
- Show the additive combination:
  context = z_b + α_key·g_key + α_txt·g_txt + α_mot·g_mot + α_brain·z_b
- Output: context (B, 226, 4096) — the conditioning tensor for the diffusion model

═══════════════════════════════════════════════════

[FAR RIGHT] Video Diffusion Decoder (green shaded region)
- Label: "CogVideoX-5B DiT + LoRA"
- Show a diffusion process icon: noise → iterative denoising → clean video
- The context (B, 226, 4096) from Multi-Guidance Adapter enters via cross-attention at each DiT block
- Show "LoRA" as small adapter icons attached to the DiT blocks (indicating fine-tuning)
- Input: Gaussian noise z_T (latent video)
- Output: reconstructed video clip (show a small video filmstrip, ~2 seconds, 13 frames @ 8fps)
- Use green tone

═══════════════════════════════════════════════════

[BOTTOM] Training Supervision Strip
- A slim horizontal strip below the main diagram, labeled "Offline Supervision Targets"
- Show cached target icons with upward dashed arrows to relevant heads:
  - "SigLIP Keyframe Emb" → Keyframe Head (L_key)
  - "SigLIP Caption Emb" → Scene-Text Head (L_txt)
  - "fMRI Features (teacher)" → P0 Distillation (L_distill)
  - "SigLIP Frame Δ Embs" → P1 Temporal Decoder (L_temporal_delta, L_temporal_abs)
  - "Optical Flow Trajectory" → P1 Flow Head (L_flow_traj)
  - "Dynamic/Static Label" → P1 Dynamics Head (L_dyn)
  - "Diffusion Target ε" → DiT output (L_diff)
- Each target-loss pair should have a small connecting line

═══════════════════════════════════════════════════

Visual design:
- Color scheme: blue=fMRI/Slow, orange=EEG/Fast, purple=Fusion, teal=Guidance, green=Decoder
- Use rounded rectangles for modules, arrows for data flow
- Dimension annotations in small gray text: (B, 226, 2048), (B, 1152), (B, 226, 4096), etc.
- Keep text concise: module names only, no equations
- The slow-fast separation should be visually prominent (top/bottom split in the middle section)
- Show the dashed distillation arrow from fMRI encoder to Fast Branch to emphasize the teacher-student relationship
- The gating weights (α) should be visually connected from Fusion to Guidance with colored lines
- Final style: polished paper figure, not a cartoon

Scientific emphasis:
- Synchronized fMRI (slow, semantic) + EEG (fast, temporal) inputs
- Feature distillation: EEG learns from fMRI as teacher (P0)
- Temporal dynamics decoder extracts multi-frame temporal structure from EEG (P1)
- Adaptive gating determines the contribution of each guidance channel
- Multi-guidance conditioning via cross-attention into a pretrained video DiT
- LoRA fine-tuning allows the DiT to adapt to brain-conditioned generation
```

---

## Figure 2: 三阶段训练流程（Three-Stage Training）

```
Create a clean, publication-quality figure showing the three-stage training curriculum for CineBrain-SF v1. Use the same visual style as Figure 1 (white background, soft shadows, color-coded modules). Layout: three panels arranged left to right, one per stage.

Stage 1 — "Branch Pre-training"
- Show the Slow Branch (blue) and Fast Branch (orange) being trained
- The DiT decoder (green) is shown grayed out / locked icon (frozen)
- The Fusion module (purple) is grayed out / locked (frozen)
- Active losses (show with colored lines from targets to heads):
  - L_key, L_txt (Slow Branch supervision)
  - L_distill_cls, L_distill_spatial (P0 distillation from frozen fMRI)
  - L_temporal_delta, L_temporal_abs, L_flow_traj, L_dyn (P1 temporal supervision)
- No L_diff (diffusion loss not computed)
- Caption: "Stage 1: Learn brain encoders and branch-specific representations"

Stage 2 — "Fusion Training"  
- The Slow Branch (blue) and Fast Branch (orange) are grayed out / locked (frozen)
- The DiT decoder (green) is grayed out / locked (frozen) but shows gradient flow through it (dashed green arrow)
- GatedFusion (purple) and MultiGuidanceAdapter (teal) are highlighted (trainable)
- Active losses:
  - L_diff (diffusion loss, gradient flows back through frozen DiT to fusion)
  - L_guide = L_gk + L_gt (guidance consistency losses)
- Caption: "Stage 2: Learn fusion weights and guidance injection"

Stage 3 — "Joint Fine-tuning"
- All modules active (no grayed out)
- The DiT decoder (green) has small "LoRA" adapter icons (only LoRA weights trainable, not full DiT)
- Fast Branch (orange) is unfrozen
- GatedFusion (purple) and MultiGuidanceAdapter (teal) are active
- Slow Branch (blue) stays frozen
- Active losses:
  - L_diff (diffusion loss, main objective)
  - λ_sf × (L_slow + L_fast + L_guide + L_struct) — auxiliary SF losses
- Caption: "Stage 3: Joint optimization with LoRA-adapted DiT"

Between the three panels, show progression arrows labeled:
Stage 1 → Stage 2: "Freeze branches, enable fusion"
Stage 2 → Stage 3: "Add LoRA, unfreeze fast branch"

Visual design:
- Trainable modules: full color + glow/highlight
- Frozen modules: grayed out + lock icon
- Gradient flow: solid colored arrows for active, dashed gray for pass-through
- Loss labels in small italic text near relevant heads
- Each panel should be compact but clearly show which modules are active vs frozen
```

---

## Figure 3: Fast Branch 内部架构（Fast Branch Detail）

```
Create a detailed architecture diagram for the Fast Branch of CineBrain-SF v1. This figure zooms into the Fast Branch to show its two-phase design (P0 distillation + P1 temporal dynamics decoder). Use the same visual style as previous figures, with orange as the primary color.

Layout: vertical, top to bottom.

[TOP] Input
- EEG signal: (B, 5, 64, 800) — 5 trials × 64 channels × 800 time samples
- EEG Encoder: 12-layer Transformer
- Outputs: eeg_cls (B, 1152) CLS token + eeg_spatial (B, 226, 2048) spatial tokens

[MIDDLE-LEFT] P0: Feature Distillation
- From eeg_spatial → Temporal Attention Pool → pooled (B, 2048)
- Spatial Projector: pooled → eeg_pooled_proj (B, 2048)
- CLS Projector: eeg_cls → eeg_cls_proj (B, 1152)
- Show dashed teacher arrows from fMRI encoder:
  - fmri_spatial (B, 226, 2048) → L_distill_spatial (MSE)
  - fmri_cls (B, 1152) → L_distill_cls (MSE)
- Label: "P0: EEG learns fMRI feature space via distillation"

[MIDDLE-RIGHT] P1: Temporal Dynamics Decoder
- Show the decoder architecture clearly:
  - T+1 learnable queries: 9 temporal (t1...t9) + 1 global (g)
  - Each query represented as a small colored square
  - 4 Decoder Layers, each containing:
    - Cross-Attention: queries attend to EEG spatial tokens (226)
    - Self-Attention: queries attend to each other (with optional causal mask shown as a triangular pattern)
    - Feed-Forward Network
  - Input projection: 2048 → 512 (d_model)
  - Output projection: 512 → 1152 (head_dim)
- Outputs (show branching arrows from decoder output):
  - temporal_tokens (B, 9, 1152): show as 9 colored squares in a row, labeled "per-frame temporal embeddings"
  - global_dyn_token (B, 1152): show as single square, labeled "global dynamics summary"
  - Flow Trajectory Head: global_dyn_token → flow_traj_pred (B, 9), labeled "predicted motion magnitude per frame"
  - Coarse Dynamics Head: global_dyn_token → dyn_logits (B, 2), labeled "dynamic vs static"

[BOTTOM] Supervision Targets
- Show four offline targets with upward arrows:
  - SigLIP frame embeddings (Δ between consecutive frames) → L_temporal_delta
  - SigLIP frame embeddings (absolute positions) → L_temporal_abs
  - Optical flow magnitude trajectory → L_flow_traj
  - Dynamic/static binary label → L_dyn (CrossEntropy)

Visual design:
- Orange as primary color for all Fast Branch components
- Blue dashed arrows for fMRI teacher signal (P0)
- The causal mask in self-attention shown as a small triangular matrix icon
- Temporal queries shown as a sequence of small boxes (t1, t2, ..., t9, g)
- Clear separation between P0 (left) and P1 (right) sub-modules
- Dimension annotations in gray: (B, 226, 2048), (B, 512), (B, 9, 1152), etc.
```

---

## Figure 4: Gated Fusion + Multi-Guidance 详图

```
Create a detailed diagram of the Cross-Modal Gated Fusion and Multi-Guidance Adapter modules. Purple/violet for Fusion, teal for Guidance.

[LEFT] Inputs
- slow_feat (B, 226, 2048) — blue arrow from top
- fast_feat (B, 226, 2048) — orange arrow from bottom

[CENTER-LEFT] Cross-Modal Gated Fusion
- Step 1: Concat → (B, 226, 4096) → Linear → (B, 226, 2048) hidden
- Step 2: Add modality embeddings:
  - h_slow = h + embed_slow (copy with blue marker)
  - h_fast = h + embed_fast (copy with orange marker)
  - Concat → (B, 452, 2048)
- Step 3: 4-layer Fusion Transformer (show as stacked blocks)
  - All 452 tokens can attend to each other (full attention)
- Step 4: Split output:
  - First 226 tokens → Output Proj → z_b (B, 226, 4096) — main output
  - Mean pool all 452 tokens → Gating Network:
    - Linear(2048) → GELU → Linear(4) → Sigmoid
    - Outputs: α_key, α_txt, α_mot, α_brain (each ∈ [0,1])

[CENTER-RIGHT] Multi-Guidance Adapter
- Show the additive combination as a clear diagram:
  - z_b (main latent, thick arrow)
  - + α_key × KeyProj(z_key) — blue thin arrow, z_key from Slow Branch
  - + α_txt × TxtProj(z_txt) — blue thin arrow, z_txt from Slow Branch
  - + α_mot × MotProj(eeg_pooled) — orange thin arrow, from Fast Branch
  - + α_brain × z_b — purple thin arrow (self-residual)
  - → LayerNorm + Linear → context (B, 226, 4096)
- Show each alpha as a small multiplier icon on its respective guidance arrow

[RIGHT] Output
- context (B, 226, 4096) → feeds into DiT cross-attention layers

Visual design:
- Purple region for Fusion, teal region for Guidance
- The alpha weights shown as small dials or percentage bars
- Arrows color-coded: blue for slow-origin, orange for fast-origin, purple for fused
- Clear dimension annotations at each stage
```

---

## 使用说明

- 以上提示词可直接输入 DALL-E 3、Midjourney 或其他 AI 绘图工具
- 如需更精确的控制，建议使用 draw.io / Figma / PowerPoint 手绘，以上提示词作为布局参考
- Figure 1 (Model Overview) 适合论文主图（Figure 1 或 Figure 2 位置）
- Figure 2 (Training Stages) 适合方法章节的补充图
- Figure 3 (Fast Branch Detail) 和 Figure 4 (Fusion Detail) 适合补充材料或方法章节的详细说明
- 配色方案统一：blue=fMRI/Slow, orange=EEG/Fast, purple=Fusion, teal=Guidance, green=Decoder

### 与初版提示词的主要差异

| 组件 | 初版设计 | 当前实现 |
|------|---------|---------|
| Fast Branch | dynamics/motion/TC/direction 四个分类头 | P0 蒸馏 + P1 TemporalDynamicsDecoder |
| Fast Branch 输出 | dynamic pattern, motion latent, TC token | temporal_tokens, global_dyn_token, flow_traj_pred, dyn_logits |
| Slow Branch | keyframe + scene-text + structure 三个头 | keyframe + scene-text（structure 头未部署） |
| Fusion | 简述为 "gated fusion" | 详细展示 concat→transformer→gating 架构 |
| Guidance | 四路 guidance 简述 | 展示 alpha-weighted additive combination |
| Decoder | 泛化的 video diffusion decoder | 明确为 CogVideoX-5B DiT + LoRA |
| 训练 | 未涉及 | 新增 Figure 2 三阶段训练流程 |
| 蒸馏关系 | 未涉及 | 新增 fMRI→EEG 的 teacher-student 虚线 |
