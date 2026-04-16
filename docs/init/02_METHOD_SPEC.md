# Method Spec — CineBrain-SF v1

## 1. Inputs

### 1.1 fMRI input
- visual ROI signal
- auditory ROI signal
- 使用现有 CineBrain 的 lag-corrected 对齐方式
- 建议保持 visual / auditory 两路 token，先分别编码

### 1.2 EEG input
- 4-second synchronized EEG segment
- overlapping short windows
- 可选附加 spectral / band features
- 优先复用现有 EEG preprocessing 与 tensor format

### 1.3 Training-only supervision
训练时允许从 stimulus 侧构建：
- video latent
- keyframe
- text / scene description embedding
- structure latent
- motion representation（motion token / flow token / dynamic pattern）

---

## 2. Model Overview

### 2.1 Slow Semantic-Structure Branch (S-Branch)
**输入**：fMRI（visual ROI + auditory ROI）
**输出**：
- `z_key`：keyframe latent / keyframe image prior
- `z_txt`：scene-text semantic embedding
- `z_str`：structure latent

**建议实现**
- `fMRIVisualEncoder`
- `fMRIAuditoryEncoder`
- `AudiovisualContextAdapter`
- `KeyframeHead`
- `SceneTextHead`
- `StructureHead`

### 2.2 Fast Motion-Dynamics Branch (F-Branch)
**输入**：EEG segment
**输出**：
- `z_dyn`：dynamic pattern embedding / fast-slow signal
- `z_mot`：motion latent or motion token
- `z_tc`：temporal coherence token

**建议实现**
- `EEGSpatialEncoder`
- `EEGTemporalEncoder`
- `EEGFusionBlock`
- `DynamicsHead`
- `MotionHead`
- `TemporalCoherenceHead`

### 2.3 Cross-Modal Gated Fusion (CMGF)
**输入**：
- slow branch representation
- fast branch representation

**输出**：
- `alpha_key`
- `alpha_txt`
- `alpha_mot`
- `alpha_brain`
- `z_b`（fused brain latent）

功能：
- 对 slow / fast 表示做高层融合
- 学习不同 guidance 的权重
- 输出统一 latent 给 decoder

### 2.4 Multi-Guidance Neuro-Latent Decoder (MG-NLD)
以现有 CogVideoX / NLD 为底座，扩展为多 guidance 输入：

**guidance set**
- `g_key = alpha_key * z_key`
- `g_txt = alpha_txt * z_txt`
- `g_mot = alpha_mot * [z_dyn, z_mot, z_tc]`
- `g_brain = alpha_brain * z_b`

**decoder target**
- reconstruct video latent / video clip

---

## 3. Mathematical Objectives

### 3.1 Alignment losses
- `L_fv`：slow ↔ video semantic
- `L_ft`：slow ↔ text semantic
- `L_ev`：fast ↔ video semantic
- `L_et`：fast ↔ text semantic
- `L_fe`：slow ↔ fast alignment

总对齐损失：
`L_align = λ_fv L_fv + λ_ft L_ft + λ_ev L_ev + λ_et L_et + λ_fe L_fe`

### 3.2 Slow-branch losses
- `L_key`
- `L_txt`
- `L_str`

`L_slow = λ_key L_key + λ_txt L_txt + λ_str L_str`

### 3.3 Fast-branch losses
- `L_dyn`
- `L_mot`
- `L_tc`

`L_fast = λ_dyn L_dyn + λ_mot L_mot + λ_tc L_tc`

### 3.4 Decoder loss
标准 diffusion denoising:
`L_diff = E[ || eps - eps_theta(x_t, conds, t) ||^2 ]`

### 3.5 Guidance consistency losses
- `L_gk`：generated video ↔ keyframe consistency
- `L_gt`：generated video ↔ text consistency
- `L_gm`：generated video ↔ motion consistency

`L_guide = λ_gk L_gk + λ_gt L_gt + λ_gm L_gm`

### 3.6 Optional auditory-context loss
- `L_aud`

### 3.7 Total loss
`L_total = L_diff + L_align + L_slow + L_fast + L_guide + λ_aud L_aud`

---

## 4. Training Stages

### Stage I — Branch pretraining
训练 S-Branch 和 F-Branch，使显式中间变量先稳定：
`L_stage1 = L_align + L_slow + L_fast`

### Stage II — Fusion training
在不完全端到端的情况下训练 gating / fusion：
`L_stage2 = L_align + L_slow + L_fast + L_guide`

### Stage III — Joint decoding
接入 MG-NLD：
`L_stage3 = L_total`

---

## 5. Minimal v1 Scope
v1 最小强版本建议只实现：
- keyframe head
- scene-text embedding head（不必先做完整 caption decoder）
- structure latent head
- dynamic head
- motion latent head（flow token 可做 optional）
- temporal coherence head
- gated fusion
- multi-guidance decoder