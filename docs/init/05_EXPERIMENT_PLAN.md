# Experiment Plan — CineBrain-SF v1

## 1. Main comparison
目标问题：
显式 slow-fast role assignment 是否优于 unified multimodal latent fusion？

### Baselines
- CineSync-EEG
- CineSync-fMRI
- CineSync
- CineBrain-SF v1 (ours)

## 2. Core ablations
### A. Unified latent vs explicit branches
- unified latent
- slow only
- fast only
- slow + fast without multi-guidance
- slow + fast + multi-guidance

### B. Slow branch ablation
- w/o keyframe
- w/o scene-text
- w/o structure

### C. Fast branch ablation
- w/o dynamics
- w/o motion
- w/o temporal coherence
- motion latent version
- flow token version

### D. Fusion ablation
- CineSync dual transformer late fusion
- adaptive gated fusion
- fixed-weight fusion

### E. Auditory contribution
- visual ROI only
- visual + auditory ROI
- visual + auditory + audio-aware context

## 3. Optional extension experiments
### Cross-subject pilot
- no adaptation
- + lightweight subject token
- + IAM-style adaptation (light version)
- + GLFA-style fMRI adapter (light version)

### Audio-sensitive subset analysis
对对白密集、说话镜头、人物交互镜头单独评测：
- 是否 auditory ROI 更有帮助
- 是否 EEG / fMRI 的贡献比例发生变化

## 4. Metrics
### Main video metrics
- Video 2-way / 50-way
- Frame 2-way / 50-way
- SSIM
- PSNR
- FVD
- CTC / DTC / CLIP-pcc（按仓库现有实现）

### Intermediate metrics
- Keyframe CLIP similarity / SSIM
- Scene-text similarity
- Structure latent regression error
- Dynamics accuracy
- Motion latent retrieval / flow token acc / EPE（若实现）
- Temporal coherence correlation

## 5. Reporting rules
- 每个实验报告 mean ± std（按 subject 或 seed）
- 对关键指标报告 relative improvement
- 所有 ablation 必须共享相同训练 budget，除非明确说明