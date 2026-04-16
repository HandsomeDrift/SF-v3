# Results Table Templates

## Main comparison
| Method | Modalities | Design | Video 2-way ↑ | Video 50-way ↑ | Frame 2-way ↑ | Frame 50-way ↑ | SSIM ↑ | PSNR ↑ | FVD ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CineSync-EEG | EEG | unified latent |  |  |  |  |  |  |  |
| CineSync-fMRI | fMRI | unified latent |  |  |  |  |  |  |  |
| CineSync | fMRI+EEG | unified latent |  |  |  |  |  |  |  |
| CineBrain-SF v1 | fMRI+EEG | slow-fast explicit |  |  |  |  |  |  |  |

## Branch ablation
| Variant | Slow | Fast | Multi-guidance | Video 2-way ↑ | Video 50-way ↑ | SSIM ↑ | FVD ↓ |
|---|---|---|---|---:|---:|---:|---:|
| Unified latent baseline | ✗ | ✗ | ✗ |  |  |  |  |
| Slow only | ✓ | ✗ | partial |  |  |  |  |
| Fast only | ✗ | ✓ | partial |  |  |  |  |
| Full (latent only) | ✓ | ✓ | ✗ |  |  |  |  |
| Full (multi-guidance) | ✓ | ✓ | ✓ |  |  |  |  |

## Slow branch ablation
| Variant | Keyframe | Scene-text | Structure | Video 2-way ↑ | SSIM ↑ |
|---|---|---|---|---:|---:|
| Full | ✓ | ✓ | ✓ |  |  |
| w/o keyframe | ✗ | ✓ | ✓ |  |  |
| w/o scene-text | ✓ | ✗ | ✓ |  |  |
| w/o structure | ✓ | ✓ | ✗ |  |  |

## Fast branch ablation
| Variant | Dynamics | Motion | Temp Coherence | Video 2-way ↑ | Motion Metric ↑/↓ | SSIM ↑ |
|---|---|---|---|---:|---:|---:|
| Full | ✓ | ✓ | ✓ |  |  |  |
| w/o dynamics | ✗ | ✓ | ✓ |  |  |  |
| w/o motion | ✓ | ✗ | ✓ |  |  |  |
| w/o temp coherence | ✓ | ✓ | ✗ |  |  |  |
| latent motion | ✓ | latent | ✓ |  |  |  |
| flow token | ✓ | flow | ✓ |  |  |  |

## Auditory study
| Variant | Visual ROI | Auditory ROI | Audio-aware context | Video 2-way ↑ | Video 50-way ↑ | SSIM ↑ |
|---|---|---|---|---:|---:|---:|
| Visual only | ✓ | ✗ | ✗ |  |  |  |
| Visual + auditory | ✓ | ✓ | ✗ |  |  |  |
| Visual + auditory + context | ✓ | ✓ | ✓ |  |  |  |

## Per-subject results
| Subject | Video 2-way ↑ | Video 50-way ↑ | Frame 2-way ↑ | SSIM ↑ | FVD ↓ |
|---|---:|---:|---:|---:|---:|
| S1 |  |  |  |  |  |
| S2 |  |  |  |  |  |
| S3 |  |  |  |  |  |
| S4 |  |  |  |  |  |
| S5 |  |  |  |  |  |
| S6 |  |  |  |  |  |
| Mean ± Std |  |  |  |  |  |