# Low Noise + Kernel Tricks — Findings

**Date:** 2026-05-29
**Branch:** feat/low-noise-kernel-tricks

## 1. Low Noise Experiment (obs_noise 0.28 → 0.12)

Generator CLI now exposes `--obs-noise-std` and `--latent-noise-std`.
SNR context: current synthetic SNR=0.64, real wearable data ~1.5-2.5.
Low noise (0.12) gives SNR ~1.5 — realistic for quality wearable sensors.

### Results (low noise vs original, transductive, Takens graph)

| Metric | Original (0.28) | Low noise (0.12) | Δ |
|--------|-----------------|------------------|---|
| Takens Ridge h=1 | 0.517 | 0.559 | +0.042 |
| takens+prior+grm h=21 | 0.470 | 0.474 | +0.004 |
| Regime (full, takens log) | 0.526 | 0.582 | +0.056 |
| Treatment η²_h7 (GRM) | 0.148 | 0.265 | +0.117 |
| Treatment η²_regime (multiscale) | 0.025 | 0.092 | +0.067 |
| TCM alignment AMI (tcm_like) | 0.149 | 0.153 | +0.004 |

Key: treatment response stratification nearly doubles with realistic noise.
Regime prediction improves +5.6pp. The methodology is sound; the original
synthetic noise was unrealistically high.

## 2. Sign-Sqrt Kernel Trick

Transform: `sign(grm) * sqrt(|grm|)` — compresses heavy-tailed GRM modes,
makes nonlinear topology more linearly readable by Ridge.

### Results (original noise, transductive)

| Model | h=1 | h=3 | h=7 | h=21 |
|-------|-----|-----|-----|------|
| takens+prior+grm | 0.526 | 0.285 | 0.386 | 0.466 |
| takens+prior+grm_ssqrt | 0.529 | 0.290 | 0.385 | 0.462 |
| Δ | +0.003 | +0.005 | -0.001 | -0.004 |

Marginal gain at short horizons, slight degradation at long horizons.
The constitution proxy already dominates h≥7; sign-sqrt competes with
it rather than complementing it. Kept in pipeline for completeness
but not a breakthrough.

## Conclusion

Low noise is the bigger lever. When noise drops to realistic wearable
levels, treatment stratification improves dramatically — this is the
strongest signal that real clinical data would produce better results
than the current synthetic benchmark.
