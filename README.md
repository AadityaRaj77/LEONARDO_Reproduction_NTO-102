# LEONARDO Reproduction and Position-Aware Ablation

Reproduction and experimental analysis of the **LEONARDO framework** for learning nanoparticle diffusion dynamics from Liquid-Phase Transmission Electron Microscopy (LPTEM) trajectories.

This project reproduces the baseline Transformer-VAE framework and investigates the effect of adding a position-aware reconstruction objective.

## 1. Project Overview

The objective is to learn statistical and temporal characteristics of nanoparticle trajectories using a Variational Autoencoder (VAE) with Transformer-based components.

The work consists of two experimental stages:

1. **Baseline reproduction:** Train and evaluate the original LEONARDO implementation.
2. **Position-aware ablation:** Introduce an additional absolute-coordinate reconstruction loss while retaining the original loss components.

## 2. Dataset

The experiments use LPTEM trajectory data.

Dataset files:
- `lptem_train.npy`
- `lptem_val.npy`
- `lptem_test.npy`

Trajectory representation:
- Sequence length: 200
- Dimensions: 2 (X, Y coordinates)

Dataset shapes:
- Training: `(38279, 200, 2)`
- Validation: `(3202, 200, 2)`
- Test: `(5934, 200, 2)`

Coordinates are scaled to the range [0, 1].

## 3. Model Architecture

The implementation uses a Transformer-based Variational Autoencoder.

**Encoder**
- 1D convolutional input projection
- Transformer encoder blocks
- Bottleneck encoding
- Latent mean and log-variance estimation
- Gaussian reparameterization

**Latent representation**
- Latent dimension: 12

**Decoder**
- Linear latent projection
- Bottleneck decoding
- Transformer decoder blocks
- 1D convolutional output projection

Output shape: `(batch_size, 200, 2)`.

## 4. Baseline Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Transformer layers | 2 |
| Attention heads | 8 |
| Forward expansion | 2 |
| Dropout | 0.0 |
| Latent dimension | 12 |
| Optimizer | Adam |
| Initial learning rate | 0.0003 |

The baseline checkpoint used for comparison is `baseline_epoch9.pt`.

## 5. Position-Aware Ablation

An additional absolute-coordinate reconstruction penalty was introduced:

\[
L_{\mathrm{modified}} =
L_{\mathrm{original}}+
\lambda_{\mathrm{pos}}L_{\mathrm{position}}
\]

where

\[
L_{\mathrm{position}} =
\frac{1}{N}\sum_i
(\hat{x}_i-x_i)^2
\]

and the position-loss coefficient was:

\[
\lambda_{\mathrm{pos}}=0.378
\]

The original statistical, autocorrelation, displacement and KL-divergence loss components were retained.

The ablation was initialized from the baseline epoch-9 checkpoint and trained with a learning rate of 0.00003.

The selected checkpoint is `best_position_ablation.pt` (epoch 14).

## 6. Final Held-Out Test Results

Both models were evaluated on the same held-out test set using deterministic inference: encoder mean followed by decoder.

| Metric | Baseline | Position Ablation |
|---|---:|---:|
| Coordinate MSE | 0.73043 | **0.08631** |
| Displacement MSE | 0.01438 | **0.01262** |
| Reconstructed X displacement std | 0.09476 | 0.08708 |
| Reconstructed Y displacement std | 0.09638 | 0.08473 |
| Reconstructed X kurtosis | 6.8564 | 6.6420 |
| Reconstructed Y kurtosis | 7.2122 | 6.8755 |

The position-aware ablation achieved:
- 88.18% reduction in coordinate MSE.
- 12.24% reduction in displacement MSE.

These results indicate substantially improved reconstruction accuracy. However, some statistical differences from real trajectories remain.

## 7. Evaluation and Visualizations

The final evaluation includes:

1. Real versus reconstructed trajectories.
2. Displacement distribution comparisons.
3. Displacement autocorrelation comparisons.
4. Coordinate distribution comparisons.

The corresponding figures are stored in `final_results/`.

Additional diagnostics investigated:
- Trajectory-level displacement variance
- Trajectory-level kurtosis
- Prior versus posterior generation
- Pearson and Spearman correlations of trajectory-level variance

## 8. Repository Structure

```text
LEONARDO/
├── LEONARDO_model.py
├── checkpoints/
│   └── baseline_epoch9.pt
├── position_ablation/
│   └── best_position_ablation.pt
└── final_results/
    ├── final_test_evaluation.npz
    ├── baseline_vs_ablation_test_results.csv
    ├── figure1_trajectories.png
    ├── figure2_displacement.png
    ├── figure3_autocorrelation.png
    └── figure4_coordinates.png
```

The dataset and complete training/evaluation notebooks should be added or linked separately according to their availability and redistribution permissions.

## 9. Limitations

Despite improvements in reconstruction accuracy, the model does not perfectly reproduce all physical characteristics of the real trajectories.

Observed limitations include:
- Generated displacement variability remains higher than real data.
- Spatial coordinate spread is underestimated.
- Lag-1 autocorrelation is not perfectly matched.
- Trajectory-level variance distributions show compression.

Therefore, the position-aware objective improves reconstruction but does not establish complete reproduction of nanoparticle diffusion dynamics.

## 10. Future Work

Potential directions include:
- Improved trajectory-level variance calibration.
- Better preservation of temporal correlations.
- Investigation of latent representations and physical constraints.
- Evaluation across additional experimental trajectory datasets.

## 11. Acknowledgements

This work is based on the LEONARDO framework and its associated research methodology.

**Research focus:** Transformer-based generative modelling, nanoparticle diffusion dynamics, LPTEM trajectories and statistical reconstruction.
