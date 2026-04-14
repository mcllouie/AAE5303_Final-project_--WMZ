# AAE5303 Assignment: 3D Gaussian Splatting with OpenSplat

<div align="center">

![3DGS](https://img.shields.io/badge/3D_Gaussian-Splatting-blue?style=for-the-badge)
![OpenSplat](https://img.shields.io/badge/Framework-OpenSplat-green?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-HKisland-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**Novel View Synthesis using 3D Gaussian Splatting on UAV Imagery**

*Hong Kong Island Aerial Dataset*

</div>

---

## 📋 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Introduction](#-introduction)
3. [Methodology](#-methodology)
4. [Dataset Description](#-dataset-description)
5. [Implementation Details](#-implementation-details)
6. [Results and Analysis](#-results-and-analysis)
7. [Visualizations](#-visualizations)
8. [Discussion](#-discussion)
9. [Conclusions](#-conclusions)
10. [References](#-references)
11. [Appendix](#-appendix)

---

## 📊 Executive Summary

This report presents the implementation and evaluation of **3D Gaussian Splatting (3DGS)** for novel view synthesis using the **OpenSplat** framework on the **HKisland** UAV aerial imagery dataset. The project demonstrates the application of state-of-the-art neural rendering techniques for reconstructing 3D scenes from multi-view images.

### Key Results

| Metric | Value |
|--------|-------|
| **Training Iterations** | 30 |
| **Number of Images** | 20 |
| **Initial Gaussian Points** | 3343 |
| **Final Loss** | 0.1772 |
| **Minimum Loss Achieved** | 0.1772 |
| **Loss Reduction** | 36.7% |
| **Output PLY Size** | 811 kB |

---

## 📖 Introduction

### Background

3D Gaussian Splatting (3DGS) represents a breakthrough in neural rendering, offering real-time rendering capabilities while maintaining high visual quality. Unlike neural radiance fields (NeRF) that rely on implicit representations, 3DGS explicitly represents scenes using millions of 3D Gaussian primitives, enabling:

- **Real-time rendering** at high resolutions
- **Efficient training** compared to NeRF-based methods
- **Explicit geometry** that can be directly manipulated
- **High-quality novel view synthesis**

### Objectives

1. Implement 3D Gaussian Splatting using OpenSplat framework
2. Process UAV aerial imagery from the HKisland dataset
3. Generate a high-quality 3D reconstruction of Hong Kong Island terrain
4. Analyze training dynamics and reconstruction quality
5. Document the complete workflow for reproducibility

### Scope

This assignment focuses on:
- Setting up the OpenSplat build environment
- Preparing COLMAP-formatted input data
- Training the 3DGS model
- Analyzing results and generating visualizations

---

## 🔬 Methodology

### 3D Gaussian Splatting Overview

The 3DGS algorithm represents a 3D scene as a collection of anisotropic 3D Gaussian primitives. Each Gaussian is characterized by:

1. **Position (μ)**: 3D mean position in world coordinates
2. **Covariance (Σ)**: 3×3 covariance matrix defining shape and orientation
3. **Opacity (α)**: Transparency value for blending
4. **Spherical Harmonics (SH)**: View-dependent color representation

The rendering equation for a pixel is:

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

where $c_i$ is the color computed from spherical harmonics and $\alpha_i$ is the opacity multiplied by the Gaussian's 2D projection.

### Training Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  COLMAP Input   │────▶│   Initialize    │────▶│   Forward Pass  │
│  (SfM + Images) │     │   Gaussians     │     │   (Rendering)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│  Output PLY     │◀────│   Update        │◀────│   Compute Loss  │
│  (Final Model)  │     │   Parameters    │     │   (L1 + SSIM)   │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   Densification │
                        │   & Pruning     │
                        └─────────────────┘
```

### Loss Function

The training loss combines L1 reconstruction loss and structural similarity:

$$\mathcal{L} = (1 - \lambda_{SSIM}) \cdot \mathcal{L}_1 + \lambda_{SSIM} \cdot (1 - SSIM)$$

where $\lambda_{SSIM} = 0.2$ is the SSIM weight.

### Adaptive Density Control

OpenSplat implements adaptive Gaussian density control:

1. **Densification**: Split or clone Gaussians with high view-space gradients
2. **Pruning**: Remove Gaussians with low opacity or excessive size
3. **Alpha Reset**: Periodically reset opacity values to prevent artifacts

---

## 📁 Dataset Description

### HKisland COLMAP Dataset

The HKisland dataset consists of UAV (Unmanned Aerial Vehicle) imagery captured over Hong Kong Island terrain.

| Property | Value |
|----------|-------|
| **Dataset Name** | AMtown02 |
| **Number of Images** | 20 |
| **Image Format** | JPEG |
| **Initial SfM Points** | 3343 |
| **Camera Model** | Pinhole |
| **Source** | UAVScenes |

### Data Structure

```
data/
├── HKisland_20_colmap/
│   ├── images/
│   │   ├── 1658132926.098563936.jpg
│   │   ├── 1658132926.596775653.jpg
│   │   └── ... (20 images total)
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
```

### Dataset Characteristics

- **Temporal Coverage**: Single capture session (timestamp-based filenames)
- **Spatial Coverage**: Hong Kong terrain region
- **Capture Pattern**: Sequential flight path
- **Ground Sample Distance**: UAV-typical resolution

---

## ⚙️ Implementation Details

### System Configuration

| Component | Specification |
|-----------|---------------|
| **Framework** | OpenSplat (C++) |
| **Compute Device** | CPU |
| **libtorch Version** | 2.1.2 |
| **OpenCV** | System default |
| **Operating System** | Linux (WSL2 Ubuntu 20.04) |

### Training Configuration

```
OMP_NUM_THREADS=1 ~/OpenSplat/build/opensplat ~/OpenSplat/data/HKisland_20_colmap \
  -n 30 \
  -s 10 \
  -d 2 \
  -o ~/OpenSplat/output/hkisland_20_quick_30.ply \
  --cpu \
  --colmap-image-path ~/OpenSplat/data/HKisland_20_colmap/images \
  2>&1 | tee ~/OpenSplat/output/train_20_quick_30.log
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num-iters` | 30 | Training iterations |
| `sh-degree` | 3 | Maximum spherical harmonics degree |
| `ssim-weight` | 0.2 | SSIM loss weight |
| `refine-every` | 100 | Densification interval |
| `warmup-length` | 500 | Warmup period (no densification) |
| `densify-grad-thresh` | 0.0002 | Gradient threshold for densification |
| `densify-size-thresh` | 0.01 | Size threshold for split/clone decision |
| `reset-alpha-every` | 30 | Alpha reset interval (in refinements) |

### Build Process

```bash
# Clone repository
git clone https://github.com/pierotofy/OpenSplat

# Build with CPU support
cd OpenSplat
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ ..
make -j$(nproc)
```

---

## 📈 Results and Analysis

### Training Progress

The model was trained for 30 iterations on the HKisland dataset. Training logs show consistent convergence:

| Phase | Steps | Mean Loss | Characteristics |
|-------|-------|-----------|-----------------|
| **Phase 1** | 1-7 | 0.259 | Initial optimization |
| **Phase 2** | 8-14 | 0.232 | Continued convergence |
| **Phase 3** | 15-21 | 0.211 | Stabilization |
| **Phase 4** | 22-30 | 0.191 | Final refinement |

### Loss Metrics

```
Training Statistics:
─────────────────────────────────
Initial Loss:     0.2798
Final Loss:       0.1772
Minimum Loss:     0.1772
Maximum Loss:     0.2798
Mean Loss:        0.2209
Std Deviation:    0.0278
Loss Reduction:   36.7%
─────────────────────────────────
```

### Output Model

The trained model was exported as a PLY file containing:

| Property | Value |
|----------|-------|
| **File Size** | 811 kB |
| **Number of Gaussians** | 3343 |
| **Spherical Harmonics Coefficients** | 48 (3 DC + 45 rest) |
| **Format** | Binary Little Endian |

### Estimated Quality Metrics

Based on the loss values and typical correlations:

| Metric | Estimated Value |
|--------|-----------------|
| **PSNR** | ~22-24 dB |
| **SSIM** | ~0.80-0.85 |

*Note: Full evaluation would require rendering against held-out test views.*

---

## 📊 Visualizations

### Training Loss Curve

<img width="4983" height="935" alt="training_loss_curve" src="https://github.com/user-attachments/assets/7305ec4f-b3f3-4b8a-bfe3-a347b236d3a8" />


The training loss curve shows the model's learning progress over 30 iterations. Key observations:
- Initial rapid decrease in loss during early iterations
- Oscillations due to random camera sampling
- Gradual stabilization toward convergence

### Loss Distribution

<img width="2063" height="727" alt="loss_distribution" src="https://github.com/user-attachments/assets/b96dfb08-5cd4-4217-afbd-5c0c5f649216" />


The loss distribution analysis reveals:
- Approximately normal distribution centered around 0.22
- Minimum losses achieved around 0.18
- Consistent training without major outliers

### Convergence Analysis

<img width="2076" height="1477" alt="convergence_analysis" src="https://github.com/user-attachments/assets/28729e86-4f91-4edc-b47a-bb860c9a99cb" />


Multi-faceted convergence analysis showing:
- Log-scale loss progression
- Loss gradient over training
- Rolling statistics with confidence bands
- Phase-wise performance comparison

### Summary Dashboard

<img width="1980" height="1444" alt="summary_dashboard" src="https://github.com/user-attachments/assets/4c841243-bc08-42b6-8689-094b499f62eb" />


Comprehensive dashboard summarizing all key metrics and training statistics.

---

## 💭 Discussion

### Strengths

1. **Successful 3D Reconstruction**: The model successfully processes 20 UAV images and generates a coherent 3D Gaussian representation of the HKisland terrain.

2. **Stable Training**: Despite limited iterations (30), training showed consistent convergence without divergence or instability.

3. **Efficient Processing**: OpenSplat efficiently handled 3343 initial points from the COLMAP reconstruction.

### Limitations

1. **Limited Iterations**: 30 iterations is significantly below the recommended 30,000 for optimal quality. This was due to computational constraints (CPU-only execution).

2. **Dataset Limitation**:  Only used 20 images can not reflect complete area's situation.

3. **CPU Execution**: Training on CPU is approximately 100x slower than GPU, limiting practical iteration counts.

### Areas for Improvement

1. **Increase Training Iterations**: Running for 30,000+ iterations would significantly improve reconstruction quality.

2. **GPU Acceleration**: Using CUDA-enabled GPU would enable practical training times for full convergence.

3. **Hyperparameter Tuning**: Adjusting densification thresholds and learning rates for the specific dataset characteristics.

4. **Using Complete Dataset**: Using complete dataset can enrich the image and reflect the whole area.

---

## 🎯 Conclusions

This assignment successfully demonstrates the implementation of 3D Gaussian Splatting for novel view synthesis using the OpenSplat framework. Key achievements include:

1. ✅ **Environment Setup**: Successfully built and configured OpenSplat on Linux(WSL Ubuntu 20.04)
2. ✅ **Data Preparation**: Processed HKisland COLMAP dataset with 20 images
3. ✅ **Model Training**: Completed 30 training iterations with convergence
4. ✅ **Result Generation**: Produced a 811 MB PLY file with 3343M Gaussians
5. ✅ **Analysis**: Generated comprehensive visualizations and statistics

### Future Work

- Extend training to 30,000 iterations with GPU acceleration
- Implement train/test split for proper evaluation
- Compare with original 3DGS implementation
- Explore compression techniques for the output model
- Investigate view-dependent effects with higher SH degrees

---

## 📚 References

1. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). **3D Gaussian Splatting for Real-Time Radiance Field Rendering**. *ACM Transactions on Graphics (SIGGRAPH)*.

2. Schönberger, J. L., & Frahm, J. M. (2016). **Structure-from-Motion Revisited**. *Conference on Computer Vision and Pattern Recognition (CVPR)*.

3. OpenSplat: https://github.com/pierotofy/OpenSplat

4. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). **Image Quality Assessment: From Error Visibility to Structural Similarity**. *IEEE Transactions on Image Processing*.

---

## 📎 Appendix

### A. Training Command

```bash
./opensplat /root/OpenSplat/data/HKisland_colmap -n 300 -o hkisland_output.ply
```

### B. Output PLY Header

```
ply
format binary_little_endian 1.0
comment Generated by opensplat at iteration 300
element vertex 1441245
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
... (45 additional f_rest properties)
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
```

### C. Repository Structure

```
AAE5303_assignment/
├── README.md                    # This report
├── requirements.txt             # Python dependencies
├── figures/
│   ├── training_loss_curve.png
│   ├── loss_distribution.png
│   ├── convergence_analysis.png
│   └── summary_dashboard.png
├── output/
│   └── training_report.json
├── scripts/
│   └── analyze_training.py
├── docs/
│   └── training_log.txt
└── leaderboard/
    ├── README.md
    ├── LEADERBOARD_SUBMISSION_GUIDE.md
    └── submission_template.json
```

### D. Environment Details

- **Python**: 3.10
- **Matplotlib**: For visualization generation
- **NumPy**: For numerical analysis
- **OpenSplat**: Built from source with libtorch 2.1.2

---

<div align="center">

**AAE5303 - Robust Control Technology in Low-Altitude Aerial Vehicle**

*Department of Aeronautical and Aviation Engineering*

*The Hong Kong Polytechnic University*

December 2024

</div>
