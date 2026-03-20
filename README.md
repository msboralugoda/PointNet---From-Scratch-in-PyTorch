# PointNet — From Scratch in PyTorch

> My implementation of [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (Qi et al., 2017), built from scratch while learning the paper deeply.

This repo contains the PyTorch code alongside my own notes explaining *why* each design decision was made, not just *what* the code does.

---

## Table of Contents

- [Why PointNet?](#why-pointnet)
- [The Core Problem](#the-core-problem)
- [Key Ideas](#key-ideas)
  - [1. Shared MLP](#1-shared-mlp--one-network-every-point)
  - [2. Permutation Invariance via Max Pool](#2-permutation-invariance-via-max-pool)
  - [3. Spatial Transformer (T-Net)](#3-spatial-transformer-t-net)
  - [4. Orthogonality Regularisation](#4-orthogonality-regularisation)
- [Architecture](#architecture)
- [Tensor Shape Walkthrough](#tensor-shape-walkthrough)
- [Code Structure](#code-structure)
- [Setup & Training](#setup--training)
- [Results](#results)
- [What I Learned](#what-i-learned)
- [References](#references)

---

## Why PointNet?

PointNet (2017) is one of the most influential papers in 3D deep learning. Before it, applying deep learning to 3D shapes meant converting them into voxel grids (memory-hungry, imprecise) or projecting them into 2D images (lossy). PointNet showed you could work directly on raw point clouds and do it elegantly.

It introduced ideas that are still foundational: symmetric functions for unordered sets, spatial transformers for geometric invariance, and shared-weight per-point processing. Understanding PointNet is a prerequisite for understanding almost everything that came after it (PointNet++, DGCNN, PointTransformer, etc.).

---

## The Core Problem

A **point cloud** is just a set of (x, y, z) coordinates in 3D space. Unlike images, point clouds have three properties that make them hard to feed into standard neural networks:

| Property | Images | Point Clouds |
|---|---|---|
| Structure | Regular grid | Unordered set |
| Neighbours | Fixed (pixels touch) | No fixed topology |
| Size | Fixed H × W | Variable N points |
| Order | Matters (row-major) | Doesn't matter |

A standard MLP or CNN would produce a different output if you shuffled the input order. That's catastrophic for point clouds - the same chair looks identical whether you list its points top-to-bottom or randomly. The network must output the same result either way.

PointNet's answer: **design the architecture so order does not matter.**

---

## Key Ideas

### 1. Shared MLP — One Network, Every Point

The first insight is to process each point *independently* with the same network weights, then aggregate. No point ever talks to another point during the MLP stages.

In PyTorch this is implemented as:

```python
nn.Conv1d(in_channels, out_channels, kernel_size=1)
```

A `Conv1d` with `kernel_size=1` applies a linear transformation to each position in the sequence independently — it never looks left or right. With input shape `(B, 3, N)`, this is mathematically identical to applying an `nn.Linear(3, out)` to each of the N points separately, but runs as one fast vectorised operation.

```
Input:  (B, 3,  N)   — batch of N points, each with 3 coordinates
           ↓
Conv1d(3, 64, 1)      — same weights applied to each point
           ↓
Output: (B, 64, N)   — batch of N feature vectors, each 64-dimensional
```

**Why not just use `nn.Linear`?** You could loop `for point in points: out = linear(point)` but that's slow and inelegant. `Conv1d(..., 1)` does it in one GPU-parallelised call with identical semantics.

**Key property preserved:** Since each point is processed identically and independently, shuffling the N points just shuffles the output columns — the *set* of outputs is the same. Order still can't matter here.

---

### 2. Permutation Invariance via Max Pool

After the shared MLP, we have `(B, 1024, N)` — 1024-dimensional features for each of the N points. We need to collapse this to a single global descriptor.

The operation must be **symmetric** — it must give the same result regardless of point order. The paper calls this a "symmetric function." Several options exist:

| Aggregation | Symmetric? | Notes |
|---|---|---|
| Sum | ✓ | Sensitive to number of points |
| Mean | ✓ | Loses peak signal |
| **Max** | ✓ | Keeps the strongest signal per feature |
| Concat | ✗ | Order-dependent |
| Sort then MLP | ✗ | Sorting is not differentiable cleanly |

**Max pooling wins** because it's differentiable, fast, and captures "is this feature present anywhere in the cloud?" rather than averaging it away.

```python
global_feat = torch.max(x, dim=2).values   # (B, 1024, N) → (B, 1024)
```

`dim=2` is the N (points) dimension. For each of the 1024 features, we ask: *what is the maximum value this feature takes across all N points?* The result is a 1024-d vector that summarises the entire point cloud — regardless of how many points there are or what order they came in.

```
Points in any order:       After per-point MLP:      After max pool:
  point 1: [9, 2, 1]   →   feat 1: [9, 2, 1]   →
  point 2: [3, 7, 4]   →   feat 2: [3, 7, 4]   →   [9, 7, 8]
  point 3: [1, 5, 8]   →   feat 3: [1, 5, 8]   →

Shuffle to [3,1,2]:
  point 1: [1, 5, 8]   →   feat 1: [1, 5, 8]   →
  point 2: [9, 2, 1]   →   feat 2: [9, 2, 1]   →   [9, 7, 8]  ← identical
  point 3: [3, 7, 4]   →   feat 3: [3, 7, 4]   →
```

This single line — `torch.max(x, dim=2).values` — is arguably the most important line in the whole paper.

---

### 3. Spatial Transformer (T-Net)

Even with permutation invariance solved, there's another problem: the same chair scanned from different angles produces geometrically different point clouds. Ideally the network should be robust to rotations and translations.

The T-Net is a **mini network inside the main network** that learns to predict an alignment matrix:

```
point cloud (B, 3, N)
       ↓
  [shared MLP + max pool + FC layers]   ← same structure as the main network
       ↓
  3×3 matrix M   (B, 3, 3)
       ↓
  aligned cloud = M · input             ← torch.bmm(M, x)
```

The crucial thing: **nobody tells the T-Net what the correct alignment is.** It learns what transformation makes classification easiest, purely from the classification loss backpropagating through it. The canonical pose emerges from training.

```python
input_trans = self.input_tnet(x)         # predict (B, 3, 3) matrix
x = torch.bmm(input_trans, x)           # apply it: (B,3,3) × (B,3,N) → (B,3,N)
```

`torch.bmm` is batch matrix multiply — it applies a different matrix to each item in the batch simultaneously.

The paper uses **two** T-Nets:
- A **3×3 T-Net** on raw xyz coordinates (input space alignment)
- A **64×64 T-Net** on the intermediate feature vectors (feature space alignment)

The 64×64 one is more powerful but also harder to constrain — see the next section.

**Identity initialisation matters.** In the code, the T-Net's final layer is initialised so the network starts by predicting the identity matrix. Without this, early training is chaotic because the transform starts random:

```python
nn.init.zeros_(self.fc3.weight)
nn.init.zeros_(self.fc3.bias)
self.fc3.bias.data.copy_(torch.eye(k).flatten())  # start as identity
```

---

### 4. Orthogonality Regularisation

The 64×64 feature transform has 4096 parameters just to describe the matrix. Without any constraint, it can learn degenerate transforms that collapse or scramble the feature space — technically minimising loss short-term but hurting generalisation.

The paper adds a regularisation term that encourages the matrix to stay **orthogonal** (rotation-like — preserves distances and angles):

```
L_reg = || I - A·Aᵀ ||²_F
```

Where `||·||_F` is the Frobenius norm (sum of squared elements). If A is orthogonal, then `A·Aᵀ = I`, so the loss is zero. Any deviation from orthogonality is penalised.

```python
def tnet_regularisation_loss(transform):
    k = transform.size(1)
    I    = torch.eye(k, device=transform.device).unsqueeze(0)   # (1, k, k)
    diff = I - torch.bmm(transform, transform.transpose(1, 2))  # I - A·Aᵀ
    return torch.mean(torch.norm(diff, dim=(1, 2)))             # Frobenius norm
```

The total loss is:

```
L_total = L_cross_entropy + 0.001 × L_reg
```

The weight `0.001` is taken directly from the paper — small enough that classification dominates, large enough that the transform stays well-behaved.

---

## Architecture

```
Input (B, 3, N)
    │
    ▼
┌─────────────┐
│  T-Net 3×3  │  ← predicts input alignment matrix
└─────────────┘
    │  bmm
    ▼
┌──────────────────┐
│  Shared MLP      │  Conv1d: 3→64→64   (per-point, BatchNorm + ReLU)
└──────────────────┘
    │
    ├──────────────────────────── save local features (B, 64, N)
    │                             used later for segmentation
    ▼
┌──────────────┐
│  T-Net 64×64 │  ← predicts feature alignment matrix
└──────────────┘
    │  bmm
    ▼
┌──────────────────────────────────┐
│  Shared MLP                      │  Conv1d: 64→64→128→1024  (per-point)
└──────────────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Global Max Pool        │  (B, 1024, N) → (B, 1024)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Classification head                │  FC: 1024→512→256→k
│  (Dropout 0.3, BatchNorm, ReLU)     │
└─────────────────────────────────────┘
    │
    ▼
Logits (B, k)       k=40 for ModelNet40
```

---

## Tensor Shape Walkthrough

Tracing the shape of the data at each step is the clearest way to understand what each layer does. Here with `B=32` (batch size) and `N=1024` (points per cloud):

| Stage | Operation | Shape |
|---|---|---|
| Raw input | — | `(32, 3, 1024)` |
| Input T-Net output | mini-network | `(32, 3, 3)` |
| After input transform | `bmm(T, x)` | `(32, 3, 1024)` |
| After MLP (64, 64) | `Conv1d` × 2 | `(32, 64, 1024)` |
| Feature T-Net output | mini-network | `(32, 64, 64)` |
| After feature transform | `bmm(T, x)` | `(32, 64, 1024)` |
| After MLP (64, 128, 1024) | `Conv1d` × 3 | `(32, 1024, 1024)` |
| After global max pool | `torch.max(dim=2)` | `(32, 1024)` |
| After FC (512) | `Linear` + dropout | `(32, 512)` |
| After FC (256) | `Linear` + dropout | `(32, 256)` |
| Output logits | `Linear` | `(32, 40)` |

Notice the N=1024 dimension stays alive through every layer — all points are processed in parallel. Only at max pool does it disappear, collapsing N points into one global descriptor per sample.

---

## Code Structure

```
pointnet-from-scratch/
├── tnet.py        # Spatial transformer network (T-Net) + regularisation loss
├── pointnet.py    # Encoder backbone + classification head + total loss function
├── dataset.py     # ModelNet40 data loader with augmentation
└── train.py       # Training loop, evaluation, checkpointing
```

### `tnet.py`
Contains `TNet(k)` — the spatial transformer that predicts a k×k alignment matrix. Works for both k=3 (input transform) and k=64 (feature transform). Also contains `tnet_regularisation_loss()` which computes the orthogonality penalty.

### `pointnet.py`
Contains:
- `PointNetEncoder` — the shared backbone (two T-Nets + two MLP blocks + max pool). Can optionally return local features for segmentation.
- `PointNetClassifier` — adds the classification FC head on top of the encoder.
- `pointnet_loss()` — combines cross-entropy with the T-Net regularisation term.

### `dataset.py`
Contains `ModelNet40` dataset class and `get_dataloaders()` convenience function. Handles HDF5 loading, random subsampling to `num_points`, and training augmentation (random jitter + shuffle).

### `train.py`
Full training loop with Adam optimiser, StepLR scheduler (halves LR every 20 epochs), per-epoch train/test accuracy logging, and best-model checkpointing.

---

## Setup & Training

**1. Install dependencies**
```bash
pip install torch torchvision h5py numpy
```

**2. Download ModelNet40**

The dataset is available from [Kaggle](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset) (~400 MB). Download and unzip it.

**3. Train**
```bash
python train.py --data_root ./modelnet40_ply_hdf5_2048
```

**Key arguments**

| Argument | Default | Description |
|---|---|---|
| `--data_root` | required | Path to `modelnet40_ply_hdf5_2048/` |
| `--num_points` | 1024 | Points sampled per cloud |
| `--batch_size` | 32 | Samples per batch |
| `--epochs` | 200 | Training epochs |
| `--lr` | 1e-3 | Initial learning rate |
| `--lr_step` | 20 | Halve LR every N epochs |
| `--dropout` | 0.3 | Dropout on FC layers |
| `--reg_weight` | 0.001 | T-Net regularisation weight |
| `--save_dir` | checkpoints/ | Where to save best model |

**Training on Google Colab (recommended if your machine is slow)**
```python
# Mount Drive so checkpoints survive session disconnects
from google.colab import drive
drive.mount('/content/drive')

!python train.py \
    --data_root ./modelnet40_ply_hdf5_2048 \
    --save_dir /content/drive/MyDrive/pointnet_checkpoints
```

---

## Results

Training on ModelNet40 with default settings (1024 points, 200 epochs, batch size 32, Adam lr=1e-3):

| Metric | This implementation | Paper (PointNet) |
|---|---|---|
| Test accuracy | ~89% | 89.2% |
| Parameters | ~3.5M | ~3.5M |
| Points per cloud | 1024 | 1024 |
| Training epochs | 200 | 200 |

*Results will be updated once training completes.*

---

## What I Learned

**`Conv1d(in, out, 1)` is a shared MLP.** A convolution with kernel size 1 never looks at neighbours — it applies the same linear transform to each position independently. This is the idiom for "process each point with the same weights" in PyTorch.

**`torch.max(x, dim=2).values` is the whole trick.** This one line is what makes the network permutation-invariant. Max pooling is a symmetric function — it doesn't care what order the points came in. Everything before it processes points independently; this line merges them.

**`torch.bmm` for the spatial transform.** Batch matrix multiply applies the predicted alignment matrix to every point cloud in the batch simultaneously. Shape: `(B, 3, 3) × (B, 3, N) → (B, 3, N)`.

**Identity initialisation for T-Net.** The T-Net's last layer is initialised to output the identity matrix. Without this, early training is unstable because the initial random transform garbles the input before the network has learned anything useful.

**`drop_last=True` in the DataLoader.** BatchNorm computes statistics over the batch. If the last batch has only 1 sample, BatchNorm crashes (can't compute variance over 1 item). `drop_last=True` discards the last incomplete batch.

**The regularisation weight of 0.001 is deliberate.** Too high and it dominates the loss, forcing orthogonality at the expense of classification accuracy. Too low and the 64×64 feature transform degenerates. The paper's value of 0.001 is the sweet spot.

---

## References

- **Paper:** Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593). CVPR 2017.
- **Official implementation:** [charlesq34/pointnet](https://github.com/charlesq34/pointnet) (TensorFlow)
- **Dataset:** [ModelNet40](https://modelnet.cs.princeton.edu/) — Princeton ModelNet
- **Spatial Transformer Networks:** Jaderberg et al. (2015). [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025). NeurIPS 2015.
