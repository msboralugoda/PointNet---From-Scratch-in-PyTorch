# PointNet — Teaching a Neural Network to "Feel" 3D Shapes

> *Most neural networks look at pixels. This one looks at point clouds.*

This is a from-scratch PyTorch implementation of **PointNet** (Qi et al., 2017), trained on the **ModelNet40** dataset for 3D object classification. No voxels, no projections, just raw points in space.

---

## Wait, What Is a Point Cloud?

A 3D object can be represented in many ways. PointNet uses the simplest one imaginable: just a bunch of (x, y, z) coordinates floating in space.

```
A chair, to PointNet:

[0.12, 0.44, 0.91]
[0.13, 0.44, 0.90]
[0.45, 0.02, 0.55]
... (1024 points total)
```

No faces. No edges. No textures. Just dots. And somehow, a neural network learns to tell a chair from an airplane from a toilet. Science is wild.

---

## The Core Idea

The big problem with point clouds is that they're **orderless**. If you shuffle the rows of the point matrix, it's still the same object, but a naive neural network would freak out.

PointNet's elegant fix: **use an operation that doesn't care about order.**

```
Point 1 ─┐
Point 2 ─┤─── per-point features ───► MAX POOL ───► global feature ───► label
Point 3 ─┤
  ...    ─┘
```

Max pooling across all points extracts the "loudest signal" regardless of order. That one idea is one of the most important of the whole paper.

---

## Architecture

```
Input (3 × 1024)
      │
      ▼
┌─────────────┐
│   T-Net 3   │  ← learns to align the input (like auto-rotating the object)
└─────────────┘
      │
      ▼
 Conv1D layers    (3 → 64 → 64)
      │
      ▼
┌──────────────┐
│   T-Net 64   │  ← learns to align features (in 64D space this time)
└──────────────┘
      │
      ▼
 Conv1D layers    (64 → 128 → 1024)
      │
      ▼
  MAX POOL         ← the magic: (B, 1024, N) → (B, 1024)
      │
      ▼
 FC layers        (1024 → 512 → 256 → 40)
      │
      ▼
  Predictions     (40 class scores)
```

### What's a T-Net?

The T-Net (Transform Network) is a mini neural network *inside* the main network. Its only job is to predict a transformation matrix that aligns the point cloud before processing.

Think of it like this: if someone hands you a chair upside down, you mentally rotate it before recognizing it. T-Net does that automatically, differentiably.

---

## Dataset

**ModelNet40** — 40 categories of 3D CAD models

| Split | Samples | Classes |
|-------|---------|---------|
| Train | 9,843 | 40 |
| Test | 2,468 | 40 |

Some of the 40 classes: `airplane`, `bathtub`, `bed`, `bench`, `chair`, `cup`, `guitar`, `lamp`, `laptop`, `person`, `piano`, `toilet`, `vase` ...

Each model is stored as an `.off` mesh file. We sample **1024 points** from the surface using `trimesh`, then normalize to a unit sphere.

---

## Implementation Details

| Hyperparameter | Value |
|----------------|-------|
| Points per object | 1024 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| LR schedule | Step decay (×0.5 every 20 epochs) |
| Regularization | Feature transform orthogonality loss (λ=0.001) |
| Dropout | 0.3 (before final FC) |

### The Orthogonality Loss — Why?

The 64D feature transform matrix should ideally be orthogonal (like a rotation matrix — it transforms without destroying information). We nudge it in that direction with a regularization term:

```
L_reg = || I - A·Aᵀ ||²_F
```

If `A·Aᵀ = I`, the matrix is perfectly orthogonal. This loss penalizes deviation from that.

---

## Training Progress

| Epoch | Loss | Train Acc | Test Acc |
|-------|------|-----------|----------|
| 1 | 2.17 | 44% | 50% |
| 2 | 1.51 | 58% | 49% |
| ... | ... | ... | ... |
| 10 | ~0.8 | ~82% | ~84% |

> Loss going down + accuracy going up = the network is actually learning something. Always a relief.

---

## Project Structure

```
pointnet-modelnet40/
├── README.md
├── dataset.py          # ModelNet40 Dataset class (.off parsing + point sampling)
├── model.py            # TNet + PointNetClassifier
├── train.py            # Training loop + evaluation
└── checkpoints/        # Saved model weights per epoch
```

---

## 🚀 How to Run (Google Colab)

### 1. Install dependencies
```python
!pip install trimesh kaggle -q
```

### 2. Download dataset from Kaggle
```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'
!kaggle datasets download -d balraj98/modelnet40-princeton-3d-object-dataset -p /content/modelnet40
```

### 3. Unzip
```python
import zipfile
with zipfile.ZipFile('/content/modelnet40/modelnet40-princeton-3d-object-dataset.zip', 'r') as z:
    z.extractall('/content/modelnet40/')
```

### 4. Train
```python
# Set config
NUM_POINTS = 1024
NUM_CLASSES = 40
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

# Then run training loop
```

---

## Key Concepts Explained Simply

**Why not use CNNs on 3D data?**
CNNs need a grid (pixels, voxels). Point clouds don't have a grid — they're irregular and unordered. You'd have to either voxelize (wastes memory, loses detail) or project to 2D (loses depth info). PointNet skips all that.

**Why max pooling and not average pooling?**
Max pooling picks the "most activated" feature across all points. It's like asking "did *any* point vote strongly for this feature?" Average pooling would dilute strong signals with noise from irrelevant points.

**Why does order not matter after max pooling?**
Because `max(a, b, c) == max(c, a, b) == max(b, c, a)`. The max operation is symmetric — shuffle the inputs all you want, the output is the same.

---

## 📚 Reference

```bibtex
@inproceedings{qi2017pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={CVPR},
  year={2017}
}
```

---

## What's Next

- [ ] Train for full 50 epochs
- [ ] Add data augmentation (random rotation, jitter)
- [ ] Implement PointNet segmentation head (ShapeNet dataset)
- [ ] Try PointNet++ (hierarchical local features)

