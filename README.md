# Multi-Modal Gated Networks for Aesthetic Face Scoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00bcd4?logo=google&logoColor=white)](https://google.github.io/mediapipe/)

A deep learning project focused on **Facial Beauty Prediction** using a disentangled methodology that separates **texture** (pixel data) from **geometry** (landmark data). This repository implements eight distinct architectures, ranging from standard CNNs and MLPs to advanced Adaptive Gating Networks and Transformer-based Cross-Attention models.

---

## 🚀 Key Innovations

### 1. Geometry vs. Texture Disentanglement
Most facial scoring models conflate lighting, makeup (texture), and facial proportions (geometry). This project explicitly separates them by feeding raw images and 3D facial landmarks into parallel branches.

### 2. Adaptive Gating Mechanisms (M4, M8)
The core novelty lies in the **Adaptive Fusion** layer. Instead of simple concatenation, a gating network analyzes the face and determines how much to trust each branch:
```latex
ŷ = α(I) · y_texture  +  β(I) · y_geometry
```
Where `α + β = 1`. This allows the model to rely more on geometry for faces with heavy makeup or challenging lighting.

### 3. Transformer-based Cross-Attention (M7)
Uses ViT-B/16 patch tokens as keys/values and facial landmark tokens as queries. This enables the model to "attend" to specific image patches (e.g., skin texture, eyes) conditioned on the geometric structure of the face.

---

## 📊 Experimental Results

Experiments conducted on the **SCUT-FBP5500** dataset. Performance is measured using Pearson Correlation (ρ), Mean Absolute Error (MAE), and Root Mean Square Error (RMSE).

| Model ID | Architecture                | Modality              | Pearson ρ | MAE    | RMSE   |
|----------|-----------------------------|-----------------------|-----------|--------|--------|
| **M1**   | ResNet-18 (CNN)             | Image Only            | 0.8857    | 0.2399 | 0.3218 |
| **M2**   | 2D Landmark MLP             | Landmarks (2D)        | 0.7109    | 0.3788 | 0.4919 |
| **M3**   | 3D Landmark MLP             | Landmarks (3D)        | 0.7565    | 0.3600 | 0.4585 |
| **M4**   | **Adaptive Fusion (CNN)**   | Multimodal (Gated)    | **0.8886**| 0.2418 | **0.3161**|
| **M5**   | ViT-B/16 Baseline           | Image Only            | 0.8265    | 0.2914 | 0.3882 |
| **M6**   | Landmark Transformer        | Landmarks (3D)        | 0.5527    | 0.4534 | 0.5764 |
| **M7**   | **Cross-Attention Fusion**  | Multimodal (Attn)     | 0.8814    | 0.2651 | 0.3409 |
| **M8**   | Gated ViT Fusion            | Multimodal (Gated)    | 0.8258    | 0.2929 | 0.3888 |

> [!NOTE]
> M4 (Adaptive Fusion with ResNet-18 backbone) achieved the highest Pearson correlation and lowest RMSE, demonstrating the effectiveness of content-dependent gating.

---

## 📂 Repository Structure

```
.
├── CodeDocs/                # Core Source Code
│   ├── config.py            # Central Environment & Hyperparams
│   ├── datasets.py          # PyTorch DataLoaders & Preprocessing
│   ├── models.py            # M1-M4 Architectures
│   ├── trainer.py           # Training loops & Early Stopping
│   ├── run_all.py           # Master script for end-to-end execution
│   └── phase6_transformer_experiments.py # M5-M8 Architectures
├── results/                 # Training logs, plots, & evaluation CSVs
│   ├── final_metrics_comparison.csv
│   └── m7_attention_heatmap.png
├── checkpoints/             # Saved model weights (.pt files)
└── cache/                   # Pre-calculated 3D landmarks & splits
```

---

## 🛠️ Installation & Setup

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/Guna-Venkat/Multi-Modal-Gated-Networks-for-Aesthetic-Face-Scoring
   cd Multi-Modal-Gated-Networks-for-Aesthetic-Face-Scoring
   ```

2. **Configure Environment**:
   Edit `CodeDocs/config.py` and set `ENV = "local"` or `"kaggle"`. Ensure `DATASET_DIR` points to your SCUT-FBP5500 image folder.

3. **Install Dependencies**:
   ```bash
   pip install torch torchvision mediapipe pandas numpy opencv-python scikit-learn
   ```

---

## 💻 Usage

### Running the Full Pipeline
The most convenient way to reproduce the results is via `run_all.py`:
```bash
python CodeDocs/run_all.py --epochs-m1 30 --epochs-m4 40
```

### Running Specific Experiments (M5-M8)
To run the transformer-based experiments:
```bash
python CodeDocs/phase6_transformer_experiments.py --run m7
```

### Distributed Training
For high-performance training on multi-GPU systems or Kaggle TPUs, refer to the [Distributed Training Guide](file:///c:/Users/gunav/Downloads/Multi-Modal-Gated-Networks-for-Aesthetic-Face-Scoring/CodeDocs/Distributed_Training_Guide.md).

---

## 📈 Visualizations

Check the `results/` folder for detailed analysis:
- **Attention Heatmaps**: (`m7_attention_heatmap.png`) See where the model looks.
- **Gating Distribution**: (`m4_beta_distribution.png`) Understand the geometry vs. texture bias across the dataset.
- **Performance Plots**: (`final_performance_comparison.png`) Model ranking summary.

---

## 🤝 Acknowledgements
- **Dataset**: SCUT-FBP5500 Database.
- **Backbones**: Torchvision ResNet and ViT implementations.
- **Landmarks**: Google MediaPipe Face Mesh.
