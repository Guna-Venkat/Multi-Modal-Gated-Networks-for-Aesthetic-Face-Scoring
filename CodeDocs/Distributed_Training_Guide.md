# Distributed Training Guide: Kaggle + Private GPU

This is your official playbook for running the Facial Beauty Prediction model splitting execution across Kaggle and your private GPU system.

## 1. KAGGLE EXECUTION (Runs M1, M2, M3)

Kaggle is perfect for training the baseline models (M1, M2, M3) because the 12-hour limit is long enough for ResNet-18, and we bypass the data processing phase because the zip file already contains the `cache/` folder.

**Step 1:** In Kaggle, click **Datasets > New Dataset** and upload `Kaggle_Project.zip`. Name it something like "CV Project Data". Kaggle will create a slug URL (e.g. `cv-project-data`).
**Step 2:** Open a New Kaggle Notebook and click **+ Add Data** to attach your new dataset.
**Step 3:** Turn on the **GPU P100** or **T4x2** in the notebook settings.
**Step 4:** Run the following 3 cells to train the models!

### Kaggle Cell 1: Copy to Writable Space
```python
!cp -r /kaggle/input/cv-project-data/CodeDocs /kaggle/working/CodeDocs
!cp -r /kaggle/input/cv-project-data/cache /kaggle/working/cache
```
*(If your dataset was named differently, change `cv-project-data` to match your exact Kaggle dataset slug!)*

### Kaggle Cell 2: Connect Paths Automatically
```python
import sys
sys.path.append('/kaggle/working/CodeDocs')

config_path = '/kaggle/working/CodeDocs/config.py'
with open(config_path, 'r') as f:
    text = f.read()

text = text.replace('ENV = "local"', 'ENV = "kaggle"')
text = text.replace('DATASET_DIR     = "/kaggle/input/scut-fbp5500"', 'DATASET_DIR     = "/kaggle/input/cv-project-data/dataset"')

with open(config_path, 'w') as f:
    f.write(text)
```

### Kaggle Cell 3: Train M1, M2, M3
```python
!python /kaggle/working/CodeDocs/phase2_m1_cnn.py
!python /kaggle/working/CodeDocs/phase3_m2_landmarks.py --m3
```

**Step 5:** Once finished, look in the Kaggle Output panel. Download `m1_preds.npy`, `m2_preds.npy`, and `m3_preds.npy` from the `/kaggle/working/results/` folder.

---

## 2. PRIVATE GPU EXECUTION (Runs M4 & Final Evaluation)

Your private GPU machine will run M4 (Adaptive Fusion), which is computationally heavy since it trains both a ResNet and an MLP simultaneously, and then will evaluate all the models!

**Step 1:** Unzip `Private_GPU_Project.zip` on your private machine. (Or use the `mainIdea` folder as-is).
**Step 2:** Open your terminal inside the project root and run M4:
```bash
python CodeDocs/phase5_m4_fusion.py --lm 3d
```

**Step 3:** Move the `m1_preds.npy`, `m2_preds.npy`, and `m3_preds.npy` files you downloaded from Kaggle directly into the `results/` folder alongside your freshly generated `m4_preds.npy`.

**Step 4:** Run the final Phase 6 Evaluation to combine all the predictions into your comparison table, beta-gating histograms, and scatter plots:
```bash
python CodeDocs/phase6_evaluation.py
```

All your final plots and `.csv` tables will be instantly generated in the `results/` folder for your presentation!
