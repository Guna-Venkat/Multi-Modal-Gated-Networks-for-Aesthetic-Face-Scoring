# Transformer Experiment Guide for Kaggle

This is the Kaggle notebook procedure for running the new `transformer_experiments.py` script.
Since M5 and M7 process images, they require the familiar `filepath` overwrite fix.

Create a single Kaggle Notebook cell and paste the following:

```python
import os
import pandas as pd
import numpy as np

import CodeDocs.config as C
from CodeDocs.transformer_experiments import run_experiment

# 1. Load data
train_csv = os.path.join(C.CACHE_DIR, "train_split.csv")
test_csv  = os.path.join(C.CACHE_DIR, "test_split.csv")
if not os.path.exists(train_csv):
    raise RuntimeError("Run phase1_data_prep.py first.")

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

# 2. Fix absolute paths to point to Kaggle's working dataset directory
train_df["filepath"] = train_df["filename"].apply(lambda fn: os.path.join(C.DATASET_DIR, "Images", "Images", fn))
test_df["filepath"]  = test_df["filename"].apply(lambda fn: os.path.join(C.DATASET_DIR, "Images", "Images", fn))

# 3. CHOOSE YOUR EXPERIMENT:
# Change "m6" to "m5" (ViT Texture) or "m7" (Cross-Attention Fusion)
EXPERIMENT_TO_RUN = "m6" 

print(f"Launching Experiment: {EXPERIMENT_TO_RUN.upper()}...")
run_experiment(EXPERIMENT_TO_RUN, train_df, test_df)

# Note: Once this stops training, it saves:
# - predictions to `checkpoints/{EXPERIMENT_TO_RUN}_transformer_preds.npy`
# - heatmaps to `checkpoints/{EXPERIMENT_TO_RUN}_transformer_heatmaps.npy`!
```

### Visualizing the Heatmaps (After M6 finishes)
If you ran `m6`, it generated a `[Batch, 468]` shape text array containing the attention weights! 
If you want to visualize which landmarks M6 cared about most, simply load the `.npy` file.

```python
import numpy as np
heatmaps = np.load('/kaggle/working/checkpoints/m6_transformer_heatmaps.npy')

# `heatmaps[0]` is a list of 468 decimal probabilities showing exactly where the 
# Transformer looked (e.g. eyes or jaw) to predict the beauty score of the first image!
print(heatmaps[0]) 
```
