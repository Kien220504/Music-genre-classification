# Music Genre Classification (CNN2D-first)

A practical audio ML project to **classify music into genres** from raw audio files. The repo provides two complementary approaches: classic feature-engineering (MFCC/Chroma) with traditional ML classifiers, and a **2D Convolutional Neural Network (CNN2D) on log-Mel spectrograms**. CNN2D is the primary focus, with detailed guidance on preprocessing, architecture, training, and evaluation.

> Main notebook: `Music_genre_classification.ipynb`

---

## Project Overview

- **Goal:** Multiclass audio classification (e.g., `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`).  
- **Input:** Audio files (`.wav` / `.mp3`) → time–frequency features (log-Mel spectrogram or MFCCs).  
- **Output:** Predicted music genre label.

### Approach
1) **Classical Baselines (Features → ML):**  
   - Extract **MFCC**, **Chroma**, **Spectral Centroid**, **Zero-Crossing Rate**, etc.  
   - Train **Logistic Regression / SVM / Random Forest**  
   - Optional: PCA / Grid search for key hyperparameters

2) **Deep Learning (Primary):**  
   - **CNN2D on log-Mel spectrograms** (PyTorch)  
   - Optional data augmentation (**SpecAugment**: time/frequency masking, random time shift)  
   - Segment-level and clip-level prediction strategies

### Evaluation
`accuracy`, macro/micro `precision/recall/F1` (classification report), **confusion matrix**, and optionally **ROC/PR curves** with macro/micro AUC.

---

## Dataset

Two supported layouts:

```
# Option A — Folder-per-class (recommended)
#   Labels inferred from folder names
/content/data/<genre>/*.wav

# Option B — metadata CSV
/content/metadata.csv   # columns: path, label
/content/audio/...      # actual audio files
```

> You can change these paths in **Section 1. Data** of the notebook.  
> Resample all audio to a consistent rate (e.g., **22,050 Hz** or **16,000 Hz**) and **mono**.  
> With folder-based layout, genre = folder name; with CSV, the `label` column is used.

**Tip:** If clips are long, consider splitting into fixed-length segments (e.g., 3 s with 1 s hop) to increase dataset size and stability.

---

## Methodology

1. **Preprocessing & Feature Extraction**
   - Resample to a target sample rate; convert to mono
   - Trim/crop or zero-pad to a fixed duration, or **split into overlapping segments**
   - Compute **log-Mel spectrogram** (`n_fft=2048`, `hop_length=512`, `n_mels=64–128`, `log(1+x)`)
   - Standardize using **train-set statistics** (mean/std) for robustness

2. **Exploratory Data Analysis (EDA)**
   - Class balance, clip/segment length distributions
   - Example spectrograms per genre
   - Summary stats for MFCC/Chroma (min/mean/max, correlations)

3. **Modeling**
   - **Baselines (MFCC → ML):** LR, SVM, RF (optional PCA and hyperparameter search)  
   - **CNN2D (focus):**  
     - Input: spectrogram tensor (shape: `1 × n_mels × time`)  
     - Stacked Conv2d blocks + **MaxPool2d** + **Global Average Pooling** → Dense → Softmax  
     - Optimized with Adam, Cross-Entropy; optional CosineAnnealing/StepLR

4. **Evaluation & Visualization**
   - `classification_report` with macro/micro metrics
   - **Confusion matrix** by class
   - **ROC/PR curves & AUC** (one-vs-rest) when applicable
   - Report both **segment**- and **clip**-level performance if segmentation is used

---

## Why **CNN2D** for Music?

Music exhibits rich **time–frequency** structure. Converting audio to a **log-Mel spectrogram** yields a 2D “image” (`frequency × time`).  
**2D Convolutions** are ideal for learning local patterns such as:

- **Spectral textures**: harmonics, timbre indicative of instruments/production styles across genres  
- **Rhythmic motifs**: periodic energy patterns (e.g., backbeat, kick/snare placement) along the time axis  
- **Invariance**: Pooling confers robustness to small tempo/pitch variations; **Global Average Pooling (GAP)** reduces dependence on exact clip length and mitigates overfitting

### Reference CNN2D Architecture

> The notebook may have slightly different layer names/shapes; the design principles are the same.

```
Input: (B, 1, n_mels=128, T)

[Block 1]
Conv2d(1,   32, kernel=3x3, padding=1) → BatchNorm → ReLU
Conv2d(32,  32, kernel=3x3, padding=1) → BatchNorm → ReLU
MaxPool2d(2x2)
Dropout(0.20)

[Block 2]
Conv2d(32,  64, kernel=3x3, padding=1) → BatchNorm → ReLU
Conv2d(64,  64, kernel=3x3, padding=1) → BatchNorm → ReLU
MaxPool2d(2x2)
Dropout(0.30)

[Block 3]
Conv2d(64, 128, kernel=3x3, padding=1) → BatchNorm → ReLU
Conv2d(128,128, kernel=3x3, padding=1) → BatchNorm → ReLU
MaxPool2d(2x2)
Dropout(0.40)

Global Average Pooling  # H×W → 1×1, fewer params, length-invariant
Dense(128 → num_classes)
Softmax
```

**Design notes**
- **3×3 kernels**: good trade-off between expressivity and compute  
- **BN + ReLU**: stabilizes training  
- **MaxPool2d**: encourages small invariances over **time/frequency**; asymmetric pooling (e.g., `2×4`) can bias robustness toward one axis  
- **GAP**: replaces large fully-connected layers to reduce parameters and overfitting  
- **Dropout** increasing with depth provides regularization

### Variable Length & Clip Inference

- **Fixed-length training**: crop/pad to 3–5 s; captures rhythm + timbre cues  
- **Segment-level inference**: predict each segment, then **average probabilities** or **majority vote** for the full clip  
- **SpecAugment**:  
  - **Time masking**: hide random time spans (tempo robustness)  
  - **Frequency masking**: hide random Mel bands (pitch/instrument robustness)  
  - **Random time shift**: small temporal offsets

### Training & Hyperparameters (Typical)

- **Loss:** Cross-Entropy  
- **Optimizer:** Adam (`lr≈1e-3 → 1e-4` with scheduler)  
- **Batch size:** 16–64 (depends on spectrogram size/GPU)  
- **Epochs:** 20–50 (with **early stopping** on `val_loss`/`macro-F1`)  
- **Scheduler:** CosineAnnealing or StepLR (`gamma 0.5/0.1` every few epochs)  
- **Weight Decay:** `1e-4`–`1e-5`

---

## Tech Stack

- **Python**, **PyTorch**
- **librosa** (or **torchaudio**) for audio I/O & features
- **scikit-learn**, **pandas**, **numpy**
- **matplotlib** (and optionally **seaborn**) for charts
- Utilities: `tqdm`, `psutil`, `gc`

---

## Repository Files

- `Music_genre_classification.ipynb` — main notebook: data → EDA → models → evaluation  
- `data/` *(optional local layout)*  
  - `<genre>/*.wav` **or** `metadata.csv` + `audio/...`  
- `figures/` *(optional, if you save plots)*

---

## How to Run

### Option A — Google Colab (recommended, GPU available)

1. Open the notebook in Colab.  
2. Upload or mount data; adjust paths in **Section 1. Data**.  
3. Install dependencies if needed:
   ```python
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   !pip install librosa scikit-learn pandas numpy matplotlib seaborn tqdm
   ```
4. Run all cells in order. Plots and metrics will show inline.

### Option B — Local (VS Code / Jupyter)

1. Create an environment and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install torch torchvision torchaudio   # choose CUDA build matching your driver
   pip install librosa scikit-learn pandas numpy matplotlib seaborn tqdm
   ```
2. Place data under:
   ```
   data/<genre>/*.wav
   # or
   data/metadata.csv  (columns: path,label)  +  data/audio/...
   ```
   Update notebook paths from `/content/...` → `data/...`.
3. Launch Jupyter:
   ```bash
   jupyter lab   # or jupyter notebook
   ```
4. Run the notebook. Verify GPU is visible (optional):
   ```python
   import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
   ```

---

## Repro Tips

- **Seeds:** set seeds for NumPy/PyTorch to improve reproducibility.  
- **Normalization:** keep train-set mean/std for spectrogram normalization and reuse on val/test.  
- **Speed:** cache spectrograms to disk for larger datasets.  
- **Class imbalance:** try **class weights** or oversampling if needed.  
- **Overfitting:** use **SpecAugment**, **Dropout**, **Weight Decay**, and **Early Stopping**.

---

## Results & Reports

- The notebook produces **classification reports**, **confusion matrices**, and optional **ROC/PR** curves.  
- Compare **Baselines (MFCC→ML)** vs **CNN2D** to select a production candidate.  
- For long clips, report both **segment**- and **clip**-level metrics.

---

## Extensions (Optional)

- **CRNN** (CNN + BiLSTM/GRU) for combining local spectral cues with longer temporal dependencies.  
- **Audio Transformers** (e.g., AST) or **wav2vec2** fine-tuning.  
- **Self-supervised pretraining** on unlabeled audio.

---

## License

This repository is for educational/research purposes. If you plan to use code or data beyond coursework/research, review the dataset license(s) and add an appropriate project license.

# Additional Information

For any questions or issues, please open an issue in the repository or contact us at [Kien-Tran](mailto:trongkien220504@gmail.com) and [Kiet-Truong](mailto:truonghongkietcute@gmail.com).

Feel free to customize the project names, descriptions, and any other details specific to your projects. If you encounter any problems or have suggestions for improvements, don't hesitate to reach out. Your feedback and contributions are welcome!

Let me know if there’s anything else you need or if you have any other questions. I’m here to help!
