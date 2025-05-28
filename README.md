# Retinal eGFR Prediction using RETFound

This repository fine-tunes a Vision Transformer (ViT)-based RETFound model to predict estimated Glomerular Filtration Rate (eGFR), a marker of kidney function, directly from retinal fundus images using deep learning regression.

---

## 📁 Project Structure

```
retinal_egfr_regression/
├── data/
│   ├── images/                  # Folder containing all images
│   ├── fold1_train.csv          # CSV file: filename,egfr
│   ├── fold1_val.csv
│   └── fold1_test.csv
├── models/
│   └── retfound_regression.py   # Regression head for RETFound
├── datasets/
│   └── retinal_dataset.py       # PyTorch Dataset class
├── engine/
│   └── train_eval.py            # Training & evaluation functions
├── main_egfr.py                 # Training entry point for eGFR
├── predict_example.py           # Predict single image or folder
├── visualize_gradcam.py        # Grad-CAM visualization
└── checkpoints/                 # Saved models
```

---

## 🚀 How to Run

1. ✅ Install dependencies:
```bash
pip install -r requirements.txt
```

2. ✅ Organize your data:
   - Place images in `data/images/`
   - Prepare CSVs (`fold1_train.csv`, etc.) with format:
     ```csv
     filename,egfr
     12345_21015_0.0.png,85.3
     ```

3. ✅ Train model for eGFR:
```bash
python main_egfr.py
```

---

## 📊 Outputs

- Trained model saved in `checkpoints/`
- Metrics: MAE, RMSE, R² score
- Optionally: Grad-CAM overlay heatmaps for interpretation

---

## 🔬 Model Interpretation

Run Grad-CAM to visualize image regions influencing predictions:

```bash
python visualize_gradcam.py --image data/images/sample.png --checkpoint checkpoints/model_fold1.pth
```

---

## 🧪 Predict on New Images

```bash
python predict_example.py --image data/images/sample.png --checkpoint checkpoints/model_fold1.pth
```

---

## ✍️ Citation

To cite this work, please refer to:
> Hashim, E.B., et al. *Deep Learning Prediction of Kidney Function from Retinal Images Using RETFound*, University of Liverpool, 2025.

---

**Author**: [Effendy Bin Hashim](https://github.com/Effendy77)  
**License**: MIT
