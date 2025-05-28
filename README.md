
# Retinal Arterial Stiffness Prediction using RETFound

This project fine-tunes a Vision Transformer (RETFound) model to predict arterial stiffness (e.g., Pulse Wave Velocity) from retinal fundus images using regression.

## 📁 Project Structure

```
retinal_stiffness_regression/
├── data/
│   ├── images/                  # Folder containing all images
│   ├── fold1_train.csv          # CSV file: filename,pwv
│   ├── fold1_val.csv
│   └── fold1_test.csv
├── models/
│   └── retfound_regression.py   # Regression head for RETFound
├── datasets/
│   └── retinal_dataset.py       # PyTorch Dataset class
├── engine/
│   └── train_eval.py            # Training & evaluation functions
├── main.py                      # Training entry point
└── checkpoints/                 # Saved models
```

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your data:
   - Place all images in `data/images/`
   - Create CSVs with columns: `filename,pwv` (and optionally `egfr` for future use)

3. Run training:
```bash
python main.py
```

## 📊 Outputs

- Model metrics: MAE, RMSE, R² score
- Model saved to `checkpoints/retfound_regression_fold1.pth`

## 🔮 Roadmap

- ✅ Predict arterial stiffness (PWV) from fundus images
- 🔜 Add support for predicting eGFR
- 🔜 Extend to 5-fold cross-validation
- 🔜 Grad-CAM visualization for model interpretability

## ✍️ Citation

To cite this work, please refer to:
> Hashim, E.B., et al. *Predicting Arterial Stiffness from Retinal Images using Deep Learning*. University of Liverpool, 2025.

---

**Author**: [Effendy Bin Hashim](https://github.com/Effendy77)  
**License**: MIT
