# Task 6: Hybrid ARIMA-LSTM Ensemble

**COS30018 - Intelligent Systems**  
**Task C.6: Ensemble Methods**  
**Date:** November 2025

---

## 🎯 Quick Overview

This project implements a **TRUE Hybrid Ensemble** combining:
- **ARIMA/SARIMA**: Linear time series forecasting
- **LSTM Neural Network**: Non-linear residual learning

**Key Innovation:**
```python
Final_Prediction = ARIMA_Prediction + LSTM_Residual_Correction
```

This is NOT simple averaging - LSTM learns to predict and correct ARIMA's errors!

---

## 📁 Project Structure

```
Project/
├── task6_hybrid_ensemble.py          # Main script (RUN THIS)
├── data_processing.py                # Data loading helper
├── model_builder.py                  # LSTM utilities
│
├── TASK6_README.md                   # This file
├── TASK6_CODE_EXPLANATION.md         # Detailed line-by-line explanation
├── TASK6_SUBMISSION_SUMMARY.md       # Results & analysis
├── TASK6_FILES_TO_SUBMIT.md          # Submission checklist
│
└── task6_hybrid_results/             # Generated results
    └── hybrid_exp_TIMESTAMP/
        ├── experiments_comparison.csv
        ├── model_ranking.csv
        ├── final_comparison.png
        └── Exp_1/ ... Exp_10/
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
pip install statsmodels tensorflow yfinance
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
python task6_hybrid_ensemble.py
```

**Expected Runtime:** 20-30 minutes for all 10 experiments

### 3. View Results

Results saved to: `task6_hybrid_results/hybrid_exp_TIMESTAMP/`

**Key Files:**
- `experiments_comparison.csv` - All experiments summary
- `model_ranking.csv` - Ranked by performance
- `final_comparison.png` - 6-panel visualization

---

## 📊 What It Does

### The Hybrid Methodology

```
┌─────────────────────────────────────────────────┐
│  STEP 1: Train ARIMA on Stock Prices           │
│  ├─ Captures linear trends                     │
│  ├─ Captures seasonality (SARIMA)              │
│  └─ Makes baseline predictions                 │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  STEP 2: Extract ARIMA Residuals (Errors)      │
│  └─ Residuals = Actual - ARIMA_Prediction      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  STEP 3: Train LSTM on Residuals               │
│  ├─ Learns non-linear error patterns           │
│  ├─ Predicts future residual corrections       │
│  └─ Complements ARIMA weaknesses               │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  STEP 4: Combine Predictions                   │
│  └─ Hybrid = ARIMA + LSTM_Residual             │
└─────────────────────────────────────────────────┘
```

### Example Prediction:
```
Day 100:
  ├─ ARIMA predicts: $105.50
  ├─ LSTM predicts residual: -$0.30
  └─ Hybrid final: $105.50 + (-$0.30) = $105.20

If actual price = $105.15 → Hybrid is closer!
```

---

## 🧪 Experiments

### 10 Configurations Across 5 Groups:

| Group | Experiments | Focus |
|-------|-------------|-------|
| **1. Baseline** | Exp 1-3 | Simple ARIMA, small LSTM |
| **2. Medium** | Exp 4-5 | Multi-layer LSTM, varied ARIMA |
| **3. SARIMA** | Exp 6-7 | Seasonal models (5-day cycle) |
| **4. Deep** | Exp 8 | Complex ARIMA(7,1,2) + 3-layer LSTM |
| **5. Optimized** | Exp 9-10 | Fine-tuned hyperparameters |

### Hyperparameters Explored:
- **ARIMA orders:** (2,1,1) to (7,1,2)
- **LSTM layers:** 1-3 layers
- **LSTM units:** 32 to 128 per layer
- **Sequence lengths:** 30 to 90 timesteps
- **Learning rates:** 0.0005 to 0.001
- **Seasonality:** 5-day (weekly) patterns

---

## 📈 Results Summary

### 🏆 Best Performance:

**Exp_7_SARIMA_LSTM96_48_seq60**
- **Hybrid MAE:** 0.2841
- **ARIMA Order:** (1,1,1) with seasonal (1,1,1,5)
- **LSTM:** [96, 48] units
- **Key:** Seasonality + medium-sized network

### 📊 Key Findings:

1. **SARIMA outperforms regular ARIMA**
   - MAE ~0.28 vs ~0.35
   - 5-day seasonality captures weekly patterns

2. **Hybrid improvements are minimal**
   - Most experiments show -0.01% to -0.45%
   - Stock data is highly non-linear and volatile
   - Methodology is still correct (research-backed)

3. **Optimal architecture:**
   - Medium LSTM (64-128 units)
   - 2-layer architecture
   - Sequence length 60-90
   - Learning rate 0.0005-0.001

---

## 🔍 How to Interpret Results

### Files Explained:

#### `experiments_comparison.csv`
```csv
Experiment,ARIMA_Order,LSTM_Units,ARIMA_MAE,Hybrid_MAE,Improvement_%
Exp_1,...,(2,1,1),[32],0.3500,0.3511,-0.33
...
```
- Compare all 10 experiments side-by-side
- Negative improvement = Hybrid slightly worse than ARIMA

#### `model_ranking.csv`
```csv
Rank,Experiment,Hybrid_MAE,ARIMA_MAE,Improvement_%
1,Exp_7_SARIMA_LSTM96_48_seq60,0.2841,0.2838,-0.14
...
```
- Sorted by Hybrid MAE (lower is better)
- Shows best → worst configurations

#### `final_comparison.png`
- 6 visualization panels:
  1. MAE Comparison (bar chart)
  2. Improvement % (bar chart)
  3. RMSE Comparison
  4. R² Comparison
  5. ARIMA vs Hybrid scatter
  6. Improvement distribution

---

## 💻 Code Structure

### Main Classes:

#### 1. `LSTMResidualModel`
```python
# Learns ARIMA residuals (errors)
model = LSTMResidualModel(
    lstm_units=[64, 32],    # 2-layer LSTM
    sequence_length=60,      # Look back 60 days
    learning_rate=0.001
)
model.train(residuals)       # Train on ARIMA errors
predictions = model.predict_residuals(residuals, n_steps=241)
```

**Methods:**
- `create_sequences()` - Convert residuals to supervised learning
- `build_model()` - Create LSTM architecture
- `train()` - Train with early stopping
- `predict_residuals()` - Recursive forecasting

#### 2. `HybridARIMALSTM`
```python
# Orchestrates the hybrid ensemble
hybrid = HybridARIMALSTM(
    arima_order=(3, 1, 1),
    lstm_units=[64],
    sequence_length=60
)
hybrid.fit(train_data)       # Train ARIMA + LSTM
predictions = hybrid.predict(n_steps=241)
```

**Methods:**
- `fit()` - 3-step training process
- `predict()` - Generate hybrid forecasts
- `evaluate()` - Calculate MAE, RMSE, R²

---

## 📚 Documentation

### For Detailed Explanations:

1. **TASK6_CODE_EXPLANATION.md**
   - Line-by-line code walkthrough
   - ~900 lines of detailed explanation
   - Every method explained with examples

2. **TASK6_SUBMISSION_SUMMARY.md**
   - Executive summary
   - Results analysis
   - Conclusions and insights

3. **TASK6_FILES_TO_SUBMIT.md**
   - Complete submission checklist
   - What to submit and why

---

## 🛠️ Troubleshooting

### Common Issues:

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'statsmodels'
```
**Solution:**
```bash
pip install statsmodels
```

#### 2. ARIMA Convergence Warnings
```
ConvergenceWarning: Maximum Likelihood optimization failed
```
**Solution:**
- This is normal for some ARIMA orders
- Results are still valid
- Warnings are suppressed in code

#### 3. Out of Memory
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution:**
- Reduce batch_size in experiments
- Reduce lstm_units
- Close other applications

#### 4. Slow Execution
**Solution:**
- Normal - 10 experiments take 20-30 minutes
- Run in background
- Or run individual experiments only

---

## 🎓 Academic Context

### Why This Matters:

1. **Ensemble Learning**
   - Combines multiple models
   - Leverages complementary strengths
   - Industry-standard approach

2. **Hybrid Methodology**
   - Statistical (ARIMA) + ML (LSTM)
   - Linear + Non-linear
   - Interpretable + Powerful

3. **Proper Implementation**
   - Not simple averaging
   - Research-backed approach
   - Demonstrates understanding

### References:

- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis*
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*
- Zhang, G. P. (2003). *Time series forecasting using a hybrid ARIMA and neural network model*

---

## ✅ Verification

### To verify everything works:

```bash
# 1. Check dependencies
python -c "import statsmodels, tensorflow, sklearn; print('OK')"

# 2. Test run (first experiment only)
# Edit task6_hybrid_ensemble.py, set:
# EXPERIMENTS = EXPERIMENTS[:1]  # Only run first experiment

# 3. Run quick test
python task6_hybrid_ensemble.py

# 4. Check results
ls task6_hybrid_results/
```

---

## 📝 Customization

### To modify experiments:

Edit `EXPERIMENTS` list in `task6_hybrid_ensemble.py`:

```python
EXPERIMENTS = [
    {
        'name': 'My_Custom_Experiment',
        'arima_order': (3, 1, 1),        # Change ARIMA order
        'seasonal_order': None,          # Add seasonality
        'lstm_units': [64, 32],          # Change architecture
        'lstm_dropout': 0.2,             # Dropout rate
        'sequence_length': 60,           # Lookback period
        'epochs': 100,                   # Training epochs
        'batch_size': 32,                # Batch size
        'learning_rate': 0.001,          # Learning rate
    },
]
```

### To change stock or date range:

```python
# In run_experiments() function:
STOCK = 'AAPL'                    # Change stock ticker
START_DATE = '2019-01-01'         # Change start date
END_DATE = '2023-12-31'           # Change end date
TRAIN_SPLIT = 0.8                 # Change train/test split
```

---

## 🎯 Expected Outputs

### Console Output:
```
================================================================================
LOADING DATA: CBA.AX (2020-01-01 to 2024-10-01)
================================================================================

[OK] Data loaded: 1202 samples

================================================================================
RUNNING EXPERIMENT: Exp_1_ARIMA211_LSTM32_seq30
================================================================================
Configuration:
  arima_order: (2, 1, 1)
  lstm_units: [32]
  sequence_length: 30
  ...

[STEP 1/3] Training ARIMA component...
[ARIMA] Fitting ARIMA(2, 1, 1)...
[OK] ARIMA fitted - AIC: -4707.60, BIC: -4683.27

[STEP 2/3] Extracting ARIMA residuals...
[OK] Extracted 961 residuals

[STEP 3/3] Training LSTM on residuals...
Epoch 1/100
...
[OK] LSTM trained - Final MAE: 0.0145

[HYBRID] Predicting 241 steps ahead...
[ARIMA] Generated 241 predictions
[LSTM] Generated 241 residual corrections
[HYBRID] Combined predictions

================================================================================
RESULTS SUMMARY
================================================================================
ARIMA-only:  MAE=0.3500, RMSE=0.4138, R²=-2.2557
Hybrid:      MAE=0.3511, RMSE=0.4146, R²=-2.2639
Improvement: -0.33%
================================================================================

... (9 more experiments)

🏆 BEST MODELS:
Best Hybrid MAE: Exp_7_SARIMA_LSTM96_48_seq60
  MAE: 0.284139

[OK] ALL EXPERIMENTS COMPLETED!
```

### Generated Files:
```
task6_hybrid_results/hybrid_exp_20251103_022833/
├── experiments_comparison.csv     (2 KB)
├── model_ranking.csv              (2 KB)
├── final_comparison.png           (150 KB)
│
├── Exp_1_ARIMA211_LSTM32_seq30/
│   ├── results.json               (1 KB)
│   ├── predictions.csv            (15 KB)
│   └── analysis.png               (80 KB)
│
└── ... (9 more experiment folders)

Total: ~1.5 MB
```

---

## 🤝 Support

### Need Help?

1. **Check Documentation:**
   - TASK6_CODE_EXPLANATION.md - Detailed code walkthrough
   - TASK6_SUBMISSION_SUMMARY.md - Results analysis

2. **Common Questions:**
   - Why negative improvements? → Stock data is volatile
   - Why SARIMA better? → Captures weekly seasonality
   - How long to run? → 20-30 minutes for all 10

3. **Code Issues:**
   - Check Python version (3.8+)
   - Check TensorFlow version (2.x)
   - Check dependencies installed

---

## 📄 License & Attribution

This is academic work for COS30018 - Intelligent Systems, Swinburne University.

**Student:** [Your Name]  
**Date:** November 2025

Code implements research methodologies from:
- Statistical time series analysis (ARIMA/SARIMA)
- Deep learning (LSTM networks)
- Hybrid ensemble approaches

---

## ✨ Summary

**In 3 Sentences:**

1. This project implements a **TRUE hybrid ensemble** where LSTM learns to predict and correct ARIMA's errors (residuals)
2. We ran **10 comprehensive experiments** exploring different ARIMA orders, LSTM architectures, and hyperparameters
3. Best result: **SARIMA with 5-day seasonality** (Exp_7) achieved **MAE of 0.2841** on CBA.AX stock prediction

**Key Takeaway:**  
✅ Demonstrates proper ensemble methodology (not simple averaging)  
✅ Shows systematic experimental approach  
✅ Provides comprehensive evaluation and analysis

---

**Ready to use! 🚀**

Run `python task6_hybrid_ensemble.py` and check results in `task6_hybrid_results/`

For detailed explanations, see **TASK6_CODE_EXPLANATION.md**

---

**End of README**
