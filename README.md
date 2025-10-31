# Bitcoin Volatility Prediction with Explainable AI 📊

A comprehensive machine learning project that predicts Bitcoin volatility using multiple models and provides explainable insights through feature importance analysis and SHAP values.

## 🎯 Project Overview

Cryptocurrency markets are extremely volatile with sudden and unpredictable price swings. This project:
- Forecasts Bitcoin's next-day realized volatility
- Compares multiple ML models (Linear Regression, Random Forest, XGBoost, LSTM)
- Provides model explainability through feature importance and SHAP analysis
- Includes comprehensive data visualizations

## 📁 Project Structure

```
bitcoin-volatility-xai/
├── bitcoin_volatility_analysis.ipynb  # Interactive Jupyter notebook (recommended!)
├── data_preparation.py                # Data fetching and cleaning
├── feature_engineering.py             # Technical indicator calculations  
├── models.py                          # ML model implementations
├── visualization.py                   # Comprehensive plotting functions
├── main.py                            # CLI pipeline orchestration
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Two Ways to Use This Project:

**🎯 Option 1: Jupyter Notebook (Recommended for Exploration)**
- Interactive cell-by-cell execution
- Inline visualizations
- Perfect for learning and experimentation
- See all plots without saving to files

**⚙️ Option 2: Command-Line Scripts (Recommended for Production)**
- Automated pipeline execution
- Batch processing
- Scheduled jobs and automation
- Saves all results to files

## 🚀 Quick Start

### Installation

1. **Clone the repository** (or navigate to project directory)
```bash
cd bitcoin-volatility-xai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

**Run the complete pipeline:**
```bash
python main.py --visualize --shap
```

This will:
1. ✅ Download BTC/USDT data from Binance
2. ✅ Engineer 50+ technical features
3. ✅ Train 5 different models (Baseline, Linear Regression, Random Forest, XGBoost, LSTM)
4. ✅ Generate comprehensive visualizations
5. ✅ Compute SHAP values for model explainability
6. ✅ Save all results to `results/` directory

### Command-Line Options

```bash
# Basic run without visualizations
python main.py

# Custom date range
python main.py --start-date 2020-01-01 --visualize

# Larger test set
python main.py --test-size 0.3 --visualize

# Force refresh data from API
python main.py --refresh --visualize

# Full analysis with SHAP
python main.py --visualize --shap
```

### Using Individual Modules

You can import and use modules individually in your own scripts:

```python
# Example: Custom analysis script
from data_preparation import load_or_fetch_data
from feature_engineering import create_all_features
from models import RandomForestModel
from visualization import plot_predictions

# Load data
df = load_or_fetch_data("BTCUSDT", "2020-01-01")

# Create features
df_features = create_all_features(df)

# Train model
model = RandomForestModel(n_estimators=400)
model.fit(X_train, y_train)

# Visualize
plot_predictions(y_test, model.predict(X_test), dates)
```

Each module can also be tested independently:
```bash
python data_preparation.py
python feature_engineering.py
python models.py
python visualization.py
```

## 📊 Features

### Data Features (50+ technical indicators)

**Price Features:**
- Returns (1d, 3d, 5d, 7d)
- Moving Averages (5, 10, 20, 30, 50-day)
- EMA (12, 26) and MACD
- Bollinger Bands

**Volume Features:**
- Rolling volume statistics
- Volume changes
- Price-volume interactions

**Volatility Features:**
- Historical volatility
- Rolling volatility statistics
- Return volatility

**Momentum Features:**
- RSI (7, 14, 21-day)
- Rate of Change (ROC)
- Momentum indicators

### Models Implemented

1. **Baseline (Persistence)** - Previous day's volatility
2. **Linear Regression** - Weighted linear relationships
3. **Random Forest** - Non-linear patterns with ensemble
4. **XGBoost** - Gradient boosting with regularization
5. **LSTM** - Sequential patterns with deep learning

### Target Variable

Next-day realized volatility = (High - Low) / Open

## 📈 Visualizations

The pipeline generates comprehensive visualizations:

### Data Exploration
- Price history with volume
- Volatility patterns over time
- Feature correlations heatmap
- Feature distributions

### Model Results
- Actual vs predicted volatility (time series)
- Prediction scatter plots
- Model performance comparison
- Feature importance charts
- SHAP summary plots

All visualizations are saved to:
- `results/data_exploration/` - Data analysis plots
- `results/model_results/` - Model evaluation plots

## 📋 Output Files

After running the pipeline, you'll find:

```
results/
├── data_exploration/          # Data visualizations
│   ├── 01_price_history.png
│   ├── 02_volatility_analysis.png
│   ├── 03_feature_correlations.png
│   └── 04_feature_distributions.png
├── model_results/             # Model visualizations
│   ├── model_comparison.png
│   ├── predictions_*.png
│   ├── feature_importance_*.png
│   └── shap_summary.png
├── predictions.csv            # All model predictions
├── metrics.csv                # Performance metrics
└── feature_importance_*.csv   # Feature rankings
```

## 🎓 Evaluation Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error) - Average prediction error
- **RMSE** (Root Mean Squared Error) - Penalizes large errors
- **R²** (R-squared) - Proportion of variance explained

## 🔍 Model Explainability

### Feature Importance
Shows which features contribute most to predictions:
- RSI and momentum indicators
- Historical volatility measures
- Moving average ratios
- Volume changes

### SHAP Values
Provides instance-level explanations:
- How each feature affects individual predictions
- Feature interaction effects
- Global feature importance rankings

## 🛠️ Technical Details

### Data Source
- **API**: Binance public API (no key required)
- **Pair**: BTC/USDT
- **Interval**: Daily candles
- **Default Range**: 2019-01-01 to present

### Data Cleaning
- Forward-fill missing values
- Z-score outlier detection (threshold: 5.0)
- Chronological sorting

### Train/Test Split
- Time-ordered split (important for time series)
- Default: 80% train, 20% test
- No data leakage

### LSTM Architecture
- Lookback window: 10 days
- 2 LSTM layers (50 → 25 units)
- Dropout: 0.2
- Early stopping with patience: 5

## 📚 Dependencies

**Core:**
- Python 3.8+
- NumPy, Pandas
- Requests

**ML:**
- scikit-learn
- XGBoost
- TensorFlow (for LSTM)

**Visualization:**
- Matplotlib
- Seaborn

**Explainability:**
- SHAP

See `requirements.txt` for specific versions.

## 🤝 Contributing

This is an educational project for CS6140. Feel free to:
- Add new features (e.g., Fear & Greed Index)
- Implement additional models
- Improve visualizations
- Optimize hyperparameters

## 📝 Notes

- First run will download ~6 years of BTC data (~2,000 rows)
- LSTM training can take 2-5 minutes on CPU
- SHAP computation can be slow for large datasets (uses subset)
- Visualizations are saved automatically with `--visualize` flag

## 🔮 Future Enhancements

- [ ] Add Fear & Greed Index data
- [ ] Implement regime classification (low/medium/high volatility)
- [ ] Add social media sentiment features
- [ ] Hyperparameter optimization
- [ ] Cross-validation for time series
- [ ] Real-time prediction API
- [ ] Interactive dashboard (Streamlit/Plotly)

## 📧 Contact

For questions or suggestions about this project, please reach out!

---

**⚠️ Disclaimer**: This project is for educational purposes only. Not financial advice.

