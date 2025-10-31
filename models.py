"""
Models Module
=============

Defines and trains different machine learning models for Bitcoin volatility prediction.

Available Models:
- BaselineModel: Persistence forecast (previous day's volatility)
- LinearRegressionModel: Linear weighted relationships
- RandomForestModel: Ensemble of decision trees
- XGBoostModel: Gradient boosting (requires xgboost package)
- LSTMModel: Deep learning for sequences (requires tensorflow)

Example:
    from models import RandomForestModel, train_all_models, evaluate_model
    
    # Train single model
    model = RandomForestModel(n_estimators=400)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Or train all models at once
    results = train_all_models(X_train, y_train, X_test, y_test, feature_cols)
"""

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Try importing XGBoost and LSTM dependencies
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠ TensorFlow/Keras not available for LSTM. Install with: pip install tensorflow")


def time_split(df, test_size=0.2):
    """
    Split data chronologically (important for time series).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to split
    test_size : float
        Fraction of data to use for testing
        
    Returns:
    --------
    tuple
        (train_df, test_df)
    """
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate regression model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of model for display
        
    Returns:
    --------
    dict
        Dictionary with MAE, RMSE, R2 metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n[{model_name}]")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r2:.4f}")
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


class BaselineModel:
    """
    Baseline model: predicts next-day volatility = today's volatility (persistence).
    """
    def __init__(self):
        self.name = "Baseline (Persistence)"
    
    def fit(self, X_train, y_train):
        """No training needed for baseline."""
        pass
    
    def predict(self, X_test):
        """Return the volatility_t column as prediction."""
        if isinstance(X_test, pd.DataFrame) and "volatility_t" in X_test.columns:
            return X_test["volatility_t"].values
        else:
            raise ValueError("Baseline requires 'volatility_t' feature in X_test")


class LinearRegressionModel:
    """
    Linear Regression model for volatility prediction.
    """
    def __init__(self):
        self.name = "Linear Regression"
        self.model = LinearRegression()
    
    def fit(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get coefficient magnitudes as feature importance."""
        coefficients = np.abs(self.model.coef_)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": coefficients
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df


class RandomForestModel:
    """
    Random Forest model for volatility prediction.
    """
    def __init__(self, n_estimators=400, max_depth=8, random_state=42):
        self.name = "Random Forest"
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the model."""
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df


class XGBoostModel:
    """
    XGBoost model for volatility prediction.
    """
    def __init__(self, n_estimators=600, max_depth=4, learning_rate=0.05, random_state=42):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.name = "XGBoost"
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state
        )
    
    def fit(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the model."""
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df


class LSTMModel:
    """
    LSTM model for time series volatility prediction.
    """
    def __init__(self, lookback=10, lstm_units=50, dropout=0.2, epochs=50, batch_size=32):
        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow/Keras not installed. Install with: pip install tensorflow")
        
        self.name = "LSTM"
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def _create_sequences(self, X, y):
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.lookback):
            X_seq.append(X[i:(i + self.lookback)])
            y_seq.append(y[i + self.lookback])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X_train, y_train):
        """Train the LSTM model."""
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Build model
        self.model = Sequential([
            LSTM(self.lstm_units, activation='relu', return_sequences=True, 
                 input_shape=(self.lookback, X_train.shape[1])),
            Dropout(self.dropout),
            LSTM(self.lstm_units // 2, activation='relu'),
            Dropout(self.dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Train
        print(f"Training LSTM with {len(X_seq)} sequences...")
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        # Scale test data
        X_scaled = self.scaler_X.transform(X_test)
        
        # Create sequences (note: we'll lose lookback samples)
        predictions = []
        for i in range(len(X_scaled) - self.lookback + 1):
            seq = X_scaled[i:(i + self.lookback)].reshape(1, self.lookback, -1)
            pred_scaled = self.model.predict(seq, verbose=0)[0, 0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
        
        # Pad predictions to match X_test length
        # Use mean prediction for first lookback samples
        mean_pred = np.mean(predictions) if predictions else 0
        full_predictions = [mean_pred] * (self.lookback - 1) + predictions
        
        return np.array(full_predictions)


def train_all_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Train all available models and return results.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    feature_names : list of feature names
    
    Returns:
    --------
    dict
        Dictionary with model objects, predictions, and metrics
    """
    results = {}
    
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # 1. Baseline
    print("\n1. Training Baseline Model...")
    baseline = BaselineModel()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_model(y_test.values, baseline_pred, "Baseline")
    results["Baseline"] = {
        "model": baseline,
        "predictions": baseline_pred,
        "metrics": baseline_metrics
    }
    
    # 2. Linear Regression
    print("\n2. Training Linear Regression...")
    lr = LinearRegressionModel()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = evaluate_model(y_test.values, lr_pred, "Linear Regression")
    results["Linear Regression"] = {
        "model": lr,
        "predictions": lr_pred,
        "metrics": lr_metrics,
        "importance": lr.get_feature_importance(feature_names)
    }
    
    # 3. Random Forest
    print("\n3. Training Random Forest...")
    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate_model(y_test.values, rf_pred, "Random Forest")
    results["Random Forest"] = {
        "model": rf,
        "predictions": rf_pred,
        "metrics": rf_metrics,
        "importance": rf.get_feature_importance(feature_names)
    }
    
    # 4. XGBoost (if available)
    if XGB_AVAILABLE:
        print("\n4. Training XGBoost...")
        xgb = XGBoostModel()
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_metrics = evaluate_model(y_test.values, xgb_pred, "XGBoost")
        results["XGBoost"] = {
            "model": xgb,
            "predictions": xgb_pred,
            "metrics": xgb_metrics,
            "importance": xgb.get_feature_importance(feature_names)
        }
    else:
        print("\n4. XGBoost not available (skipping)")
    
    # 5. LSTM (if available)
    if LSTM_AVAILABLE:
        print("\n5. Training LSTM...")
        try:
            lstm = LSTMModel(lookback=10, epochs=50)
            lstm.fit(X_train, y_train)
            lstm_pred = lstm.predict(X_test)
            lstm_metrics = evaluate_model(y_test.values, lstm_pred, "LSTM")
            results["LSTM"] = {
                "model": lstm,
                "predictions": lstm_pred,
                "metrics": lstm_metrics
            }
        except Exception as e:
            print(f"⚠ LSTM training failed: {e}")
    else:
        print("\n5. TensorFlow/Keras not available (skipping LSTM)")
    
    print("\n" + "="*60)
    print("✓ MODEL TRAINING COMPLETE")
    print("="*60)
    
    return results


def get_best_model(results):
    """
    Identify best model based on RMSE.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from train_all_models
        
    Returns:
    --------
    str
        Name of best model
    """
    best_model = None
    best_rmse = float('inf')
    
    for model_name, result in results.items():
        rmse = result["metrics"]["RMSE"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_name
    
    return best_model


if __name__ == "__main__":
    # Test models
    from data_preparation import load_or_fetch_data
    from feature_engineering import create_all_features, get_feature_columns
    
    print("Loading data...")
    df = load_or_fetch_data("BTCUSDT", "2019-01-01")
    df_features = create_all_features(df)
    
    feature_cols = get_feature_columns()
    X = df_features[feature_cols]
    y = df_features["volatility_t_plus_1"]
    
    train_df, test_df = time_split(df_features, test_size=0.2)
    X_train = train_df[feature_cols]
    y_train = train_df["volatility_t_plus_1"]
    X_test = test_df[feature_cols]
    y_test = test_df["volatility_t_plus_1"]
    
    results = train_all_models(X_train, y_train, X_test, y_test, feature_cols)
    
    best = get_best_model(results)
    print(f"\n🏆 Best Model: {best}")

