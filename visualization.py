"""
Visualization Module
====================

Comprehensive visualization tools for cryptocurrency data exploration and model evaluation.

This module provides publication-ready plots for:
- Price history and volume analysis
- Volatility patterns and distributions
- Feature correlations and distributions
- Model predictions (time series and scatter)
- Model performance comparisons
- Feature importance rankings
- SHAP explainability analysis

All plots can be displayed interactively or saved to files.

Example:
    from visualization import plot_price_history, plot_predictions
    
    # Display price chart
    plot_price_history(df)
    
    # Save predictions plot
    plot_predictions(y_test, y_pred, dates, model_name="XGBoost", 
                     save_path="results/predictions.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_price_history(df, save_path=None):
    """
    Plot Bitcoin price history with volume.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date, close, and volume columns
    save_path : str, optional
        Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Price
    ax1.plot(df["date"], df["close"], linewidth=1.5, color='#F7931A')
    ax1.set_ylabel("BTC Price (USDT)", fontsize=12, fontweight='bold')
    ax1.set_title("Bitcoin Price History", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Volume
    ax2.bar(df["date"], df["volume"], color='#4A90E2', alpha=0.6, width=0.8)
    ax2.set_ylabel("Volume", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved price history to {save_path}")
    
    plt.show()


def plot_volatility_analysis(df, save_path=None):
    """
    Plot volatility patterns over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with volatility features
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Volatility over time
    axes[0, 0].plot(df["date"], df["volatility_t"], linewidth=1, alpha=0.7, color='#E74C3C')
    axes[0, 0].plot(df["date"], df["volatility_roll_20"], linewidth=2, 
                     label='20-day MA', color='#3498DB')
    axes[0, 0].set_title("Realized Volatility Over Time", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Volatility (HL/Open)", fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Volatility distribution
    axes[0, 1].hist(df["volatility_t"], bins=50, color='#9B59B6', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df["volatility_t"].mean(), color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {df["volatility_t"].mean():.4f}')
    axes[0, 1].axvline(df["volatility_t"].median(), color='green', linestyle='--', 
                        linewidth=2, label=f'Median: {df["volatility_t"].median():.4f}')
    axes[0, 1].set_title("Volatility Distribution", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Volatility", fontsize=10)
    axes[0, 1].set_ylabel("Frequency", fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Volatility vs Returns
    axes[1, 0].scatter(df["ret_1d"], df["volatility_t"], alpha=0.3, s=10, color='#16A085')
    axes[1, 0].set_title("Volatility vs Daily Returns", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Daily Return", fontsize=10)
    axes[1, 0].set_ylabel("Volatility", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Volatility regime over time
    if "volatility_regime" in df.columns:
        regime_colors = {"low": "#2ECC71", "medium": "#F39C12", "high": "#E74C3C"}
        for regime, color in regime_colors.items():
            mask = df["volatility_regime"] == regime
            axes[1, 1].scatter(df.loc[mask, "date"], df.loc[mask, "close"], 
                              label=regime.capitalize(), alpha=0.5, s=5, color=color)
        axes[1, 1].set_title("Price by Volatility Regime", fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel("Price (USDT)", fontsize=10)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved volatility analysis to {save_path}")
    
    plt.show()


def plot_feature_correlations(df, feature_cols, save_path=None):
    """
    Plot correlation heatmap of features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    feature_cols : list
        List of feature column names to include
    save_path : str, optional
        Path to save the figure
    """
    # Select subset of features for readability
    important_features = [
        "volatility_t", "volatility_roll_5", "volatility_roll_20",
        "ret_1d", "ret_5d", "rsi_14", "macd", "bb_width",
        "vol_change_1d", "vol_ratio_5", "return_vol_10", "momentum_10"
    ]
    
    available_features = [f for f in important_features if f in df.columns]
    
    corr_matrix = df[available_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved correlation heatmap to {save_path}")
    
    plt.show()


def plot_feature_distributions(df, feature_cols, n_features=12, save_path=None):
    """
    Plot distributions of top features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    feature_cols : list
        List of feature column names
    n_features : int
        Number of features to plot
    save_path : str, optional
        Path to save the figure
    """
    features_to_plot = feature_cols[:n_features]
    n_cols = 4
    n_rows = int(np.ceil(len(features_to_plot) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, feature in enumerate(features_to_plot):
        if feature in df.columns:
            axes[idx].hist(df[feature].dropna(), bins=30, color='#3498DB', 
                          alpha=0.7, edgecolor='black')
            axes[idx].set_title(feature, fontsize=10, fontweight='bold')
            axes[idx].set_ylabel("Frequency", fontsize=8)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Feature Distributions", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature distributions to {save_path}")
    
    plt.show()


def plot_predictions(y_true, y_pred, dates, model_name="Model", save_path=None):
    """
    Plot actual vs predicted volatility.
    
    Parameters:
    -----------
    y_true : array-like
        Actual volatility values
    y_pred : array-like
        Predicted volatility values
    dates : array-like
        Dates corresponding to predictions
    model_name : str
        Name of the model for title
    save_path : str, optional
        Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series comparison
    ax1.plot(dates, y_true, label='Actual', linewidth=1.5, alpha=0.8, color='#2C3E50')
    ax1.plot(dates, y_pred, label='Predicted', linewidth=1.5, alpha=0.8, color='#E74C3C')
    ax1.set_title(f"{model_name}: Actual vs Predicted Volatility", 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Volatility", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.4, s=20, color='#3498DB')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction')
    
    ax2.set_xlabel("Actual Volatility", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Predicted Volatility", fontsize=12, fontweight='bold')
    ax2.set_title("Prediction Scatter Plot", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved predictions plot to {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """
    Compare performance metrics across multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metric dicts as values
        e.g., {"Model1": {"MAE": 0.01, "RMSE": 0.02, "R2": 0.85}, ...}
    save_path : str, optional
        Path to save the figure
    """
    models = list(results_dict.keys())
    metrics = ["MAE", "RMSE", "R2"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        bars = axes[idx].bar(models, values, color='#3498DB', alpha=0.7, edgecolor='black')
        
        # Highlight best model
        if metric == "R2":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_color('#2ECC71')
        
        axes[idx].set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison to {save_path}")
    
    plt.show()


def plot_feature_importance(importance_df, top_n=20, save_path=None):
    """
    Plot feature importance from tree-based models.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save the figure
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_features)), top_features["importance"], color='#E74C3C', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance", fontsize=12, fontweight='bold')
    plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance to {save_path}")
    
    plt.show()


def plot_shap_summary(shap_values, X, save_path=None):
    """
    Plot SHAP summary (if shap is available).
    
    Parameters:
    -----------
    shap_values : array
        SHAP values from explainer
    X : pd.DataFrame
        Feature dataframe
    save_path : str, optional
        Path to save the figure
    """
    try:
        import shap
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved SHAP summary to {save_path}")
        
        plt.show()
    except ImportError:
        print("⚠ SHAP not installed. Skipping SHAP visualization.")
    except Exception as e:
        print(f"⚠ Error creating SHAP plot: {e}")


def create_data_exploration_report(df, df_features, feature_cols, output_dir="visualizations"):
    """
    Create a comprehensive set of visualizations for data exploration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with OHLCV
    df_features : pd.DataFrame
        Data with all features
    feature_cols : list
        List of feature column names
    output_dir : str
        Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING DATA EXPLORATION VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Price History...")
    plot_price_history(df, save_path=output_path / "01_price_history.png")
    
    print("\n2. Volatility Analysis...")
    plot_volatility_analysis(df_features, save_path=output_path / "02_volatility_analysis.png")
    
    print("\n3. Feature Correlations...")
    plot_feature_correlations(df_features, feature_cols, 
                              save_path=output_path / "03_feature_correlations.png")
    
    print("\n4. Feature Distributions...")
    plot_feature_distributions(df_features, feature_cols, 
                               save_path=output_path / "04_feature_distributions.png")
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to '{output_dir}/' directory")
    print("="*60)


if __name__ == "__main__":
    # Test visualizations
    from data_preparation import load_or_fetch_data
    from feature_engineering import create_all_features, get_feature_columns
    
    print("Loading data for visualization test...")
    df = load_or_fetch_data("BTCUSDT", "2019-01-01")
    df_features = create_all_features(df)
    feature_cols = get_feature_columns()
    
    create_data_exploration_report(df, df_features, feature_cols)

