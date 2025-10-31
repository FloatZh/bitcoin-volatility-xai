"""
Main Pipeline - Bitcoin Volatility Prediction with XAI
=======================================================

Orchestrates the entire workflow: data → features → models → evaluation → visualization

This is the main entry point for running the complete Bitcoin volatility prediction pipeline.
It coordinates all modules to provide a comprehensive analysis from raw data to model explainability.

Pipeline Steps:
1. Data Preparation: Fetch and clean BTC/USDT data from Binance
2. Feature Engineering: Create 50+ technical indicators
3. Data Visualization: Generate exploration plots (optional)
4. Train/Test Split: Time-based splitting to prevent data leakage
5. Model Training: Train multiple models (Baseline, LR, RF, XGBoost, LSTM)
6. Model Comparison: Compare performance metrics across models
7. Result Visualization: Create prediction and importance plots (optional)
8. SHAP Explainability: Compute SHAP values for model interpretation (optional)
9. Save Results: Export predictions, metrics, and visualizations

Usage:
    # Basic run
    python main.py
    
    # With visualizations
    python main.py --visualize
    
    # Full analysis with SHAP
    python main.py --visualize --shap
    
    # Custom parameters
    python main.py --start-date 2020-01-01 --test-size 0.3 --visualize

For more examples, see README.md or run: python main.py --help
"""

__version__ = "2.0.0"
__author__ = "Bitcoin Volatility XAI Project"

import argparse
from pathlib import Path
import pandas as pd

from data_preparation import load_or_fetch_data
from feature_engineering import create_all_features, get_feature_columns
from models import time_split, train_all_models, get_best_model
from visualization import (
    create_data_exploration_report,
    plot_predictions,
    plot_model_comparison,
    plot_feature_importance,
    plot_shap_summary
)


def main(args):
    """
    Main pipeline execution.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("\n" + "="*70)
    print(" "*15 + "BITCOIN VOLATILITY PREDICTION WITH XAI")
    print("="*70)
    
    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    df = load_or_fetch_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        force_refresh=args.refresh
    )
    
    print(f"\nDataset info:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total days: {len(df)}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    df_features = create_all_features(df)
    feature_cols = get_feature_columns()
    
    print(f"\nFeature summary:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Usable samples: {len(df_features)}")
    
    # ========================================================================
    # STEP 3: DATA VISUALIZATION (Optional)
    # ========================================================================
    if args.visualize:
        print("\n" + "="*70)
        print("STEP 3: DATA VISUALIZATION")
        print("="*70)
        
        vis_dir = Path(args.output_dir) / "data_exploration"
        create_data_exploration_report(df, df_features, feature_cols, output_dir=str(vis_dir))
    
    # ========================================================================
    # STEP 4: PREPARE TRAIN/TEST SPLIT
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("="*70)
    
    X = df_features[feature_cols]
    y = df_features["volatility_t_plus_1"]
    
    train_df, test_df = time_split(df_features, test_size=args.test_size)
    X_train = train_df[feature_cols]
    y_train = train_df["volatility_t_plus_1"]
    X_test = test_df[feature_cols]
    y_test = test_df["volatility_t_plus_1"]
    
    print(f"\nSplit summary:")
    print(f"  Training samples: {len(X_train)} ({(1-args.test_size)*100:.0f}%)")
    print(f"  Test samples:     {len(X_test)} ({args.test_size*100:.0f}%)")
    print(f"  Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  Test date range:  {test_df['date'].min()} to {test_df['date'].max()}")
    
    # ========================================================================
    # STEP 5: MODEL TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING & EVALUATION")
    print("="*70)
    
    results = train_all_models(X_train, y_train, X_test, y_test, feature_cols)
    
    # ========================================================================
    # STEP 6: MODEL COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: MODEL COMPARISON")
    print("="*70)
    
    # Extract metrics for comparison
    metrics_dict = {name: res["metrics"] for name, res in results.items()}
    
    print("\n📊 Performance Summary:")
    print("-" * 70)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
    print("-" * 70)
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<20} {metrics['MAE']:<12.6f} {metrics['RMSE']:<12.6f} {metrics['R2']:<10.4f}")
    print("-" * 70)
    
    best_model_name = get_best_model(results)
    print(f"\n🏆 Best Model (lowest RMSE): {best_model_name}")
    print(f"   RMSE: {results[best_model_name]['metrics']['RMSE']:.6f}")
    
    # ========================================================================
    # STEP 7: VISUALIZATIONS
    # ========================================================================
    if args.visualize:
        print("\n" + "="*70)
        print("STEP 7: MODEL VISUALIZATIONS")
        print("="*70)
        
        vis_dir = Path(args.output_dir) / "model_results"
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Model comparison
        print("\n1. Creating model comparison plot...")
        plot_model_comparison(metrics_dict, save_path=vis_dir / "model_comparison.png")
        
        # Predictions for each model
        for model_name, result in results.items():
            print(f"\n2. Creating predictions plot for {model_name}...")
            plot_predictions(
                y_test.values,
                result["predictions"],
                test_df["date"].values,
                model_name=model_name,
                save_path=vis_dir / f"predictions_{model_name.lower().replace(' ', '_')}.png"
            )
        
        # Feature importance for best model
        if "importance" in results[best_model_name]:
            print(f"\n3. Creating feature importance plot for {best_model_name}...")
            plot_feature_importance(
                results[best_model_name]["importance"],
                top_n=20,
                save_path=vis_dir / f"feature_importance_{best_model_name.lower().replace(' ', '_')}.png"
            )
        
        print(f"\n✓ All model visualizations saved to '{vis_dir}/'")
    
    # ========================================================================
    # STEP 8: EXPLAINABILITY (SHAP) - Optional
    # ========================================================================
    if args.shap and best_model_name in ["Random Forest", "XGBoost"]:
        print("\n" + "="*70)
        print("STEP 8: MODEL EXPLAINABILITY (SHAP)")
        print("="*70)
        
        try:
            import shap
            
            best_model = results[best_model_name]["model"]
            
            # Use subset of training data for faster computation
            background_size = min(500, len(X_train))
            X_background = X_train.tail(background_size)
            
            print(f"\nComputing SHAP values for {best_model_name}...")
            print(f"Using {background_size} background samples...")
            
            explainer = shap.TreeExplainer(best_model.model)
            shap_values = explainer.shap_values(X_background)
            
            if args.visualize:
                vis_dir = Path(args.output_dir) / "model_results"
                plot_shap_summary(shap_values, X_background, 
                                 save_path=vis_dir / "shap_summary.png")
            
            print("\n✓ SHAP analysis complete")
            
        except ImportError:
            print("\n⚠ SHAP library not installed. Install with: pip install shap")
        except Exception as e:
            print(f"\n⚠ SHAP analysis failed: {e}")
    
    # ========================================================================
    # STEP 9: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 9: SAVING RESULTS")
    print("="*70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save predictions
    predictions_df = test_df[["date", "close", "volatility_t", "volatility_t_plus_1"]].copy()
    for model_name, result in results.items():
        predictions_df[f"pred_{model_name.lower().replace(' ', '_')}"] = result["predictions"]
    
    predictions_file = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"\n✓ Predictions saved to: {predictions_file}")
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_file = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_file)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save feature importance (for best model)
    if "importance" in results[best_model_name]:
        importance_file = output_dir / f"feature_importance_{best_model_name.lower().replace(' ', '_')}.csv"
        results[best_model_name]["importance"].to_csv(importance_file, index=False)
        print(f"✓ Feature importance saved to: {importance_file}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test RMSE: {results[best_model_name]['metrics']['RMSE']:.6f}")
    print(f"Test R²:   {results[best_model_name]['metrics']['R2']:.4f}")
    print(f"\nAll results saved to: {output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bitcoin Volatility Prediction with Explainable AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default settings
  python main.py
  
  # Full run with all visualizations
  python main.py --visualize --shap
  
  # Custom date range and test size
  python main.py --start-date 2020-01-01 --test-size 0.3 --visualize
  
  # Force refresh data from API
  python main.py --refresh --visualize
        """
    )
    
    # Data arguments
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                       help="Trading pair symbol (default: BTCUSDT)")
    parser.add_argument("--start-date", type=str, default="2019-01-01",
                       help="Start date for data (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date for data (YYYY-MM-DD)")
    parser.add_argument("--refresh", action="store_true",
                       help="Force refresh data from API")
    
    # Model arguments
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Fraction of data for testing (default: 0.2)")
    
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true",
                       help="Generate all visualizations")
    parser.add_argument("--shap", action="store_true",
                       help="Compute SHAP values for explainability")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results (default: results)")
    
    args = parser.parse_args()
    
    main(args)

