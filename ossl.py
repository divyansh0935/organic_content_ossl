


import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings('ignore')

def check_and_install_packages():
    """Check and install required packages if missing."""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'xgboost': 'xgboost',
        'soilspecdata': 'soilspecdata'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f" {package} is available")
        except ImportError:
            missing_packages.append(pip_name)
            print(f" {package} is missing")
    
    if missing_packages:
        print(f"\n Missing packages: {missing_packages}")
        print("Please install with:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

def setup_environment():
    """Setup SSL certificates and matplotlib backend."""
    try:
        import ssl
        import certifi
        ssl._create_default_https_context = ssl._create_unverified_context
        ssl._create_default_https_context().load_verify_locations(certifi.where())
        print(" SSL certificates configured")
    except Exception as e:
        print(f"SSL setup warning: {e}")
    
    try:
        # Set matplotlib backend
        matplotlib.use('Agg')
        print(" Matplotlib backend set to: Agg")
    except Exception as e:
        print(f" Matplotlib setup failed: {e}")
        sys.exit(1)

def download_ossl_data():
    """Download OSSL data with progress tracking."""
    try:
        from soilspecdata.datasets.ossl import get_ossl
        print(" soilspecdata imported successfully")
    except ImportError as e:
        print(f" Failed to import soilspecdata: {e}")
        print("Please install with: pip install soilspecdata")
        sys.exit(1)
    
    print("\n Downloading OSSL dataset...")
    start_time = time.time()
    
    try:
        ossl = get_ossl(force_download=False)
        download_time = time.time() - start_time
        print(f"\nData downloaded in {download_time:.1f} seconds")
        return ossl
    except Exception as e:
        print(f" Download failed: {e}")
        sys.exit(1)

def find_target_column(ossl):
    """Find and validate target column for organic carbon."""
    df = ossl.df
    oc_keywords = ['oc_', 'orgc', 'organic_c', 'soc_']
    target_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in oc_keywords):
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_data.notna().sum() > 100:
                target_col = col
                break
    
    if not target_col:
        print(" No valid organic carbon column found")
        sys.exit(1)
    
    print(f" Using target column: {target_col}")
    return target_col

def preprocess_spectra(X):
    """Apply spectral preprocessing techniques."""
    from scipy.signal import savgol_filter
    
    
    X_smoothed = savgol_filter(X, window_length=11, polyorder=2, deriv=0)
    

    X_deriv = savgol_filter(X, window_length=11, polyorder=2, deriv=1)
    
 
    X_processed = np.hstack((X_smoothed, X_deriv))
    
    return X_processed

def process_data(ossl, target_col):
    """Process and align data with enhanced preprocessing."""
    print("\n Processing data with spectral preprocessing...")
    
    try:
        
        X, y, _ = ossl.get_aligned_data(target_cols=[target_col])
        y = pd.Series(y).astype(float)
        
      
        X_processed = preprocess_spectra(X)
        
      
        valid_mask = ~pd.isnull(y)
        X_clean = X_processed[valid_mask]
        y_clean = y[valid_mask]
        
        print(f" Final dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        return X_clean, y_clean
        
    except Exception as e:
        print(f" Data processing error: {e}")
        sys.exit(1)

def feature_engineering(X, y):
    """Apply dimensionality reduction and feature selection."""
    print("\nApplying feature engineering...")
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
  
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced features: {X_pca.shape[1]} components")
    

    selector = SelectKBest(mutual_info_regression, k=min(100, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"Selected top {X_selected.shape[1]} features")
    
    return X_pca, X_selected, pca, selector, scaler

def train_model(X, y):
    """Train model with hyperparameter tuning and cross-validation."""
    print("\nTraining model with hyperparameter tuning...")
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    

    model = GradientBoostingRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    print(" Running grid search...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f" Best parameters: {grid_search.best_params_}")
    
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nFinal Model Performance:")
    print(f"Training Set: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
    print(f"Test Set:     RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
    
    return best_model, X_train, X_test, y_train, y_test, y_pred_test, test_rmse, test_r2

def create_plot(y_test, y_pred, rmse, r2):
    """Create prediction plot with enhanced visuals."""
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    
    # Scatter plot with regression line
    ax = sns.regplot(
        x=y_test, 
        y=y_pred, 
        scatter_kws={'alpha':0.6, 's':50},
        line_kws={'color':'red', 'lw':2}
    )
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
    
    # Labels and titles
    plt.xlabel("True Organic Carbon (%)", fontsize=12)
    plt.ylabel("Predicted Organic Carbon (%)", fontsize=12)
    plt.title(f"Soil Organic Carbon Prediction\nR² = {r2:.3f} | RMSE = {rmse:.3f}", fontsize=14)
    
    # Add stats annotations
    plt.annotate(
        f'Test Samples: {len(y_test)}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}',
        xy=(0.05, 0.85),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    filename = "oc_prediction_plot.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f" Plot saved as {filename}")
    return filename

def save_results(model, X, y, rmse, r2, feature_names=None):
    """Save model and results with comprehensive metadata."""
    results = {
        'model': model,
        'features': feature_names if feature_names else [f"feat_{i}" for i in range(X.shape[1])],
        'rmse': rmse,
        'r2': r2,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    joblib.dump(results, "soil_oc_model.joblib")
    print("Model and results saved")
    
    # Save feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': results['features'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance.to_csv("feature_importance.csv", index=False)
        print(" Feature importance saved")

def main():
    """Main execution pipeline."""
    print(" Enhanced Soil Organic Carbon Prediction Pipeline")
    print("=" * 60)
    start_time = time.time()
    
    try:
        # Setup environment
        check_and_install_packages()
        setup_environment()
        
        # Download and prepare data
        ossl = download_ossl_data()
        target_col = find_target_column(ossl)
        X, y = process_data(ossl, target_col)
        
        # Feature engineering
        X_pca, X_selected, pca, selector, scaler = feature_engineering(X, y)
        
        # Train model (using PCA-reduced features)
        model, X_train, X_test, y_train, y_test, y_pred, rmse, r2 = train_model(X_pca, y)
        
        # Create and save outputs
        plot_file = create_plot(y_test, y_pred, rmse, r2)
        save_results(model, X_pca, y, rmse, r2)
        
        # Final report
        total_time = time.time() - start_time
        print(f"\n Pipeline completed in {total_time:.1f} seconds")
        print(f"Final model R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
    except Exception as e:
        print(f"\n Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
