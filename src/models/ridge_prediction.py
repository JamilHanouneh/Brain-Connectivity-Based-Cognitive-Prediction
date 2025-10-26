"""
Ridge regression prediction models with nested cross-validation.

Implements the prediction framework from Dhamala et al. (2021).
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RidgePredictionModel:
    """
    Ridge regression model for cognitive prediction.
    
    Attributes
    ----------
    alpha : float
        Regularization parameter
    model : Ridge
        Trained ridge regression model
    scaler : StandardScaler
        Feature scaler
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        normalize_features: bool = True,
        fit_intercept: bool = True
    ):
        """
        Initialize ridge regression model.
        
        Parameters
        ----------
        alpha : float
            Regularization strength
        normalize_features : bool
            Whether to normalize features
        fit_intercept : bool
            Whether to fit intercept
        """
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.fit_intercept = fit_intercept
        
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self.scaler = StandardScaler() if normalize_features else None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgePredictionModel':
        """
        Fit the model.
        
        Parameters
        ----------
        X : np.ndarray
            Features (n_samples x n_features)
        y : np.ndarray
            Target values (n_samples,)
        
        Returns
        -------
        self
        """
        if self.normalize_features:
            X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Features (n_samples x n_features)
        
        Returns
        -------
        np.ndarray
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.normalize_features:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def get_coefficients(self) -> np.ndarray:
        """
        Get model coefficients.
        
        Returns
        -------
        np.ndarray
            Model coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.coef_


def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: List[float],
    n_outer_folds: int = 3,
    n_inner_folds: int = 3,
    n_iterations: int = 5,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform nested cross-validation for model selection and evaluation.
    
    Parameters
    ----------
    X : np.ndarray
        Features (n_samples x n_features)
    y : np.ndarray
        Target values (n_samples,)
    alpha_range : list of float
        Regularization parameters to test
    n_outer_folds : int
        Number of outer CV folds for evaluation
    n_inner_folds : int
        Number of inner CV folds for hyperparameter tuning
    n_iterations : int
        Number of CV iterations with different random splits
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': Array of predictions
        - 'actuals': Array of actual values
        - 'best_alphas': Selected alpha values
        - 'fold_scores': R² scores for each fold
        - 'coefficients': Model coefficients from each fold
    """
    logger.info(f"Starting nested CV with {n_iterations} iterations, "
                f"{n_outer_folds} outer folds, {n_inner_folds} inner folds")
    
    all_predictions = []
    all_actuals = []
    all_best_alphas = []
    all_fold_scores = []
    all_coefficients = []
    
    for iteration in range(n_iterations):
        logger.debug(f"CV Iteration {iteration + 1}/{n_iterations}")
        
        # Outer CV for evaluation
        outer_cv = KFold(
            n_splits=n_outer_folds,
            shuffle=True,
            random_state=random_seed + iteration
        )
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            X_train_outer, X_test = X[train_idx], X[test_idx]
            y_train_outer, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            inner_cv = KFold(
                n_splits=n_inner_folds,
                shuffle=True,
                random_state=random_seed + iteration + fold_idx
            )
            
            # Use RidgeCV for efficient alpha selection
            ridge_cv = RidgeCV(
                alphas=alpha_range,
                cv=inner_cv,
                scoring='r2'
            )
            
            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_outer)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model with cross-validated alpha selection
            ridge_cv.fit(X_train_scaled, y_train_outer)
            
            # Make predictions
            y_pred = ridge_cv.predict(X_test_scaled)
            
            # Calculate R² score
            from sklearn.metrics import r2_score
            fold_score = r2_score(y_test, y_pred)
            
            # Store results
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            all_best_alphas.append(ridge_cv.alpha_)
            all_fold_scores.append(fold_score)
            all_coefficients.append(ridge_cv.coef_)
    
    results = {
        'predictions': np.array(all_predictions),
        'actuals': np.array(all_actuals),
        'best_alphas': np.array(all_best_alphas),
        'fold_scores': np.array(all_fold_scores),
        'coefficients': np.array(all_coefficients)
    }
    
    logger.info(f"Nested CV complete. Mean R² = {np.mean(all_fold_scores):.4f} "
                f"± {np.std(all_fold_scores):.4f}")
    
    return results


def train_test_prediction(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: List[float],
    n_splits: int = 100,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform repeated train/test splits for robust evaluation.
    
    This follows the paper's approach of 100 random 80/20 splits.
    
    Parameters
    ----------
    X : np.ndarray
        Features (n_samples x n_features)
    y : np.ndarray
        Target values (n_samples,)
    alpha_range : list of float
        Regularization parameters to test
    n_splits : int
        Number of random train/test splits (paper uses 100)
    test_size : float
        Proportion of data for testing (paper uses 0.2)
    random_seed : int
        Random seed
    
    Returns
    -------
    dict
        Dictionary containing results for each split
    """
    logger.info(f"Starting train/test prediction with {n_splits} splits...")
    
    r2_scores = []
    correlations = []
    predictions_all = []
    actuals_all = []
    coefficients_all = []
    best_alphas = []
    
    for split_idx in tqdm(range(n_splits), desc="Train/Test Splits"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed + split_idx
        )
        
        # Inner CV for alpha selection on training data
        ridge_cv = RidgeCV(
            alphas=alpha_range,
            cv=3,
            scoring='r2'
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        ridge_cv.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = ridge_cv.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import r2_score
        from scipy.stats import pearsonr
        
        r2 = r2_score(y_test, y_pred)
        corr, _ = pearsonr(y_test, y_pred)
        
        # Store results
        r2_scores.append(r2)
        correlations.append(corr)
        predictions_all.append(y_pred)
        actuals_all.append(y_test)
        coefficients_all.append(ridge_cv.coef_)
        best_alphas.append(ridge_cv.alpha_)
    
    results = {
        'r2_scores': np.array(r2_scores),
        'correlations': np.array(correlations),
        'predictions': predictions_all,
        'actuals': actuals_all,
        'coefficients': np.array(coefficients_all),
        'best_alphas': np.array(best_alphas),
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_correlation': np.mean(correlations),
        'std_correlation': np.std(correlations)
    }
    
    logger.info(f"Train/test prediction complete:")
    logger.info(f"  Mean R² = {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    logger.info(f"  Mean correlation = {results['mean_correlation']:.4f} ± {results['std_correlation']:.4f}")
    
    return results
