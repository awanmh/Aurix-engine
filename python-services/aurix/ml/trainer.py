"""
AURIX ML Training Engine

Implements gradient boosting models with probability calibration.
Includes Audit Fixes #2 (calibration), #3 (asymmetric cost), #5 (PSI monitoring).
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from model training."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    log_loss: float
    train_samples: int
    feature_importance: Dict[str, float]
    calibration_error: float
    psi: Optional[float] = None


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    min_samples: int = 200
    training_window_days: int = 30
    model_type: str = "lightgbm"
    psi_threshold: float = 0.25


@dataclass
class ModelWrapper:
    """Wrapper for trained model with metadata."""
    model: Any
    calibrator: Optional[IsotonicRegression]
    version: str
    trained_at: datetime
    metrics: TrainingMetrics
    feature_names: List[str]
    training_distribution: Dict[str, Any]


class PSICalculator:
    """
    Population Stability Index calculator.
    
    Implements Audit Fix #5: Model degradation detection.
    """
    
    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate PSI between expected and actual distributions.
        
        Args:
            expected: Training/reference distribution
            actual: Current/production distribution
            bins: Number of bins for bucketing
            
        Returns:
            PSI value (>0.25 indicates significant shift)
        """
        # Create bins from expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate proportions
        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]
        
        # Normalize
        expected_pct = expected_counts / len(expected) + 1e-10
        actual_pct = actual_counts / len(actual) + 1e-10
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
    
    @staticmethod
    def interpret_psi(psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return "STABLE - No significant change"
        elif psi < 0.25:
            return "MODERATE - Some change, monitor closely"
        else:
            return "CRITICAL - Significant shift, consider retraining"


class AsymmetricCostFunction:
    """
    Asymmetric sample weighting for imbalanced costs.
    
    Implements Audit Fix #3: Asymmetric cost function.
    """
    
    def __init__(
        self,
        loss_weight: float = 2.0,  # Losses hurt more than wins help
        consecutive_loss_penalty: float = 0.1  # Additional penalty per consecutive loss
    ):
        self.loss_weight = loss_weight
        self.consecutive_penalty = consecutive_loss_penalty
    
    def compute_weights(
        self,
        y: np.ndarray,
        net_returns: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute sample weights with asymmetric costs.
        
        Args:
            y: Binary labels (1=win, 0=loss)
            net_returns: Optional net returns for magnitude-based weighting
            
        Returns:
            Array of sample weights
        """
        weights = np.ones(len(y))
        
        # Apply loss weight
        weights[y == 0] *= self.loss_weight
        
        # If net returns provided, weight by magnitude
        if net_returns is not None:
            # Larger losses get higher weights
            loss_mask = y == 0
            if loss_mask.any():
                loss_magnitudes = np.abs(net_returns[loss_mask])
                # Normalize to 0-1 range and add baseline
                if loss_magnitudes.max() > 0:
                    normalized = loss_magnitudes / loss_magnitudes.max()
                    weights[loss_mask] *= (1 + normalized)
        
        # Normalize
        weights = weights / weights.mean()
        
        return weights


class MLTrainer:
    """
    ML training engine with calibration and monitoring.
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model_type: str = "lightgbm",
        model_dir: str = "models",
        psi_threshold: float = 0.25
    ):
        """
        Initialize trainer.
        
        Args:
            config: Optional TrainingConfig object
            model_type: 'lightgbm' or 'xgboost' (overridden by config if provided)
            model_dir: Directory to save models
            psi_threshold: Threshold for PSI alerts (overridden by config if provided)
        """
        if config:
            self.config = config
            self.model_type = config.model_type
            self.psi_threshold = config.psi_threshold
        else:
            self.config = TrainingConfig(
                model_type=model_type,
                psi_threshold=psi_threshold
            )
            self.model_type = model_type
            self.psi_threshold = psi_threshold
            
        self.model_dir = model_dir
        self.psi_calculator = PSICalculator()
        self.cost_function = AsymmetricCostFunction()
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Current models
        self.long_model: Optional[ModelWrapper] = None
        self.short_model: Optional[ModelWrapper] = None
    
    def train(
        self,
        X: Any,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        direction: str = "LONG",
        net_returns: Optional[np.ndarray] = None,
        reference_X: Optional[np.ndarray] = None
    ) -> ModelWrapper:
        """
        Train a model with calibration.
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
            y: Labels
            feature_names: Feature column names (optional if X is DataFrame)
            direction: LONG or SHORT
            net_returns: Net returns for asymmetric weighting
            reference_X: Reference feature distribution for PSI
            
        Returns:
            Trained ModelWrapper
        """
        import pandas as pd
        
        # Handle DataFrame input
        if hasattr(X, 'columns'):
            if feature_names is None:
                feature_names = X.columns.tolist()
            # Convert all columns to numeric, coercing errors to NaN
            X_numeric = X.apply(pd.to_numeric, errors='coerce')
            X_arr = X_numeric.values.astype(np.float64)
        else:
            X_arr = X
            # Ensure numpy array is float64
            if X_arr.dtype.kind == 'O':  # Object dtype
                logger.warning("X has object dtype, converting to float64...")
                try:
                    X_arr = X_arr.astype(np.float64)
                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to convert X to float64: {e}")
                    raise
            elif not np.issubdtype(X_arr.dtype, np.floating):
                X_arr = X_arr.astype(np.float64)
        
        # Handle NaN and inf values
        nan_mask = np.isnan(X_arr)
        inf_mask = np.isinf(X_arr)
        if np.any(nan_mask) or np.any(inf_mask):
            nan_count = int(np.sum(nan_mask))
            inf_count = int(np.sum(inf_mask))
            logger.warning(f"X contains {nan_count} NaN and {inf_count} inf values, replacing with 0")
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
            
        if feature_names is None:
            raise ValueError("feature_names must be provided if X is not a DataFrame")
            
        logger.info(f"Training {direction} model with {len(X_arr)} samples...")
        
        # Calculate sample weights
        sample_weights = self.cost_function.compute_weights(y, net_returns)
        
        # Create base model
        if self.model_type == "lightgbm":
            model = self._create_lightgbm()
        else:
            model = self._create_xgboost()
        
        # Add feature names to LightGBM
        fit_params = {}
        if self.model_type == "lightgbm":
            fit_params['feature_name'] = feature_names
            
        # Train
        model.fit(X_arr, y, sample_weight=sample_weights, **fit_params)
        
        # Calibrate probabilities (Audit Fix #2)
        calibrator = self._calibrate_probabilities(model, X_arr, y)
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, calibrator, X_arr, y, feature_names)
        
        # Calculate PSI if reference provided
        if reference_X is not None:
            ref_arr = reference_X.values if hasattr(reference_X, 'columns') else reference_X
            for i, fname in enumerate(feature_names):
                psi = self.psi_calculator.calculate_psi(ref_arr[:, i], X_arr[:, i])
                if psi > self.psi_threshold:
                    logger.warning(f"High PSI for {fname}: {psi:.3f}")
            
            # Overall feature PSI (mean across features)
            psi_values = [
                self.psi_calculator.calculate_psi(ref_arr[:, i], X_arr[:, i])
                for i in range(X_arr.shape[1])
            ]
            metrics.psi = np.mean(psi_values)
        
        # Create version string
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{direction.lower()}"
        
        # Save training distribution for future PSI
        training_dist = {
            'means': X_arr.mean(axis=0).tolist(),
            'stds': X_arr.std(axis=0).tolist(),
            'percentiles': {
                '10': np.percentile(X_arr, 10, axis=0).tolist(),
                '50': np.percentile(X_arr, 50, axis=0).tolist(),
                '90': np.percentile(X_arr, 90, axis=0).tolist()
            }
        }
        
        wrapper = ModelWrapper(
            model=model,
            calibrator=calibrator,
            version=version,
            trained_at=datetime.now(),
            metrics=metrics,
            feature_names=feature_names,
            training_distribution=training_dist
        )
        
        # Store model
        if direction == "LONG":
            self.long_model = wrapper
        else:
            self.short_model = wrapper
        
        # Save to disk
        self._save_model(wrapper, direction)
        
        logger.info(f"Model trained: {version}")
        logger.info(f"  Accuracy: {metrics.accuracy:.3f}")
        logger.info(f"  AUC-ROC: {metrics.auc_roc:.3f}")
        logger.info(f"  Calibration Error: {metrics.calibration_error:.3f}")
        
        return wrapper
    
    def _create_lightgbm(self):
        """Create LightGBM classifier."""
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=20,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    def _create_xgboost(self):
        """Create XGBoost classifier."""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def _calibrate_probabilities(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> IsotonicRegression:
        """
        Calibrate model probabilities using Isotonic Regression.
        
        Implements Audit Fix #2: Probability calibration.
        """
        # Get uncalibrated probabilities
        raw_probs = model.predict_proba(X)[:, 1]
        
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(raw_probs, y)
        
        return calibrator
    
    def _calculate_metrics(
        self,
        model: Any,
        calibrator: IsotonicRegression,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> TrainingMetrics:
        """Calculate comprehensive training metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, log_loss
        )
        
        # Predictions
        y_pred = model.predict(X)
        raw_probs = model.predict_proba(X)[:, 1]
        calibrated_probs = calibrator.predict(raw_probs)
        
        # Core metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y, raw_probs)
        logloss = log_loss(y, calibrated_probs)
        
        # Calibration error (ECE - Expected Calibration Error)
        calibration_error = self._calculate_ece(calibrated_probs, y)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
        else:
            importance = {}
        
        return TrainingMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            log_loss=logloss,
            train_samples=len(X),
            feature_importance=importance,
            calibration_error=calibration_error
        )
    
    def _calculate_ece(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_confidence = probs[mask].mean()
                bin_accuracy = y[mask].mean()
                ece += mask.sum() * abs(bin_accuracy - bin_confidence)
        
        return ece / len(probs)
    
    def predict(
        self,
        X: np.ndarray,
        direction: str = "LONG"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make calibrated predictions.
        
        Args:
            X: Feature matrix
            direction: LONG or SHORT
            
        Returns:
            Tuple of (raw_probabilities, calibrated_probabilities)
        """
        model_wrapper = self.long_model if direction == "LONG" else self.short_model
        
        if model_wrapper is None:
            raise ValueError(f"No {direction} model loaded")
        
        if hasattr(X, 'values'):
            X = X.values
            
        raw_probs = model_wrapper.model.predict_proba(X)[:, 1]
        calibrated_probs = model_wrapper.calibrator.predict(raw_probs)
        
        return raw_probs, calibrated_probs
    
    def check_psi(
        self,
        X: np.ndarray,
        direction: str = "LONG"
    ) -> Dict[str, float]:
        """
        Check PSI for current data against training distribution.
        
        Args:
            X: Current feature matrix
            direction: LONG or SHORT
            
        Returns:
            Dict with PSI values per feature and overall status
        """
        model_wrapper = self.long_model if direction == "LONG" else self.short_model
        
        if model_wrapper is None or 'percentiles' not in model_wrapper.training_distribution:
            return {'status': 'NO_REFERENCE'}
        
        # Reconstruct reference distribution from percentiles
        p10 = np.array(model_wrapper.training_distribution['percentiles']['10'])
        p50 = np.array(model_wrapper.training_distribution['percentiles']['50'])
        p90 = np.array(model_wrapper.training_distribution['percentiles']['90'])
        
        psi_values = {}
        critical_features = []
        
        for i, fname in enumerate(model_wrapper.feature_names):
            # Simple PSI approximation using percentile comparison
            current_p50 = np.percentile(X[:, i], 50)
            drift = abs(current_p50 - p50[i]) / (abs(p50[i]) + 1e-10)
            
            psi_values[fname] = drift
            if drift > self.psi_threshold:
                critical_features.append(fname)
        
        overall_psi = np.mean(list(psi_values.values()))
        
        return {
            'overall_psi': overall_psi,
            'status': 'CRITICAL' if overall_psi > self.psi_threshold else 'OK',
            'critical_features': critical_features,
            'feature_psi': psi_values
        }
    
    def _save_model(self, wrapper: ModelWrapper, direction: str):
        """Save model to disk."""
        path = os.path.join(self.model_dir, f"{direction.lower()}_model.pkl")
        with open(path, 'wb') as f:
            pickle.dump(wrapper, f)
        
        # Save metadata separately as JSON
        meta_path = os.path.join(self.model_dir, f"{direction.lower()}_model_meta.json")
        meta = {
            'version': wrapper.version,
            'trained_at': wrapper.trained_at.isoformat(),
            'accuracy': wrapper.metrics.accuracy,
            'auc_roc': wrapper.metrics.auc_roc,
            'train_samples': wrapper.metrics.train_samples,
            'calibration_error': wrapper.metrics.calibration_error
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def load_model(self, direction: str) -> Optional[ModelWrapper]:
        """Load model from disk."""
        path = os.path.join(self.model_dir, f"{direction.lower()}_model.pkl")
        
        if not os.path.exists(path):
            logger.warning(f"No saved {direction} model found at {path}")
            return None
        
        with open(path, 'rb') as f:
            wrapper = pickle.load(f)
        
        if direction == "LONG":
            self.long_model = wrapper
        else:
            self.short_model = wrapper
        
        logger.info(f"Loaded {direction} model: {wrapper.version}")
        return wrapper
    
    def get_model_version(self, direction: str = "LONG") -> Optional[str]:
        """Get current model version."""
        wrapper = self.long_model if direction == "LONG" else self.short_model
        return wrapper.version if wrapper else None
