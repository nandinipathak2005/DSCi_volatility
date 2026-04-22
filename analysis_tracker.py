from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
import pandas as pd
import numpy as np
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisStep:
    def __init__(self, step_name: str, description: str, timestamp=None):
        self.step_name = step_name
        self.description = description
        self.timestamp = timestamp or datetime.now()
        self.input_data = {}
        self.output_data = {}
        self.parameters = {}
        self.metrics = {}
        self.data_snapshots = {}  # Raw vs cleaned data
        self.feature_vectors = {}  # Feature engineering outputs
        self.statistical_outputs = {}  # Statistical calculations
        self.model_outputs = {}  # ML/model predictions
        self.decision_inputs = {}  # Scoring and weights
        
    def to_dict(self):
        return {
            'step_name': self.step_name,
            'description': self.description,
            'timestamp': self.timestamp.strftime('%H:%M:%S'),
            'input_data': self._serialize_data(self.input_data),
            'output_data': self._serialize_data(self.output_data),
            'parameters': self.parameters,
            'metrics': self._serialize_data(self.metrics),
            'data_snapshots': self._serialize_data(self.data_snapshots),
            'feature_vectors': self._serialize_data(self.feature_vectors),
            'statistical_outputs': self._serialize_data(self.statistical_outputs),
            'model_outputs': self._serialize_data(self.model_outputs),
            'decision_inputs': self._serialize_data(self.decision_inputs)
        }
    
    def _serialize_data(self, data):
        """Convert complex data types to serializable format"""
        if isinstance(data, dict):
            return {k: self._serialize_value(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_value(item) for item in data]
        else:
            return self._serialize_value(data)
    
    def _serialize_value(self, value):
        """Convert individual values to serializable format"""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            return value.to_dict() if isinstance(value, pd.Series) else value.to_dict('records')
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return str(value)  # Fallback to string representation

class AnalysisTracker:
    def __init__(self):
        self.steps = []
        self.current_ticker = None
        self.current_analysis_session = {}
        self.final_recommendation = {}
        self.raw_data_snapshot = None
        self.cleaned_data_snapshot = None
        
    def start_ticker_analysis(self, ticker: str, raw_data: Optional[pd.DataFrame] = None):
        self.current_ticker = ticker
        self.steps = []
        self.raw_data_snapshot = raw_data
        self.current_analysis_session = {
            'ticker': ticker,
            'start_time': datetime.now(),
            'steps': [],
            'raw_data_shape': raw_data.shape if raw_data is not None else None
        }
        
    def add_step(self, step_name, description, input_data=None, output_data=None, parameters=None, metrics=None,
                 data_snapshots=None, feature_vectors=None, statistical_outputs=None, 
                 model_outputs=None, decision_inputs=None):
        step = AnalysisStep(step_name, description)
        step.input_data = input_data or {}
        step.output_data = output_data or {}
        step.parameters = parameters or {}
        step.metrics = metrics or {}
        step.data_snapshots = data_snapshots or {}
        step.feature_vectors = feature_vectors or {}
        step.statistical_outputs = statistical_outputs or {}
        step.model_outputs = model_outputs or {}
        step.decision_inputs = decision_inputs or {}
        
        self.steps.append(step)
        if self.current_analysis_session:
            self.current_analysis_session['steps'].append(step)
    
    def add_data_collection_step(self, ticker: str, raw_data: pd.DataFrame, data_source: str, 
                                interval: str, period: str):
        """Log raw data collection"""
        self.add_step(
            step_name="Data Collection",
            description=f"Fetching raw market data for {ticker}",
            output_data={
                'ticker': ticker,
                'data_source': data_source,
                'interval': interval,
                'period': period,
                'data_shape': raw_data.shape,
                'columns': list(raw_data.columns),
                'date_range': f"{raw_data.index.min()} to {raw_data.index.max()}" if len(raw_data) > 0 else "No data"
            },
            data_snapshots={
                'raw_data': raw_data
            },
            metrics={
                'total_rows': len(raw_data),
                'total_columns': len(raw_data.columns),
                'missing_values': raw_data.isnull().sum().sum(),
                'missing_percentage': (raw_data.isnull().sum().sum() / (len(raw_data) * len(raw_data.columns)) * 100) if len(raw_data) > 0 else 0
            }
        )
    
    def add_data_preprocessing_step(self, raw_data: pd.DataFrame, cleaned_data: pd.DataFrame,
                                   preprocessing_steps: List[str], removed_outliers: int = 0,
                                   filled_missing: int = 0):
        """Log data preprocessing with before/after snapshots"""
        self.cleaned_data_snapshot = cleaned_data
        self.add_step(
            step_name="Data Preprocessing",
            description="Cleaning and preprocessing raw market data",
            input_data={
                'raw_shape': raw_data.shape,
                'raw_missing': raw_data.isnull().sum().sum()
            },
            output_data={
                'cleaned_shape': cleaned_data.shape,
                'cleaned_missing': cleaned_data.isnull().sum().sum(),
                'preprocessing_steps': preprocessing_steps,
                'data_quality_improvement': (raw_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum()) / raw_data.isnull().sum().sum() * 100 if raw_data.isnull().sum().sum() > 0 else 100
            },
            data_snapshots={
                'raw_data_sample': raw_data.head(5),
                'cleaned_data_sample': cleaned_data.head(5),
                'raw_vs_cleaned_comparison': {
                    'rows_removed': len(raw_data) - len(cleaned_data),
                    'outliers_removed': removed_outliers,
                    'missing_filled': filled_missing
                }
            },
            metrics={
                'data_retention_rate': (len(cleaned_data) / len(raw_data) * 100) if len(raw_data) > 0 else 0,
                'outlier_percentage': (removed_outliers / len(raw_data) * 100) if len(raw_data) > 0 else 0,
                'missing_fill_rate': (filled_missing / raw_data.isnull().sum().sum() * 100) if raw_data.isnull().sum().sum() > 0 else 100
            }
        )
    
    def add_feature_engineering_step(self, input_df: pd.DataFrame, feature_matrix: pd.DataFrame,
                                    calculated_features: List[str], feature_parameters: Dict[str, Any]):
        """Log feature engineering with feature vectors"""
        self.add_step(
            step_name="Feature Engineering",
            description="Creating statistical features and technical indicators",
            input_data={
                'input_shape': input_df.shape,
                'input_columns': list(input_df.columns)
            },
            output_data={
                'feature_matrix_shape': feature_matrix.shape,
                'calculated_features': calculated_features,
                'feature_count': len(calculated_features)
            },
            feature_vectors={
                'feature_matrix': feature_matrix,
                'feature_descriptions': {feature: f"Calculated {feature} indicator" for feature in calculated_features},
                'feature_statistics': {
                    feature: {
                        'mean': feature_matrix[feature].mean(),
                        'std': feature_matrix[feature].std(),
                        'min': feature_matrix[feature].min(),
                        'max': feature_matrix[feature].max(),
                        'missing': feature_matrix[feature].isnull().sum()
                    } for feature in calculated_features if feature in feature_matrix.columns
                }
            },
            parameters=feature_parameters,
            metrics={
                'total_features': len(calculated_features),
                'feature_completeness': (1 - feature_matrix.isnull().sum().sum() / (len(feature_matrix) * len(feature_matrix.columns))) * 100
            }
        )
    
    def add_statistical_analysis_step(self, statistical_measures: Dict[str, float], 
                                     distribution_tests: Dict[str, Dict[str, float]],
                                     stationarity_tests: Dict[str, Dict[str, float]],
                                     time_series_properties: Dict[str, Any]):
        """Log comprehensive statistical analysis"""
        self.add_step(
            step_name="Statistical Analysis",
            description="Comprehensive statistical characterization of market data",
            statistical_outputs={
                'measures': statistical_measures,
                'distribution_tests': distribution_tests,
                'stationarity_tests': stationarity_tests,
                'time_series_properties': time_series_properties
            },
            metrics={
                'hurst_exponent': statistical_measures.get('hurst', 0.5),
                'volatility': statistical_measures.get('volatility', 0),
                'skewness': statistical_measures.get('skewness', 0),
                'kurtosis': statistical_measures.get('kurtosis', 0),
                'is_stationary': stationarity_tests.get('adf', {}).get('p_value', 1) < 0.05,
                'is_normal': distribution_tests.get('jarque_bera', {}).get('p_value', 1) < 0.05
            }
        )
    
    def add_model_training_step(self, model_name: str, model_type: str, training_data: pd.DataFrame,
                               model_parameters: Dict[str, Any], training_metrics: Dict[str, float],
                               model_artifacts: Dict[str, Any]):
        """Log model training and outputs"""
        self.add_step(
            step_name=f"Model Training: {model_name}",
            description=f"Training {model_type} model for {model_name}",
            input_data={
                'training_data_shape': training_data.shape,
                'training_period': f"{training_data.index.min()} to {training_data.index.max()}" if len(training_data) > 0 else "No data"
            },
            model_outputs={
                'model_name': model_name,
                'model_type': model_type,
                'model_parameters': model_parameters,
                'training_metrics': training_metrics,
                'model_artifacts': model_artifacts
            },
            parameters=model_parameters,
            metrics=training_metrics
        )
    
    def add_regime_detection_step(self, regime: str, confidence: float, detection_metrics: Dict[str, float], 
                                 decision_logic: str, regime_features: Dict[str, Any]):
        """Enhanced regime detection logging"""
        metrics_combined = detection_metrics.copy()
        metrics_combined['confidence'] = confidence
        
        self.add_step(
            step_name="Regime Detection",
            description="Detecting current market regime using statistical rules",
            model_outputs={
                'detected_regime': regime,
                'confidence': confidence,
                'regime_features': regime_features,
                'decision_logic': decision_logic
            },
            decision_inputs={
                'regime_thresholds': {
                    'hurst_trending': 0.6,
                    'hurst_mean_reverting': 0.4,
                    'volatility_high': 0.02,
                    'volatility_extreme': 0.05,
                    'volume_z_threshold': 2.0,
                    'extreme_volume_z': 3.0
                },
                'detection_metrics': detection_metrics
            },
            metrics=metrics_combined
        )
    
    def add_portfolio_optimization_step(self, portfolio_weights: Dict[str, float], 
                                       optimization_method: str, constraints: Dict[str, Any],
                                       risk_metrics: Dict[str, float], expected_returns: Dict[str, float]):
        """Log portfolio optimization results"""
        self.add_step(
            step_name="Portfolio Optimization",
            description=f"Optimizing portfolio using {optimization_method}",
            model_outputs={
                'optimal_weights': portfolio_weights,
                'optimization_method': optimization_method,
                'constraints': constraints
            },
            decision_inputs={
                'risk_metrics': risk_metrics,
                'expected_returns': expected_returns,
                'optimization_objective': 'max_sharpe' if optimization_method == 'max_sharpe' else 'min_volatility'
            },
            metrics={
                'portfolio_volatility': risk_metrics.get('portfolio_volatility', 0),
                'portfolio_sharpe': risk_metrics.get('portfolio_sharpe', 0),
                'expected_portfolio_return': sum(w * r for w, r in zip(portfolio_weights.values(), expected_returns.values()))
            }
        )
    
    def add_recommendation_engine_step(self, recommendation_scores: Dict[str, float],
                                      decision_weights: Dict[str, float], threshold_rules: Dict[str, Any],
                                      final_recommendation: str, confidence_score: float):
        """Log recommendation engine decision process"""
        self.add_step(
            step_name="Recommendation Engine",
            description="Generating trading recommendation based on multi-factor analysis",
            decision_inputs={
                'recommendation_scores': recommendation_scores,
                'decision_weights': decision_weights,
                'threshold_rules': threshold_rules,
                'scoring_components': {
                    'hurst_score': recommendation_scores.get('hurst_score', 0),
                    'volatility_score': recommendation_scores.get('volatility_score', 0),
                    'sharpe_score': recommendation_scores.get('sharpe_score', 0),
                    'rsi_score': recommendation_scores.get('rsi_score', 0)
                }
            },
            output_data={
                'final_recommendation': final_recommendation,
                'confidence_score': confidence_score,
                'total_score': sum(recommendation_scores.values())
            },
            metrics={
                'total_score': sum(recommendation_scores.values()),
                'confidence': confidence_score,
                'recommendation': final_recommendation
            }
        )
        
    def add_data_processing_step(self, processed_rows, removed_outliers, missing_values_filled, processing_methods):
        self.add_step(
            step_name="Data Processing",
            description="Raw market data preprocessing and cleaning",
            output_data={
                'processed_rows': processed_rows,
                'removed_outliers': removed_outliers,
                'missing_values_filled': missing_values_filled,
                'methods_applied': processing_methods
            },
            metrics={
                'data_quality': (processed_rows - removed_outliers) / processed_rows if processed_rows > 0 else 0,
                'outliers_percentage': (removed_outliers / processed_rows * 100) if processed_rows > 0 else 0
            }
        )
        
    def add_statistical_test_step(self, test_name, test_result, interpretation):
        self.add_step(
            step_name=f"Statistical Test: {test_name}",
            description=f"Running {test_name} to determine market characteristics",
            output_data={
                'test_name': test_name,
                'test_result': test_result,
                'interpretation': interpretation
            },
            metrics=test_result
        )
        
    def add_recommendation_step(self, action, reasoning, position_size=0, risk_level="MEDIUM", supporting_factors=None):
        self.add_step(
            step_name="Final Recommendation",
            description="Generating trading recommendation based on all analysis",
            output_data={
                'action': action,
                'position_size': position_size,
                'risk_level': risk_level,
                'supporting_factors': supporting_factors or []
            },
            metrics={'confidence': 0.8}
        )
        
        self.final_recommendation = {
            'ticker': self.current_ticker,
            'action': action,
            'reasoning': reasoning,
            'position_size': position_size,
            'risk_level': risk_level,
            'supporting_factors': supporting_factors or [],
            'timestamp': datetime.now()
        }
        
    def get_detailed_breakdown(self):
        start_time = self.current_analysis_session.get('start_time', datetime.now())
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'ticker': self.current_ticker,
            'all_steps': [step.to_dict() for step in self.steps],
            'final_recommendation': self.final_recommendation,
            'total_steps': len(self.steps),
            'analysis_duration': duration
        }

def get_tracker():
    if "tracker" not in st.session_state:
        st.session_state.tracker = AnalysisTracker()
    return st.session_state.tracker