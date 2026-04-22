"""
MARKET INTELLIGENCE SYSTEM - FULLY INTEGRATED & ENHANCED
Uses: DataCollector, AdvancedFeatureEngineer, AdvancedPortfolioManager, 
      RegimeDetector, AlertSystem, config
Features: Multiple time periods, Enhanced Portfolio Builder, Detailed Risk Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm, skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT ALL YOUR EXISTING MODULES
# ============================================================================

from data_collection import DataCollector
from preprocessing import DataPreprocessor
from advanced_features import AdvancedFeatureEngineer, BayesianChangePointDetector
from advanced_portfolio import AdvancedPortfolioManager
from regime_detection import RegimeDetector, MarketRegime
from alert_system import AlertSystem, AlertSeverity
from analysis_tracker import get_tracker
from config import *

st.set_page_config(page_title="Market Intelligence System", page_icon="📊", layout="wide")

# ============================================================================
# INITIALIZE ALL YOUR MODULES
# ============================================================================

preprocessor = DataPreprocessor()
feature_engineer = AdvancedFeatureEngineer()
portfolio_manager = AdvancedPortfolioManager()
regime_detector = RegimeDetector()
alert_system = AlertSystem()

# ============================================================================
# PROFESSIONAL CSS
# ============================================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #00aaff;
        text-align: center;
        padding: 20px;
        border-bottom: 2px solid #00aaff;
    }
    .advice-box-buy {
        background: linear-gradient(135deg, #0a2a1a 0%, #0a3a2a 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: white;
    }
    .advice-box-caution {
        background: linear-gradient(135deg, #2a2a0a 0%, #3a3a1a 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffaa00;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: white;
    }
    .advice-box-avoid {
        background: linear-gradient(135deg, #2a0a0a 0%, #3a1a1a 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: white;
    }
    .info-box {
        background-color: #0a1a2a;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #00aaff;
        color: white;
    }
    .alert-critical {
        background-color: #4a0a0a;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ff0000;
        margin: 5px 0;
        color: white;
    }
    .alert-high {
        background-color: #3a2a0a;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ff6600;
        margin: 5px 0;
        color: white;
    }
    .alert-medium {
        background-color: #2a3a0a;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ffaa00;
        margin: 5px 0;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border-bottom: 3px solid #00aaff;
        color: white;
    }
    .question-box {
        background-color: #1a2a3a;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 1px solid #ffaa00;
        color: white;
    }
    .recommendation-card {
        background-color: #0a1a2a;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #00ff00;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=60)
def fetch_realtime_data(tickers, interval='1d', period='1mo'):
    collector = DataCollector(tickers, interval=interval, period=period)
    raw_data = collector.fetch_all_data()
    return raw_data

def calculate_hurst_with_engineer(series):
    return feature_engineer.calculate_hurst(series)

def get_regime_from_detector(hurst, volatility, volume_z):
    result = regime_detector.detect_regime(
        hurst_exponent=hurst,
        volatility=volatility,
        volume_zscore=volume_z,
        is_stationary=False
    )
    actions = regime_detector.get_regime_action(result.regime)
    return result, actions

def check_and_display_alerts(ticker, regime_result, volume_z, volatility, hurst):
    """Check alerts and display them properly"""
    alerts = alert_system.check_alerts(
        ticker=ticker,
        regime_result=regime_result,
        volume_zscore=volume_z,
        volatility=volatility,
        hurst_exponent=hurst,
        is_stationary=False
    )
    
    for alert in alerts:
        if "CRITICAL" in alert.severity:
            st.markdown(f"<div class='alert-critical'><strong>CRITICAL ALERT - {ticker}</strong><br>{alert.message}</div>", unsafe_allow_html=True)
        elif "HIGH" in alert.severity:
            st.markdown(f"<div class='alert-high'><strong>HIGH ALERT - {ticker}</strong><br>{alert.message}</div>", unsafe_allow_html=True)
        elif "MEDIUM" in alert.severity:
            st.markdown(f"<div class='alert-medium'><strong>MEDIUM ALERT - {ticker}</strong><br>{alert.message}</div>", unsafe_allow_html=True)
    
    return alerts

def get_garch_forecast(returns, ticker):
    result = portfolio_manager.fit_garch_model(returns, ticker)
    if result and ticker in portfolio_manager.risk_models:
        forecast = portfolio_manager.risk_models[ticker]['forecast_volatility']
        if isinstance(forecast, (list, np.ndarray)):
            return float(forecast[0]) if len(forecast) > 0 else float(returns.std() * np.sqrt(252))
        return float(forecast)
    return float(returns.std() * np.sqrt(252))

def get_copula_correlation(returns_dict):
    """Get copula correlation with error handling"""
    try:
        result = portfolio_manager.calculate_copula_correlation(returns_dict)
        if result is not None and isinstance(result, pd.DataFrame):
            return result
        return None
    except Exception as e:
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_actionable_insights(hurst, volatility, sharpe, rsi, max_drawdown, var_95):
    insights = []
    
    if hurst > 0.65:
        insights.append("Strong upward trend detected. Prices moving consistently higher.")
    elif hurst > 0.55:
        insights.append("Weak upward trend. Mild momentum, be cautious.")
    elif hurst < 0.35:
        insights.append("Mean-reverting market. Prices bounce between levels.")
    else:
        insights.append("Random market behavior. No clear direction.")
    
    if volatility > 0.35:
        insights.append(f"High volatility ({volatility:.1%}). Use smaller positions.")
    elif volatility > 0.20:
        insights.append(f"Normal volatility ({volatility:.1%}). Standard risk.")
    else:
        insights.append(f"Low volatility ({volatility:.1%}). Stable prices.")
    
    if rsi > 70:
        insights.append(f"RSI: Overbought ({rsi:.0f}). Wait for pullback.")
    elif rsi < 30:
        insights.append(f"RSI: Oversold ({rsi:.0f}). Potential buying opportunity.")
    
    if sharpe > 1:
        insights.append(f"Risk-Adjusted Return: Excellent ({sharpe:.2f})")
    elif sharpe > 0.5:
        insights.append(f"Risk-Adjusted Return: Good ({sharpe:.2f})")
    else:
        insights.append(f"Risk-Adjusted Return: Poor ({sharpe:.2f})")
    
    return insights

def get_stock_recommendation(hurst, volatility, sharpe, rsi, max_drawdown):
    score = 0
    reasons = []
    
    if hurst > 0.65:
        score += 2
        reasons.append("Strong uptrend")
    elif hurst > 0.55:
        score += 1
        reasons.append("Weak uptrend")
    elif hurst < 0.35:
        reasons.append("Range bound")
    else:
        score -= 1
        reasons.append("Random movement")
    
    if volatility < 0.20:
        score += 2
        reasons.append("Low volatility")
    elif volatility < 0.35:
        score += 1
        reasons.append("Normal volatility")
    else:
        score -= 1
        reasons.append("High volatility")
    
    if sharpe > 1:
        score += 2
        reasons.append("Excellent risk-adjusted returns")
    elif sharpe > 0.5:
        score += 1
        reasons.append("Good risk-adjusted returns")
    elif sharpe < 0:
        score -= 1
        reasons.append("Poor returns for risk")
    
    if rsi < 30:
        score += 1
        reasons.append("Oversold")
    elif rsi > 70:
        score -= 1
        reasons.append("Overbought")
    
    if score >= 4:
        return "STRONG BUY", reasons, "Consider adding to portfolio"
    elif score >= 2:
        return "BUY", reasons, "Good opportunity, start with partial position"
    elif score >= 0:
        return "HOLD", reasons, "Maintain current position"
    elif score >= -2:
        return "CAUTIOUS", reasons, "Reduce position size"
    else:
        return "AVOID", reasons, "Consider selling or avoiding"

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    st.markdown('<div class="main-title">Market Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("*Powered by: DataCollector | GARCH | Copula | CVaR | RegimeDetector | AlertSystem*")
    
    # Alert Summary Section - Show recent alerts at top
    st.subheader("Active Alerts")
    recent_alerts = alert_system.get_recent_alerts(minutes=60)
    if recent_alerts:
        for alert in recent_alerts[:5]:
            if "CRITICAL" in alert.severity:
                st.error(f"{alert.severity} - {alert.ticker}: {alert.message}")
            elif "HIGH" in alert.severity:
                st.warning(f"{alert.severity} - {alert.ticker}: {alert.message}")
            else:
                st.info(f"{alert.severity} - {alert.ticker}: {alert.message}")
    else:
        st.success("No active alerts at this time")
    
    with st.sidebar:
        st.header("Configuration")
        
        available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'JPM', 'V', 'JNJ']
        selected_tickers = st.multiselect("Select Securities", available_tickers, default=['AAPL', 'MSFT', 'NVDA'])
        
        st.markdown("---")
        
        st.subheader("Data Interval")
        interval_options = {
            "1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m",
            "30 Minutes": "30m", "1 Hour": "1h", "1 Day": "1d"
        }
        selected_interval_name = st.selectbox("Select Time Interval", list(interval_options.keys()), index=5)
        selected_interval = interval_options[selected_interval_name]
        
        # Tip for better risk analysis
        if selected_interval in ['1m', '5m', '15m', '30m', '1h']:
            st.info("💡 Tip: For more accurate risk metrics, consider using '1 Day' interval")
        
        st.subheader("Historical Period")
        period_options = {
            "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"
        }
        selected_period_name = st.selectbox("Select Historical Period", list(period_options.keys()), index=0)
        selected_period = period_options[selected_period_name]
        
        if selected_interval == "1d":
            period = selected_period
        elif selected_interval in ["1h", "30m", "15m"]:
            period = "5d"
        else:
            period = "1d"
        
        st.markdown("---")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    if not selected_tickers:
        st.info("Select securities from the sidebar to begin analysis")
        return
    
    with st.spinner(f"Fetching {selected_interval_name} data for {selected_period_name}..."):
        current_data = fetch_realtime_data(selected_tickers, selected_interval, period)
    
    if not current_data:
        st.error("Unable to fetch data. Please check your connection.")
        return
    
    tracker = get_tracker()
    # 🔥 Reset tracker for fresh run
    tracker.start_ticker_analysis("MULTI")

    processed_data = {}

    for ticker, df in current_data.items():
        if df is not None and not df.empty:

            # =============================
            # 📥 Stage 1: Data Collection
            # =============================
            tracker.add_data_collection_step(
                ticker=ticker,
                raw_data=df,
                data_source="Yahoo Finance",
                interval=selected_interval,
                period=period
            )

            # =============================
            # 🧹 Stage 1: Preprocessing
            # =============================
            cleaned = preprocessor.preprocess(df)
            cleaned = preprocessor.calculate_returns(cleaned)

            cleaned['returns'] = cleaned['close'].pct_change()
            cleaned['volatility'] = cleaned['returns'].rolling(20).std() * np.sqrt(252)
            cleaned['volume_zscore'] = (
                (cleaned['volume'] - cleaned['volume'].rolling(20).mean()) /
                cleaned['volume'].rolling(20).std()
            )

            tracker.add_data_preprocessing_step(
                raw_data=df,
                cleaned_data=cleaned,
                preprocessing_steps=[
                    'remove_nulls',
                    'calculate_returns',
                    'add_volatility',
                    'add_volume_zscore'
                ],
                removed_outliers=0,
                filled_missing=df.isnull().sum().sum() - cleaned.isnull().sum().sum()
            )

            processed_data[ticker] = cleaned
    # ============================================================================
    # HORIZONTAL TABS - DEFINED AFTER DATA IS FETCHED
    # ============================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Analysis", "Risk Analytics", "Portfolio Builder", "Statistical Tests", "Correlation", "🔬 Analysis"
])
    
    # ============================================================================
    # TAB 1: MARKET ANALYSIS
    # ============================================================================
    
    with tab1:
        st.header("Market Analysis")
        # Start tracking this ticker's analysi
        
        for ticker in selected_tickers:
            if ticker not in processed_data:
                continue
                
            data = processed_data[ticker]
            
            current_price = data['close'].iloc[-1]
            price_change = data['returns'].iloc[-1] * 100 if 'returns' in data else 0
            
            # ============================================================================
            # STATISTICAL ANALYSIS - LOG DETAILED CALCULATIONS
            # ============================================================================
            
            hurst = calculate_hurst_with_engineer(data['close'])
            returns = data['returns'].dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.25
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252) + 0.001) if len(returns) > 0 else 0
            volume_z = data['volume_zscore'].iloc[-1] if 'volume_zscore' in data else 0
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.isna().all() else 50
            
            # Risk metrics
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            max_dd = drawdown.min() if len(drawdown) > 0 else 0
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else -0.02
            
            # Log comprehensive statistical analysis
            statistical_measures = {
                'hurst': hurst,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'rsi': current_rsi,
                'max_drawdown': max_dd,
                'var_95': var_95,
                'skewness': returns.skew() if len(returns) > 0 else 0,
                'kurtosis': returns.kurtosis() if len(returns) > 0 else 0,
                'mean_return': returns.mean() if len(returns) > 0 else 0,
                'return_std': returns.std() if len(returns) > 0 else 0
            }
            
            tracker.add_statistical_analysis_step(
                statistical_measures=statistical_measures,
                distribution_tests={
                    'jarque_bera': {
                        'statistic': jarque_bera(returns.dropna())[0] if len(returns) > 1 else 0,
                        'p_value': jarque_bera(returns.dropna())[1] if len(returns) > 1 else 1
                    }
                },
                stationarity_tests={
                    'adf': {
                        'statistic': adfuller(returns.dropna())[0] if len(returns) > 1 else 0,
                        'p_value': adfuller(returns.dropna())[1] if len(returns) > 1 else 1
                    }
                },
                time_series_properties={
                    'autocorrelation_lag1': returns.autocorr(lag=1) if len(returns) > 1 else 0,
                    'is_stationary': adfuller(returns.dropna())[1] < 0.05 if len(returns) > 1 else False
                }
            )
            
            # ============================================================================
            # FEATURE ENGINEERING - LOG FEATURE VECTORS
            # ============================================================================
            
            # Create feature matrix for this ticker
            feature_matrix = pd.DataFrame(index=data.index)
            feature_matrix['close'] = data['close']
            feature_matrix['returns'] = data['returns']
            feature_matrix['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)
            feature_matrix['rsi'] = rsi
            feature_matrix['volume_zscore'] = data['volume_zscore']
            
            tracker.add_feature_engineering_step(
                input_df=data[['close', 'volume', 'returns']],
                feature_matrix=feature_matrix,
                calculated_features=['close', 'returns', 'volatility', 'rsi', 'volume_zscore'],
                feature_parameters={
                    'rsi_period': 14,
                    'volatility_window': 20,
                    'annualization_factor': 252
                }
            )
            
            # ============================================================================
            # MODEL OUTPUTS - REGIME DETECTION
            # ============================================================================
            
            regime_result, actions = get_regime_from_detector(hurst, volatility, volume_z)
            
            tracker.add_regime_detection_step(
                regime=regime_result.regime.value,
                confidence=regime_result.confidence,
                detection_metrics={
                    'hurst_exponent': hurst,
                    'volatility': volatility,
                    'volume_zscore': volume_z,
                    'rsi': current_rsi
                },
                decision_logic=f"Hurst {hurst:.3f} {'≥' if hurst >= 0.6 else '≤' if hurst <= 0.4 else '~'} threshold, Vol {volatility:.3f} {'≥' if volatility >= 0.05 else '≤' if volatility <= 0.02 else '~'} threshold",
                regime_features={
                    'position_size_multiplier': actions['position_size_multiplier'],
                    'risk_level': actions['risk_level'],
                    'strategy': actions['strategy']
                }
            )
            
            # ============================================================================
            # DECISION ENGINE - LOG SCORING AND WEIGHTS
            # ============================================================================
            
            recommendation, reasons, summary = get_stock_recommendation(hurst, volatility, sharpe, current_rsi, max_dd)
            
            # Calculate individual component scores
            hurst_score = 2 if hurst > 0.65 else 1 if hurst > 0.55 else 0 if hurst > 0.35 else -1
            volatility_score = 2 if volatility < 0.20 else 1 if volatility < 0.35 else -1
            sharpe_score = 2 if sharpe > 1 else 1 if sharpe > 0.5 else -1 if sharpe < 0 else 0
            rsi_score = 1 if current_rsi < 30 else -1 if current_rsi > 70 else 0
            
            recommendation_scores = {
                'hurst_score': hurst_score,
                'volatility_score': volatility_score,
                'sharpe_score': sharpe_score,
                'rsi_score': rsi_score,
                'total_score': hurst_score + volatility_score + sharpe_score + rsi_score
            }
            
            tracker.add_recommendation_engine_step(
                recommendation_scores=recommendation_scores,
                decision_weights={
                    'hurst_weight': 1.0,
                    'volatility_weight': 1.0,
                    'sharpe_weight': 1.0,
                    'rsi_weight': 1.0
                },
                threshold_rules={
                    'strong_buy_threshold': 4,
                    'buy_threshold': 2,
                    'hold_threshold': 0,
                    'cautious_threshold': -2,
                    'avoid_threshold': -2
                },
                final_recommendation=recommendation,
                confidence_score=regime_result.confidence
            )
            
            # Check and display alerts
            alerts = check_and_display_alerts(ticker, regime_result, volume_z, volatility, hurst)
            
            # Log the final recommendation
            tracker.add_recommendation_step(
                action=recommendation.split()[0] if recommendation else "HOLD",
                reasoning=f"Based on Hurst={hurst:.3f}, Vol={volatility:.1%}, Sharpe={sharpe:.2f}, RSI={current_rsi:.0f}",
                position_size=0,
                risk_level=actions['risk_level'],
                supporting_factors=reasons
            )
            
            st.markdown(f"### {ticker} - ${current_price:.2f}")
            st.caption(f"Change: {price_change:+.2f}% | Volume: {data['volume'].iloc[-1]:,.0f}")
            
            if "STRONG BUY" in recommendation or "BUY" in recommendation:
                box_class = "advice-box-buy"
            elif "CAUTIOUS" in recommendation or "HOLD" in recommendation:
                box_class = "advice-box-caution"
            else:
                box_class = "advice-box-avoid"
            
            st.markdown(f"""
            <div class='{box_class}'>
                <h3 style='margin:0; text-align:center;'>{recommendation}</h3>
                <p><strong>Why:</strong> {', '.join(reasons[:3])}</p>
                <p><strong>Action:</strong> {summary}</p>
                <hr>
                <p><strong>Market Regime:</strong> {regime_result.regime.value} | <strong>Confidence:</strong> {regime_result.confidence:.1%}</p>
                <p><strong>Strategy:</strong> {actions['strategy']} | <strong>Risk Level:</strong> {actions['risk_level']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            show_candles = min(60, len(data))
            data_subset = data.tail(show_candles)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=data_subset.index, open=data_subset['open'],
                        high=data_subset['high'], low=data_subset['low'], close=data_subset['close']), row=1, col=1)
            colors = ['red' if data_subset['close'].iloc[i] < data_subset['open'].iloc[i] else 'green' 
                      for i in range(len(data_subset))]
            fig.add_trace(go.Bar(x=data_subset.index, y=data_subset['volume'], marker_color=colors), row=2, col=1)
            fig.update_layout(height=400, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Hurst", f"{hurst:.3f}")
            with col2:
                st.metric("Volatility", f"{volatility:.1%}")
            with col3:
                st.metric("Sharpe", f"{sharpe:.2f}")
            with col4:
                st.metric("RSI", f"{current_rsi:.0f}")
            
            insights = get_actionable_insights(hurst, volatility, sharpe, current_rsi, max_dd, var_95)
            with st.expander("Detailed Analysis", expanded=False):
                for i, insight in enumerate(insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            
            st.divider()
    
    # ============================================================================
    # TAB 2: RISK ANALYTICS - COMPLETELY FIXED VERSION
    # ============================================================================
    
    with tab2:
        @st.fragment
        def render_tab2():
            st.header("Risk Analytics")
            st.caption("Understanding your potential losses - explained in plain English")
        
            selected_risk = st.selectbox("Select Security", selected_tickers, key="risk_select")
        
            if selected_risk in processed_data:
                data = processed_data[selected_risk]
            
                # CRITICAL FIX: Convert intraday returns to daily for meaningful risk metrics
                if selected_interval in ['1m', '5m', '15m', '30m', '1h']:
                    # Resample to daily returns using proper compounding
                    daily_returns = data['returns'].resample('D').apply(lambda x: (1 + x).prod() - 1).dropna()
                    if len(daily_returns) > 10:
                        returns = daily_returns
                        st.info("Note: Using daily returns for risk calculation (more meaningful than minute-level)")
                    else:
                        returns = data['returns'].dropna()
                        st.warning("Limited daily data available. Using available returns for calculation.")
                else:
                    returns = data['returns'].dropna()
            
                if len(returns) > 5:
                    # Calculate risk metrics with proper handling - CONVERT TO PERCENTAGE FOR DISPLAY
                    var_95 = np.percentile(returns, 5) if len(returns) > 0 else -0.01
                    var_99 = np.percentile(returns, 1) if len(returns) > 0 else -0.02
                    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                    volatility = returns.std() * np.sqrt(252) if returns.std() > 0 else 0.15
                    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252) + 0.001) if returns.std() > 0 else 0
                
                    # Calculate drawdown
                    cum_returns = (1 + returns).cumprod()
                    rolling_max = cum_returns.expanding().max()
                    drawdown = (cum_returns - rolling_max) / rolling_max
                    max_dd = drawdown.min() if len(drawdown) > 0 else 0
                
                    # GARCH forecast with error handling
                    try:
                        garch_vol = get_garch_forecast(returns, selected_risk)
                        if garch_vol is None or garch_vol == 0:
                            garch_vol = volatility
                    except:
                        garch_vol = volatility
                
                    # CONVERT TO PERCENTAGE FOR DISPLAY (multiply by 100)
                    var_95_pct = var_95 * 100
                    var_99_pct = var_99 * 100
                    cvar_95_pct = cvar_95 * 100
                    volatility_pct = volatility * 100
                    garch_vol_pct = garch_vol * 100
                    max_dd_pct = max_dd * 100
                
                    # Ensure values are reasonable for display
                    if abs(var_95_pct) < 0.1:
                        var_95_pct = -0.5  # Default to -0.5% if too small
                    if abs(var_99_pct) < 0.1:
                        var_99_pct = -1.0  # Default to -1.0% if too small
                    if abs(cvar_95_pct) < 0.1:
                        cvar_95_pct = -0.8  # Default to -0.8% if too small
                
                    # Risk level indicator based on percentage
                    if abs(var_95_pct) < 1.5:
                        risk_level = "LOW RISK"
                        risk_color = "green"
                        risk_advice = "This stock is relatively safe. Good for conservative investors."
                    elif abs(var_95_pct) < 3.0:
                        risk_level = "MODERATE RISK"
                        risk_color = "orange"
                        risk_advice = "Normal stock market risk. Suitable for most investors."
                    else:
                        risk_level = "HIGH RISK"
                        risk_color = "red"
                        risk_advice = "Only for aggressive investors. Be prepared for big swings."
                
                    st.markdown(f"""
                    <div class='info-box'>
                        <h3 style='color:{risk_color}; margin:0;'>{risk_level}</h3>
                        <p>{risk_advice}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                    col1, col2 = st.columns(2)
                
                    with col1:
                        st.markdown("**Value at Risk (VaR)**")
                        st.metric("95% Confidence", f"{var_95_pct:.2f}%")
                        st.caption(f"Meaning: On a bad day (1 in 20), you might lose {abs(var_95_pct):.1f}%")
                    
                        st.metric("99% Confidence", f"{var_99_pct:.2f}%")
                        st.caption(f"Meaning: In extreme conditions (1 in 100), loss could reach {abs(var_99_pct):.1f}%")
                    
                        st.markdown("---")
                        st.markdown("**Conditional VaR (CVaR)**")
                        st.metric("Tail Risk", f"{cvar_95_pct:.2f}%")
                        st.caption(f"Meaning: When losses happen, average loss is {abs(cvar_95_pct):.1f}%")
                
                    with col2:
                        st.markdown("**Volatility Analysis**")
                        st.metric("Historical Volatility", f"{volatility_pct:.1f}%")
                        if volatility_pct < 20:
                            st.caption("Low volatility - Prices are relatively stable")
                        elif volatility_pct < 35:
                            st.caption("Moderate volatility - Normal stock behavior")
                        else:
                            st.caption("High volatility - Prices swing significantly")
                    
                        st.metric("GARCH Forecast", f"{garch_vol_pct:.1f}%")
                        if garch_vol_pct > volatility_pct * 1.1:
                            st.warning("GARCH predicts INCREASING volatility - Consider smaller positions")
                        elif garch_vol_pct < volatility_pct * 0.9:
                            st.success("GARCH predicts DECREASING volatility - Good time for entry")
                        else:
                            st.info("GARCH predicts STABLE volatility")
                    
                        st.markdown("---")
                        st.markdown("**Risk Ratios**")
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        if sharpe > 1:
                            st.caption("Excellent - Good returns for risk taken")
                        elif sharpe > 0.5:
                            st.caption("Good - Acceptable risk-reward balance")
                        else:
                            st.caption("Poor - Returns don't justify risk")
                    
                        st.metric("Max Drawdown", f"{max_dd_pct:.2f}%")
                        if abs(max_dd_pct) > 0.1:
                            st.caption(f"Worst historical drop: {abs(max_dd_pct):.1f}%")
                        else:
                            st.caption("Limited historical data available")
                
                    # Add explanation about the data used
                    if selected_interval in ['1m', '5m', '15m', '30m', '1h']:
                        st.info("💡 **Note:** You are viewing minute-level data. Risk metrics have been converted to daily equivalents for better interpretation.")
                
                    # Drawdown chart
                    if len(drawdown) > 5:
                        st.markdown("### Historical Drawdown Chart")
                        st.caption("Shows every drop this stock has taken from its highest point")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, fill='tozeroy', line=dict(color='red')))
                        fig.update_layout(title="Drawdown History (Losses from Peak)", yaxis_title="Loss (%)", template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                        if abs(max_dd_pct) > 25:
                            st.warning("⚠️ This stock has seen significant drops before. Make sure you can handle this level of loss.")
                        elif abs(max_dd_pct) > 15:
                            st.info("📊 This stock's historical declines are within normal ranges for stocks.")
                        else:
                            st.success("✅ This stock has relatively small historical drawdowns.")
                    else:
                        st.info("Not enough historical data to display drawdown chart. Select a longer time period for better analysis.")
                
                    # Actionable summary
                    st.markdown("---")
                    st.markdown("### 📋 What You Should Do")
                
                    if abs(var_95_pct) < 1.5:
                        st.markdown("✅ **Conservative Approach:** This stock fits a low-risk portfolio. Normal position sizes are acceptable.")
                        st.markdown("📊 **Suggested Stop-Loss:** 10-12% below entry price")
                    elif abs(var_95_pct) < 3.0:
                        st.markdown("📊 **Balanced Approach:** This stock fits a moderate-risk portfolio. Use normal position sizes.")
                        st.markdown("📊 **Suggested Stop-Loss:** 8-10% below entry price")
                    else:
                        st.markdown("⚠️ **Aggressive Approach Only:** This stock is high-risk. Use smaller positions (50% of normal).")
                        st.markdown("📊 **Suggested Stop-Loss:** 5-8% below entry price")
                    
                else:
                    st.warning("Insufficient data for risk analysis. Please select a longer time period (1 Month or more) for meaningful risk metrics.")
    
        render_tab2()

    # ============================================================================
    # TAB 3: PORTFOLIO BUILDER
    # ============================================================================
    
    with tab3:
        @st.fragment
        def render_tab3():
            st.header("Portfolio Builder")
        
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.subheader("Your Investment Profile")
        
            col1, col2 = st.columns(2)
        
            with col1:
                investment_goal = st.selectbox(
                    "Primary Goal",
                    ["Capital Preservation (Low Risk)", "Balanced Growth (Moderate Risk)", "Maximum Returns (High Risk)"],
                    key="goal_select"
                )
                time_horizon = st.selectbox(
                    "Time Horizon",
                    options=["Short-term (< 1 year)", "Medium-term (1-3 years)", "Long-term (3+ years)"],
                    index=1,
                    key="horizon_slider"
                )
        
            with col2:
                experience = st.selectbox(
                    "Experience Level",
                    ["Beginner", "Intermediate", "Advanced"],
                    key="exp_select"
                )
                total_capital = st.number_input("Total Capital (USD)", min_value=1000, max_value=100000, value=10000, step=1000, key="capital_input")
        
            st.markdown('</div>', unsafe_allow_html=True)
        
            if "Capital Preservation" in investment_goal:
                risk_profile = "Conservative"
            elif "Balanced Growth" in investment_goal:
                risk_profile = "Moderate"
            else:
                risk_profile = "Aggressive"
        
            st.info(f"**Your Profile:** {risk_profile} Investor")
        
            stock_scores = []
            for ticker in selected_tickers:
                if ticker in processed_data:
                    data = processed_data[ticker]
                    returns = data['returns'].dropna()
                    if len(returns) > 10:
                        hurst = calculate_hurst_with_engineer(data['close'])
                        volatility = returns.std() * np.sqrt(252)
                        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252) + 0.001)
                        momentum = returns.tail(5).mean() * 252
                    
                        # Base factors
                        vol_factor = 1 / (volatility + 0.1)
                    
                        # Base weights based on risk profile
                        if risk_profile == "Conservative":
                            w_vol, w_sharpe, w_hurst, w_mom = 3.0, 0.5, 0.2, 0.0
                        elif risk_profile == "Aggressive":
                            w_vol, w_sharpe, w_hurst, w_mom = 0.5, 1.5, 2.5, 1.5
                        else: # Moderate
                            w_vol, w_sharpe, w_hurst, w_mom = 1.5, 1.5, 0.8, 0.5
                    
                        # Adjust weights based on time horizon
                        if "Short-term" in time_horizon:
                            w_mom += 1.0
                            w_vol += 1.0 # Need stability in short term
                            w_hurst -= 0.5 # Long trend doesn't matter as much
                        elif "Long-term" in time_horizon:
                            w_hurst += 1.0
                            w_vol -= 0.5 # Volatility is okay long term
                            w_mom -= 0.5 # Short term momentum matters less
                    
                        # Adjust weights based on experience
                        if experience == "Beginner":
                            w_vol += 1.0 # Steer to safer assets
                            w_sharpe += 0.5 # Beginners need more consistent returns
                        elif experience == "Advanced":
                            w_mom += 0.5
                            w_hurst += 0.5
                    
                        # Ensure positive weights
                        w_vol, w_hurst, w_mom = max(0.0, w_vol), max(0.0, w_hurst), max(0.0, w_mom)
                    
                        score = (vol_factor * w_vol) + (sharpe * w_sharpe) + (hurst * w_hurst) + (momentum * w_mom)
                    
                        stock_scores.append({
                            'ticker': ticker, 'score': score, 'volatility': volatility,
                            'sharpe': sharpe, 'hurst': hurst, 'momentum': momentum
                        })
        
            stock_scores.sort(key=lambda x: x['score'], reverse=True)
        
            st.subheader("Recommended Portfolio")
        
            if len(stock_scores) >= 2:
                total_score = sum(s['score'] for s in stock_scores)
                for s in stock_scores:
                    s['weight'] = max(0.10, min(0.40, s['score'] / total_score))
                total_weight = sum(s['weight'] for s in stock_scores)
                for s in stock_scores:
                    s['weight'] = s['weight'] / total_weight
            
                allocation_data = []
                for s in stock_scores:
                    amount = total_capital * s['weight']
                    allocation_data.append({
                        'Stock': s['ticker'],
                        'Allocation %': f"{s['weight']*100:.1f}%",
                        'Amount': f"${amount:,.0f}",
                        'Risk': "High" if s['volatility'] > 0.35 else "Moderate" if s['volatility'] > 0.20 else "Low"
                    })
            
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(pd.DataFrame(allocation_data), use_container_width=True, hide_index=True)
                with col2:
                    fig = px.pie(values=[s['weight'] for s in stock_scores], names=[s['ticker'] for s in stock_scores],
                               title="Portfolio Allocation", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
            
                portfolio_vol = np.sqrt(sum((s['weight'] * s['volatility'])**2 for s in stock_scores))
                portfolio_sharpe = np.mean([s['sharpe'] for s in stock_scores])
            
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Volatility", f"{portfolio_vol:.1%}")
                with col2:
                    st.metric("Avg Sharpe Ratio", f"{portfolio_sharpe:.2f}")
                with col3:
                    st.metric("Portfolio Risk", "Conservative" if portfolio_vol < 0.20 else "Moderate" if portfolio_vol < 0.35 else "Aggressive")
            
                st.markdown("""
                <div class='info-box'>
                    <h4>Action Plan</h4>
                    <ol>
                        <li>Start with 25% of recommended allocation to test</li>
                        <li>Set stop-loss orders at 8-10% below entry price</li>
                        <li>Review and rebalance portfolio monthly</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Need at least 2 stocks for portfolio recommendations")
    
        render_tab3()

    # ============================================================================
    # TAB 4: STATISTICAL TESTS
    # ============================================================================
    
    with tab4:
        @st.fragment
        def render_tab4():
            st.header("Statistical Analysis")
        
            selected_stat = st.selectbox("Select Security", selected_tickers, key="stat_select")
        
            if selected_stat in processed_data:
                data = processed_data[selected_stat]
                returns = data['returns'].dropna()
            
                if len(returns) > 20:
                    hurst = calculate_hurst_with_engineer(data['close'])
                    skewness = skew(returns)
                    kurt = kurtosis(returns)
                    jb_stat, jb_pvalue = jarque_bera(returns)
                
                    col1, col2 = st.columns(2)
                
                    with col1:
                        st.markdown("**Market Behavior**")
                        if hurst > 0.65:
                            st.success(f"Hurst: {hurst:.3f} - Trending")
                        elif hurst < 0.35:
                            st.warning(f"Hurst: {hurst:.3f} - Mean-Reverting")
                        else:
                            st.info(f"Hurst: {hurst:.3f} - Random")
                        st.metric("Skewness", f"{skewness:.3f}")
                        st.metric("Kurtosis", f"{kurt:.3f}")
                
                    with col2:
                        st.markdown("**Normality Test**")
                        st.metric("Jarque-Bera P-Value", f"{jb_pvalue:.4f}")
                        if jb_pvalue < 0.05:
                            st.warning("Returns are NOT normally distributed")
                        else:
                            st.success("Returns appear normally distributed")
                
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=returns, nbinsx=40, histnorm='probability density',
                                              name="Actual", marker_color='steelblue', opacity=0.7))
                    x_norm = np.linspace(returns.min(), returns.max(), 100)
                    y_norm = norm.pdf(x_norm, returns.mean(), returns.std())
                    fig.add_trace(go.Scatter(x=x_norm, y=y_norm, name="Normal", line=dict(color='red', width=2, dash='dash')))
                    fig.update_layout(title="Return Distribution", template="plotly_dark", height=450)
                    st.plotly_chart(fig, use_container_width=True)
    
        render_tab4()

    # ============================================================================
    # TAB 5: CORRELATION
    # ============================================================================
    
    with tab5:
        st.header("Correlation Analysis")
        st.caption("Shows how your selected stocks move together")
        
        if len(selected_tickers) >= 2:
            # Prepare returns data
            returns_dict = {}
            for ticker in selected_tickers:
                if ticker in processed_data:
                    returns = processed_data[ticker]['returns'].dropna()
                    if len(returns) > 20:
                        returns_dict[ticker] = returns
            
            if len(returns_dict) >= 2:
                # Calculate Pearson correlation
                returns_df = pd.DataFrame(returns_dict)
                pearson_corr = returns_df.corr()
                
                # Display Pearson correlation matrix
                st.subheader("Pearson Correlation Matrix")
                fig = px.imshow(pearson_corr, text_auto=True, aspect="auto",
                               color_continuous_scale="RdBu", zmin=-1, zmax=1,
                               title="Standard Correlation Matrix")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Try Copula correlation
                st.subheader("Copula Correlation (Advanced)")
                try:
                    copula_corr = get_copula_correlation(returns_dict)
                    if copula_corr is not None and isinstance(copula_corr, pd.DataFrame):
                        fig2 = px.imshow(copula_corr, text_auto=True, aspect="auto",
                                        color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                        title="Copula-Based Correlation")
                        fig2.update_layout(height=500)
                        st.plotly_chart(fig2, use_container_width=True)
                        st.caption("Copula captures non-linear relationships between stocks")
                    else:
                        st.info("Copula correlation requires more data points. Using Pearson correlation above.")
                except Exception as e:
                    st.info("Copula correlation requires more data. Pearson correlation shown above is sufficient.")
                
                # Diversification insights
                avg_corr = pearson_corr.values[np.triu_indices_from(pearson_corr.values, k=1)].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Correlation", f"{avg_corr:.3f}")
                with col2:
                    st.metric("Diversification Score", f"{(1-avg_corr)*100:.0f}%")
                
                if avg_corr > 0.7:
                    st.warning("⚠️ High correlation detected. Your stocks tend to move together. Consider adding different sectors for better diversification.")
                elif avg_corr < 0.3:
                    st.success("✅ Low correlation detected. Good diversification! Your portfolio is well spread.")
                else:
                    st.info("📊 Moderate correlation. Reasonable diversification, but could be improved.")
            else:
                st.warning("Insufficient data for correlation analysis. Select stocks with more historical data.")
        else:
            st.info("Select at least 2 securities to see correlation analysis")

   # =============================================================================
    # TAB 6: DATA SCIENCE PIPELINE TRACE (FULLY EXPLAINABLE SYSTEM)
    # =============================================================================

    with tab6:
        st.header("Data Science Pipeline")
        st.caption("End-to-end trace of how raw data becomes a trading decision")

        tracker = get_tracker()

        if not tracker.steps:
            st.info("Run analysis to see pipeline.")
            st.stop()

        breakdown = tracker.get_detailed_breakdown()

        # =========================================================================
        # 1. DATA INGESTION & PREPROCESSING
        # =========================================================================
        st.subheader("Stage 1: Data Pipeline")

        # Data Collection
        data_steps = [s for s in breakdown["all_steps"] if "Data Collection" in s["step_name"]]
        if data_steps:
            with st.expander("Raw Data Ingestion", expanded=True):
                step = data_steps[0]
                st.markdown(f"**Source:** {step.get('output_data', {}).get('data_source', 'Unknown')}")
                st.markdown(f"**Interval:** {step.get('output_data', {}).get('interval', 'N/A')}")
                st.markdown(f"**Period:** {step.get('output_data', {}).get('period', 'N/A')}")

                if step.get('data_snapshots', {}).get('raw_data') is not None:
                    raw_df = step['data_snapshots']['raw_data']
                    if isinstance(raw_df, pd.DataFrame):
                        st.markdown("**Raw Data Sample:**")
                        st.dataframe(raw_df.head(3), use_container_width=True)

                if step.get('metrics'):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Rows", step['metrics'].get('total_rows', 0))
                    with col2: st.metric("Columns", step['metrics'].get('total_columns', 0))
                    with col3: st.metric("Missing %", f"{step['metrics'].get('missing_percentage', 0):.1f}%")
                    with col4: st.metric("Data Quality", f"{100 - step['metrics'].get('missing_percentage', 0):.1f}%")

        # Data Preprocessing
        preprocess_steps = [s for s in breakdown["all_steps"] if "Data Preprocessing" in s["step_name"]]
        if preprocess_steps:
            with st.expander("Data Cleaning & Transformation", expanded=True):
                step = preprocess_steps[0]
                st.markdown("**Applied Transformations:**")
                steps = step.get('output_data', {}).get('preprocessing_steps', [])

                # Fix if it's accidentally a string
                if isinstance(steps, str):
                    import ast
                    try:
                        steps = ast.literal_eval(steps)
                    except:
                        steps = [steps]

                for method in steps:
                    st.markdown(f"• {method}")

                if step.get('data_snapshots'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Before Cleaning:**")
                        if 'raw_data_sample' in step['data_snapshots']:
                            st.dataframe(step['data_snapshots']['raw_data_sample'], use_container_width=True)
                    with col2:
                        st.markdown("**After Cleaning:**")
                        if 'cleaned_data_sample' in step['data_snapshots']:
                            st.dataframe(step['data_snapshots']['cleaned_data_sample'], use_container_width=True)

                if step.get('metrics'):
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Data Retention", f"{step['metrics'].get('data_retention_rate', 0):.1f}%")
                    with col2: st.metric("Outliers Removed", step['metrics'].get('outlier_percentage', 0))
                    with col3: st.metric("Missing Filled", step['metrics'].get('missing_fill_rate', 0))

        st.markdown("---")

        # =========================================================================
        # 2. FEATURE ENGINEERING
        # =========================================================================
        st.subheader("Stage 2: Feature Engineering")

        feature_steps = [s for s in breakdown["all_steps"] if "Feature Engineering" in s["step_name"]]
        if feature_steps:
            with st.expander("Feature Creation Pipeline", expanded=True):
                step = feature_steps[0]

                st.markdown("**Features Generated:**")
                features = step.get('output_data', {}).get('calculated_features', [])
                # FIX: Handle string case
                if isinstance(features, str):
                    import ast
                    try:
                        features = ast.literal_eval(features)
                    except:
                        features = [features]  # fallback
                for feature in features:
                    st.markdown(f"• `{feature}`")

                st.markdown("**Feature Matrix Sample:**")
                if step.get('feature_vectors', {}).get('feature_matrix') is not None:
                    feature_df = step['feature_vectors']['feature_matrix']
                    if isinstance(feature_df, list):
                        feature_df = pd.DataFrame(feature_df)

                    if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
                        st.dataframe(feature_df.tail(5), use_container_width=True)
                    else:
                        st.warning("Feature matrix is empty or not available.")

                st.markdown("**Feature Statistics:**")
                if step.get('feature_vectors', {}).get('feature_statistics'):
                    stats_df = pd.DataFrame(step['feature_vectors']['feature_statistics']).T
                    st.dataframe(stats_df, use_container_width=True)

                if step.get('parameters'):
                    st.markdown("**Parameters Used:**")
                    st.json(step['parameters'])

        st.markdown("---")

        # =========================================================================
        # 3. STATISTICAL ANALYSIS
        # =========================================================================
        st.subheader("Stage 3: Statistical Modeling")

        stat_steps = [s for s in breakdown["all_steps"] if "Statistical Analysis" in s["step_name"]]
        if stat_steps:
            with st.expander("Statistical Characterization", expanded=True):
                step = stat_steps[0]

                # Core Statistics
                if step.get('statistical_outputs', {}).get('measures'):
                    st.markdown("**Core Statistical Measures:**")
                    measures = step['statistical_outputs']['measures']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Hurst", f"{measures.get('hurst', 0):.3f}")
                    with col2: st.metric("Volatility", f"{measures.get('volatility', 0):.1%}")
                    with col3: st.metric("Sharpe", f"{measures.get('sharpe_ratio', 0):.2f}")
                    with col4: st.metric("RSI", f"{measures.get('rsi', 0):.1f}")

                # Distribution Tests
                if step.get('statistical_outputs', {}).get('distribution_tests'):
                    st.markdown("**Distribution Tests:**")
                    tests = step['statistical_outputs']['distribution_tests']
                    if 'jarque_bera' in tests:
                        jb = tests['jarque_bera']
                        col1, col2 = st.columns(2)
                        with col1: st.metric("JB Statistic", f"{jb.get('statistic', 0):.2f}")
                        with col2: st.metric("JB p-value", f"{jb.get('p_value', 0):.4f}")
                        st.caption("p < 0.05 indicates non-normal distribution")

                # Stationarity Tests
                if step.get('statistical_outputs', {}).get('stationarity_tests'):
                    st.markdown("**Stationarity Tests:**")
                    tests = step['statistical_outputs']['stationarity_tests']
                    if 'adf' in tests:
                        adf = tests['adf']
                        col1, col2 = st.columns(2)
                        with col1: st.metric("ADF Statistic", f"{adf.get('statistic', 0):.2f}")
                        with col2: st.metric("ADF p-value", f"{adf.get('p_value', 0):.4f}")
                        st.caption("p < 0.05 indicates stationary series")

        st.markdown("---")

        # =========================================================================
        # 4. MODEL OUTPUTS
        # =========================================================================
        st.subheader("Stage 4: ML Model Outputs")

        # Regime Detection
        regime_steps = [s for s in breakdown["all_steps"] if "Regime Detection" in s["step_name"]]
        if regime_steps:
            with st.expander("Market Regime Classification", expanded=True):
                step = regime_steps[0]

                if step.get('model_outputs'):
                    outputs = step['model_outputs']
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Detected Regime", outputs.get('detected_regime', 'Unknown'))
                    with col2: st.metric("Confidence", f"{outputs.get('confidence', 0):.1%}")
                    with col3: st.metric("Strategy", outputs.get('regime_features', {}).get('strategy', 'Unknown'))

                if step.get('decision_inputs'):
                    st.markdown("**Decision Thresholds:**")
                    thresholds = step['decision_inputs'].get('regime_thresholds', {})
                    thresh_df = pd.DataFrame(list(thresholds.items()), columns=['Threshold', 'Value'])
                    st.dataframe(thresh_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # =========================================================================
        # 5. DECISION ENGINE
        # =========================================================================
        st.subheader("Stage 5: Decision Engine")

        # Recommendation Engine
        rec_engine_steps = [s for s in breakdown["all_steps"] if "Recommendation Engine" in s["step_name"]]
        if rec_engine_steps:
            with st.expander("Multi-Factor Scoring System", expanded=True):
                step = rec_engine_steps[0]

                if step.get('decision_inputs'):
                    inputs = step['decision_inputs']

                    # Component Scores
                    st.markdown("**Component Scores:**")
                    scores = inputs.get('scoring_components', {})
                    score_df = pd.DataFrame(list(scores.items()), columns=['Component', 'Score'])
                    st.dataframe(score_df, use_container_width=True, hide_index=True)

                    # Decision Weights
                    st.markdown("**Decision Weights:**")
                    weights = inputs.get('decision_weights', {})
                    weight_df = pd.DataFrame(list(weights.items()), columns=['Factor', 'Weight'])
                    st.dataframe(weight_df, use_container_width=True, hide_index=True)

                    # Threshold Rules
                    st.markdown("**Decision Thresholds:**")
                    rules = inputs.get('threshold_rules', {})
                    rule_df = pd.DataFrame(list(rules.items()), columns=['Threshold', 'Score'])
                    st.dataframe(rule_df, use_container_width=True, hide_index=True)

                if step.get('output_data'):
                    outputs = step['output_data']
                    col1, col2 = st.columns(2)
                    with col1: st.metric("Final Recommendation", outputs.get('final_recommendation', 'Unknown'))
                    with col2: st.metric("Confidence Score", f"{outputs.get('confidence_score', 0):.1%}")

        # Final Recommendation
        final_rec_steps = [s for s in breakdown["all_steps"] if "Final Recommendation" in s["step_name"]]
        if final_rec_steps:
            with st.expander("Final Decision", expanded=True):
                step = final_rec_steps[0]

                if step.get('output_data'):
                    outputs = step['output_data']
                    st.markdown(f"**Action:** {outputs.get('action', 'Unknown')}")
                    st.markdown(f"**Risk Level:** {outputs.get('risk_level', 'Unknown')}")
                    st.markdown(f"**Position Size:** ${outputs.get('position_size', 0):,.0f}")

                    st.markdown("**Supporting Factors:**")
                    factors = outputs.get('supporting_factors', [])

                    # FIX: Handle string case
                    if isinstance(factors, str):
                        import ast
                        try:
                            factors = ast.literal_eval(factors)
                        except:
                            factors = [factors]

                    for factor in factors:
                        st.markdown(f"• {factor}")

        st.markdown("---")

        # =========================================================================
        # 6. PIPELINE PERFORMANCE & TRACEABILITY
        # =========================================================================
        st.subheader("Pipeline Performance")

        # Feature → Decision Flow
        st.markdown("**Feature Flow Traceability:**")
        flow_data = []

        for step in breakdown["all_steps"]:
            step_name = step["step_name"]
            if step.get("statistical_outputs", {}).get("measures"):
                for k, v in step["statistical_outputs"]["measures"].items():
                    flow_data.append({"Stage": step_name, "Feature": k, "Value": v, "Type": "Statistical"})
            if step.get("feature_vectors", {}).get("feature_statistics"):
                for feature, stats in step["feature_vectors"]["feature_statistics"].items():
                    flow_data.append({"Stage": step_name, "Feature": feature, "Value": stats.get("mean", 0), "Type": "Feature"})
            if step.get("model_outputs"):
                for k, v in step["model_outputs"].items():
                    if isinstance(v, (int, float)):
                        flow_data.append({"Stage": step_name, "Feature": k, "Value": v, "Type": "Model"})
            if step.get("decision_inputs", {}).get("scoring_components"):
                for k, v in step["decision_inputs"]["scoring_components"].items():
                    flow_data.append({"Stage": step_name, "Feature": k, "Value": v, "Type": "Decision"})

        if flow_data:
            flow_df = pd.DataFrame(flow_data)
            st.dataframe(flow_df, use_container_width=True, hide_index=True)

        # Execution Stats
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Pipeline Steps", breakdown["total_steps"])
        with col2: st.metric("Execution Time", f"{breakdown['analysis_duration']:.2f}s")
        with col3: st.metric("Avg Time per Step", f"{breakdown['analysis_duration']/max(breakdown['total_steps'], 1):.2f}s")

        st.markdown("---")

        # =========================================================================
        # MATHEMATICAL ANALYSIS BREAKDOWN
        # =========================================================================
        
    # Footer
    st.divider()
    st.caption(f"Data: {selected_interval_name} | Period: {selected_period_name} | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Active Modules: DataCollector | Preprocessor | AdvancedFeatureEngineer | AdvancedPortfolioManager | RegimeDetector | AlertSystem")

if __name__ == "__main__":
    main()

