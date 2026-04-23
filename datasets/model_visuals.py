# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt


# # # # =========================
# # # # LOAD DATA
# # # # =========================
# # # tech = pd.read_csv("technical_dataset.csv")
# # # model = pd.read_csv("model_dataset.csv")


# # # # =========================================================
# # # # 1. STATISTICAL & PREDICTIVE MODEL VISUALIZATIONS
# # # # =========================================================

# # # # --- 1. Volatility (Feature Representation) ---
# # # plt.figure(figsize=(10,5))
# # # plt.plot(tech["volatility"], linewidth=1)
# # # plt.title("Volatility (Statistical Feature - Market Risk Proxy)")
# # # plt.xlabel("Time")
# # # plt.ylabel("Volatility")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig("volatility_feature.png", dpi=300)
# # # plt.show()


# # # # --- 2. Log Returns Distribution (Stationarity Check) ---
# # # plt.figure(figsize=(10,5))
# # # plt.hist(tech["returns_log"], bins=50, edgecolor="black")
# # # plt.title("Log Returns Distribution (Stationarity Transformation)")
# # # plt.xlabel("Returns")
# # # plt.ylabel("Frequency")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig("returns_stationarity.png", dpi=300)
# # # plt.show()


# # # # --- 3. Market Regime Classification ---
# # # plt.figure(figsize=(10,5))

# # # regime_codes = model["regime"].astype("category").cat.codes

# # # plt.scatter(
# # #     range(len(model)),
# # #     model["close"],
# # #     c=regime_codes,
# # #     cmap="viridis",
# # #     s=8
# # # )

# # # plt.colorbar(label="Regime Type (Encoded)")
# # # plt.title("Market Regime Classification (Rule-Based System Output)")
# # # plt.xlabel("Time")
# # # plt.ylabel("Price")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig("regime_classification.png", dpi=300)
# # # plt.show()


# # # # =========================================================
# # # # 2. FORECASTING RESULTS WITH PERFORMANCE METRICS
# # # # =========================================================

# # # returns = model["returns_log"].dropna()


# # # # --- 4. Value at Risk (VaR 95%) ---
# # # var_95 = np.percentile(returns, 5)

# # # plt.figure(figsize=(10,5))
# # # plt.hist(returns, bins=50, edgecolor="black")
# # # plt.axvline(var_95, color="red", linestyle="--", linewidth=2, label="VaR (95%)")
# # # plt.title("Value at Risk (95% Confidence Level)")
# # # plt.xlabel("Returns")
# # # plt.ylabel("Frequency")
# # # plt.legend()
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig("var_95.png", dpi=300)
# # # plt.show()


# # # # --- 5. CVaR (Tail Risk Visualization) ---
# # # cvar_95 = returns[returns <= var_95].mean()

# # # plt.figure(figsize=(10,5))
# # # plt.hist(returns, bins=50, edgecolor="black")
# # # plt.axvline(var_95, color="red", linestyle="--", label="VaR")
# # # plt.axvline(cvar_95, color="black", linestyle="--", label="CVaR")
# # # plt.title("Conditional VaR (Tail Risk Exposure)")
# # # plt.xlabel("Returns")
# # # plt.ylabel("Frequency")
# # # plt.legend()
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig("cvar_95.png", dpi=300)
# # # plt.show()


# # # # --- 6. Volatility Forecast (Risk Over Time) ---
# # # vol = returns.rolling(20).std() * np.sqrt(252)

# # # plt.figure(figsize=(10,5))
# # # plt.plot(vol, linewidth=1)
# # # plt.title("Volatility Forecast (Annualized Risk: σ × √252)")
# # # plt.xlabel("Time")
# # # plt.ylabel("Annualized Volatility")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig("volatility_forecast.png", dpi=300)
# # # plt.show()


# # # # --- 7. Maximum Drawdown (Risk Stability) ---
# # # cum = (1 + returns).cumprod()
# # # peak = np.maximum.accumulate(cum)
# # # drawdown = (cum - peak) / peak

# # # plt.figure(figsize=(10,5))
# # # plt.plot(drawdown, color="red")
# # # plt.title("Maximum Drawdown (Portfolio Risk Exposure)")
# # # plt.xlabel("Time")
# # # plt.ylabel("Drawdown")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig("drawdown.png", dpi=300)
# # # plt.show()


# # # print("✅ REPORT-LEVEL VISUALIZATIONS GENERATED SUCCESSFULLY")

# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # =========================
# # # 1. LOAD DATA
# # # =========================
# # # Load the files
# # tech = pd.read_csv("technical_dataset.csv")
# # model = pd.read_csv("model_dataset.csv")

# # # --- ROBUST FIX: Clean column names ---
# # # This removes leading/trailing spaces and converts to lowercase
# # tech.columns = tech.columns.str.strip().str.lower()
# # model.columns = model.columns.str.strip().str.lower()

# # # Convert dates to datetime objects
# # tech["date"] = pd.to_datetime(tech["date"])
# # model["date"] = pd.to_datetime(model["date"])

# # # =========================================================
# # # 2. STATISTICAL & PREDICTIVE MODEL VISUALIZATIONS
# # # =========================================================

# # # --- 1. Volatility (Feature Representation) ---
# # plt.figure(figsize=(10,5))
# # plt.plot(tech["date"], tech["volatility"], linewidth=1)
# # plt.title("Volatility (Statistical Feature - Market Risk Proxy)")
# # plt.xlabel("Time")
# # plt.ylabel("Volatility")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("volatility_feature.png", dpi=300)

# # # --- 2. Log Returns Distribution (Stationarity Check) ---
# # plt.figure(figsize=(10,5))
# # plt.hist(tech["returns_log"].dropna(), bins=50, edgecolor="black")
# # plt.title("Log Returns Distribution (Stationarity Transformation)")
# # plt.xlabel("Returns")
# # plt.ylabel("Frequency")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("returns_stationarity.png", dpi=300)

# # # --- 3. Market Regime Classification ---
# # plt.figure(figsize=(10,5))
# # # Use numeric codes for categorical data
# # regime_codes = model["regime"].astype("category").cat.codes

# # plt.scatter(
# #     model["date"],
# #     model["close"],
# #     c=regime_codes,
# #     cmap="viridis",
# #     s=8
# # )
# # plt.colorbar(label="Regime Type (Encoded)")
# # plt.title("Market Regime Classification (Rule-Based System Output)")
# # plt.xlabel("Time")
# # plt.ylabel("Price")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("regime_classification.png", dpi=300)

# # # =========================================================
# # # 3. FORECASTING RESULTS WITH PERFORMANCE METRICS
# # # =========================================================

# # returns = model["returns_log"].dropna()

# # # --- 4. Value at Risk (VaR 95%) ---
# # var_95 = np.percentile(returns, 5)

# # plt.figure(figsize=(10,5))
# # plt.hist(returns, bins=50, edgecolor="black")
# # plt.axvline(var_95, color="red", linestyle="--", linewidth=2, label="VaR (95%)")
# # plt.title("Value at Risk (95% Confidence Level)")
# # plt.xlabel("Returns")
# # plt.ylabel("Frequency")
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("var_95.png", dpi=300)

# # # --- 5. CVaR (Tail Risk Visualization) ---
# # cvar_95 = returns[returns <= var_95].mean()

# # plt.figure(figsize=(10,5))
# # plt.hist(returns, bins=50, edgecolor="black")
# # plt.axvline(var_95, color="red", linestyle="--", label="VaR")
# # plt.axvline(cvar_95, color="black", linestyle="--", label="CVaR")
# # plt.title("Conditional VaR (Tail Risk Exposure)")
# # plt.xlabel("Returns")
# # plt.ylabel("Frequency")
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("cvar_95.png", dpi=300)

# # # --- 6. Volatility Forecast (Risk Over Time) ---
# # vol = returns.rolling(20).std() * np.sqrt(252)

# # plt.figure(figsize=(10,5))
# # plt.plot(model["date"].iloc[vol.index], vol, linewidth=1)
# # plt.title("Volatility Forecast (Annualized Risk: σ × √252)")
# # plt.xlabel("Time")
# # plt.ylabel("Annualized Volatility")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("volatility_forecast.png", dpi=300)

# # # --- 7. Maximum Drawdown (Risk Stability) ---
# # cum = (1 + returns).cumprod()
# # peak = np.maximum.accumulate(cum)
# # drawdown = (cum - peak) / peak

# # plt.figure(figsize=(10,5))
# # plt.plot(model["date"].iloc[drawdown.index], drawdown, color="red")
# # plt.title("Maximum Drawdown (Portfolio Risk Exposure)")
# # plt.xlabel("Time")
# # plt.ylabel("Drawdown")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("drawdown.png", dpi=300)

# # print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set style for professional report quality
# sns.set_style("whitegrid")

# # Load your data (assuming final model dataset)
# df = pd.read_csv("model_dataset.csv")
# df["date"] = pd.to_datetime(df["date"])
# df = df.sort_values("date")

# # =========================================================
# # REPORT 1: STATISTICAL MODELING FRAMEWORK
# # =========================================================
# def generate_statistical_visuals(data):
#     print("Generating Statistical Framework Visuals...")
    
#     # 1. Smoothing (EMA Visualization)
#     plt.figure(figsize=(10, 5))
#     plt.plot(data["date"], data["close"], label="Actual Price", alpha=0.5)
#     plt.plot(data["date"], data["close_smoothed"], label="EMA (Smoothed)", linewidth=2)
#     plt.title("Statistical Framework: Noise Reduction via EMA")
#     plt.legend()
#     plt.savefig("report_ema_smoothing.png", dpi=300)
#     plt.close()

#     # 2. Stationarity Transformation (Log Returns)
#     plt.figure(figsize=(10, 5))
#     plt.hist(data["returns_log"].dropna(), bins=50, color='skyblue', edgecolor='black')
#     plt.title("Stationarity Check: Distribution of Log Returns ($r_t = \log(P_t / P_{t-1})$)")
#     plt.xlabel("Log Returns")
#     plt.savefig("report_log_returns.png", dpi=300)
#     plt.close()

#     # 3. Feature Engineering Signal Space
#     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#     axes[0].plot(data["date"], data["volatility"], color='orange')
#     axes[0].set_title("Feature: Rolling Volatility")
#     axes[1].plot(data["date"], data["rsi"], color='green')
#     axes[1].set_title("Feature: Relative Strength Index (RSI)")
#     axes[2].bar(data["date"], data["volume_zscore"], color='purple', alpha=0.6)
#     axes[2].set_title("Feature: Volume Z-Score (Anomaly Detection)")
#     plt.tight_layout()
#     plt.savefig("report_feature_engineering.png", dpi=300)
#     plt.close()

# # =========================================================
# # REPORT 2: PREDICTIVE MODELING & PERFORMANCE
# # =========================================================
# def generate_predictive_visuals(data):
#     print("Generating Predictive Modeling Visuals...")
    
#     # 1. Market Regime Classification
#     plt.figure(figsize=(12, 6))
#     regimes = data["regime"].unique()
#     for regime in regimes:
#         subset = data[data["regime"] == regime]
#         plt.scatter(subset["date"], subset["close"], label=regime, s=10)
#     plt.title("Predictive Model: Market Regime Classification")
#     plt.legend()
#     plt.savefig("report_regime_classification.png", dpi=300)
#     plt.close()

#     # 2. Risk Metrics (VaR/CVaR)
#     returns = data["returns_log"].dropna()
#     var_95 = np.percentile(returns, 5)
#     cvar_95 = returns[returns <= var_95].mean()
    
#     plt.figure(figsize=(10, 5))
#     sns.histplot(returns, kde=True, color='gray')
#     plt.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.3f}')
#     plt.axvline(cvar_95, color='black', linestyle='-', label=f'CVaR 95%: {cvar_95:.3f}')
#     plt.title("Risk Performance: VaR vs. CVaR (Tail Risk)")
#     plt.legend()
#     plt.savefig("report_risk_metrics.png", dpi=300)
#     plt.close()

#     # 3. Maximum Drawdown
#     cum = (1 + returns).cumprod()
#     peak = np.maximum.accumulate(cum)
#     drawdown = (cum - peak) / peak
    
#     plt.figure(figsize=(10, 5))
#     plt.fill_between(data["date"].iloc[drawdown.index], drawdown, 0, color='red', alpha=0.3)
#     plt.title("Performance: Maximum Drawdown Profile")
#     plt.savefig("report_drawdown.png", dpi=300)
#     plt.close()

# # Run both sections
# if __name__ == "__main__":
#     generate_statistical_visuals(df)
#     generate_predictive_visuals(df)
#     print("✅ All Report Visualizations Generated Successfully.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    if 'Date' in df.columns: df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['ticker', 'date'])

# def plot_risk_metrics(df, ticker):
#     subset = df[df['ticker'] == ticker].copy().sort_values('date')
    
#     # 1. Calculate returns and align properly
#     subset['returns'] = subset['close'].pct_change() # Use simple returns for drawdown
#     subset = subset.dropna(subset=['returns'])
    
#     # 2. Calculate Drawdown
#     cumulative = (1 + subset['returns']).cumprod()
#     peak = cumulative.cummax()
#     subset['drawdown'] = (cumulative - peak) / peak
    
#     # 3. Create a clean subset for plotting
#     plot_df = subset.iloc[1:].copy() 
    
#     # --- NOW EVERYTHING IS EXACTLY THE SAME LENGTH ---
    
#     var_95 = np.percentile(subset['returns'], 5)
#     cvar_95 = subset['returns'][subset['returns'] <= var_95].mean()
    
#     fig = plt.figure(figsize=(15, 10))
    
#     # Plot A: Returns Distribution
#     ax1 = plt.subplot(2, 1, 1)
#     sns.histplot(subset['returns'], bins=50, kde=True, ax=ax1)
#     ax1.axvline(var_95, color='red', linestyle='--', label=f'VaR (95%): {var_95:.2%}')
#     ax1.axvline(cvar_95, color='black', linestyle='--', label=f'CVaR: {cvar_95:.2%}')
#     ax1.legend()
    
#     # Plot B: Max Drawdown
#     ax2 = plt.subplot(2, 1, 2)
#     ax2.fill_between(plot_df['date'], plot_df['drawdown'], 0, color='red', alpha=0.3)
#     ax2.plot(plot_df['date'], plot_df['drawdown'], color='red')
#     ax2.set_title(f"Maximum Drawdown: {ticker}")
    
#     plt.tight_layout()
#     plt.show()

# def plot_full_dashboard(df, ticker):
#     subset = df[df['ticker'] == ticker].copy().sort_values('date')
    
#     # Set up a large 4x1 dashboard
#     fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
#     # 1. Volatility
#     axes[0].plot(subset['date'], subset['volatility'], color='blue')
#     axes[0].set_title(f"Volatility (Annualized): {ticker}")
    
#     # 2. RSI
#     axes[1].plot(subset['date'], subset['rsi'], color='purple')
#     axes[1].axhline(70, linestyle='--', color='red', alpha=0.5)
#     axes[1].axhline(30, linestyle='--', color='green', alpha=0.5)
#     axes[1].set_title("RSI (Momentum)")
    
#     # 3. Volume Z-Score
#     axes[2].bar(subset['date'], subset['volume_zscore'], color='orange', alpha=0.6)
#     axes[2].set_title("Volume Z-Score (Anomaly Detection)")
    
#     # 4. Hurst Exponent
#     axes[3].plot(subset['date'], subset['hurst'], color='teal')
#     axes[3].axhline(0.5, linestyle='--', color='black')
#     axes[3].set_title("Hurst Exponent (Trending vs Mean-Reverting)")
    
#     plt.tight_layout()
#     plt.show()
# # Run it
# df = load_data("datasets/final.csv") # Assuming final.csv has the 'returns' column
# plot_risk_metrics(df, "AAPL")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()
#     if 'Date' in df.columns: df = df.rename(columns={'Date': 'date'})
#     df['date'] = pd.to_datetime(df['date'])
#     return df.sort_values(['ticker', 'date'])

# def plot_separate(df, ticker):
#     subset = df[df['ticker'] == ticker].copy()
#     subset['returns'] = subset['close'].pct_change()
    
#     # 1. Volatility Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(subset['date'], subset['volatility'], color='blue')
#     plt.title(f"Annualized Volatility: {ticker}")
#     plt.show()

#     # 2. Risk Metrics (VaR/CVaR)
#     plt.figure(figsize=(10, 5))
#     var_95 = np.percentile(subset['returns'].dropna(), 5)
#     sns.histplot(subset['returns'].dropna(), kde=True)
#     plt.axvline(var_95, color='red', linestyle='--', label=f'VaR (95%): {var_95:.2%}')
#     plt.title(f"Return Distribution & VaR: {ticker}")
#     plt.legend()
#     plt.show()

#     # 3. Maximum Drawdown
#     cumulative = (1 + subset['returns'].fillna(0)).cumprod()
#     peak = cumulative.cummax()
#     drawdown = (cumulative - peak) / peak
#     plt.figure(figsize=(10, 5))
#     plt.fill_between(subset['date'], drawdown, 0, color='red', alpha=0.3)
#     plt.title(f"Maximum Drawdown: {ticker}")
#     plt.show()

#     # 4. Hurst Exponent / Regime
#     plt.figure(figsize=(10, 5))
#     plt.plot(subset['date'], subset['hurst'], color='teal')
#     plt.axhline(0.5, color='black', linestyle='--')
#     plt.title(f"Hurst Exponent (Trend vs Reversion): {ticker}")
#     plt.show()

# if __name__ == "__main__":
#     df = load_data("datasets/model.csv")
#     plot_separate(df, "AAPL")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and ensure proper structure
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    if 'Date' in df.columns: df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['ticker', 'date'])

def generate_report_visuals(df, ticker):
    subset = df[df['ticker'] == ticker].copy()
    
    # Pre-requisite: Calculate Log Returns for Stationarity (Section 2.4)
    subset['log_returns'] = np.log(subset['close'] / subset['close'].shift(1))
    subset = subset.dropna()

    # --- FIGURE 1: STATIONARITY (Section 2.4) ---
    plt.figure(figsize=(10, 4))
    sns.histplot(subset['log_returns'], kde=True, bins=50)
    plt.title(f"Log Returns Distribution (Stationarity): {ticker}")
    plt.show()

    # --- FIGURE 2: VOLATILITY (Section 3 & Forecast 1) ---
    plt.figure(figsize=(10, 4))
    plt.plot(subset['date'], subset['volatility'], color='blue')
    plt.title(f"Annualized Volatility (Risk Proxy): {ticker}")
    plt.show()

    # --- FIGURE 3: RISK METRICS (Performance 2) ---
    plt.figure(figsize=(10, 4))
    var_95 = np.percentile(subset['log_returns'], 5)
    sns.histplot(subset['log_returns'], kde=True)
    plt.axvline(var_95, color='red', linestyle='--', label=f'VaR (95%): {var_95:.2%}')
    plt.legend()
    plt.title(f"Value at Risk (VaR) Analysis: {ticker}")
    plt.show()

    # --- FIGURE 4: MAXIMUM DRAWDOWN (Performance 2) ---
    cumulative = (1 + subset['log_returns']).cumsum().apply(np.exp)
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    plt.figure(figsize=(10, 4))
    plt.fill_between(subset['date'], drawdown, 0, color='red', alpha=0.3)
    plt.title(f"Maximum Drawdown (Portfolio Risk): {ticker}")
    plt.show()

    # --- FIGURE 5: HURST EXPONENT (Section 3 & Decision Logic 4.2) ---
    plt.figure(figsize=(10, 4))
    plt.plot(subset['date'], subset['hurst'], color='teal')
    plt.axhline(0.5, color='black', linestyle='--')
    plt.title(f"Hurst Exponent (Trend vs Mean-Reversion): {ticker}")
    plt.show()

    # --- FIGURE 6: REGIME CLASSIFICATION (Predictive 4.1) ---
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=subset, x='date', y='close', hue='regime', palette='viridis', s=20)
    plt.title(f"Market Regime Classification Timeline: {ticker}")
    plt.show()

if __name__ == "__main__":
    data = load_data("datasets/model.csv")
    generate_report_visuals(data, "AAPL")