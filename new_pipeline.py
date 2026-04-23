# # # """
# # # FINAL MARKET PIPELINE (ROBUST VERSION)

# # # RAW → CLEAN → TECH → MODEL
# # # ✔ 10 years historical data
# # # ✔ multi-stock
# # # ✔ datetime-safe
# # # ✔ returns column fix
# # # ✔ CSV export at every stage
# # # """

# # # import pandas as pd
# # # import numpy as np
# # # import os
# # # import time

# # # from data_collection import DataCollector
# # # from preprocessing import DataPreprocessor
# # # from advanced_features import AdvancedFeatureEngineer
# # # from regime_detection import RegimeDetector

# # # # =========================
# # # # INIT
# # # # =========================
# # # preprocessor = DataPreprocessor()
# # # feature_engineer = AdvancedFeatureEngineer()
# # # regime_detector = RegimeDetector()

# # # os.makedirs("datasets", exist_ok=True)

# # # # =========================
# # # # STAGE 1: RAW DATA
# # # # =========================
# # # def get_raw(tickers):
# # #     collector = DataCollector(tickers)
# # #     raw = {}

# # #     for t in tickers:
# # #         print(f"📥 Fetching {t}...")

# # #         df = collector.get_historical_data(t, days=365*10)

# # #         if df is None or df.empty:
# # #             print(f"⚠️ No data for {t}")
# # #             continue

# # #         # ✅ Ensure datetime index
# # #         df.index = pd.to_datetime(df.index)
# # #         df = df.sort_index()

# # #         raw[t] = df
# # #         time.sleep(1)

# # #     return raw


# # # # =========================
# # # # STAGE 2: CLEAN DATA
# # # # =========================
# # # def clean_data(raw_data):
# # #     cleaned = {}

# # #     for t, df in raw_data.items():
# # #         df = df.copy()

# # #         try:
# # #             df = preprocessor.preprocess(df)
# # #             df = preprocessor.calculate_returns(df)

# # #             # ✅ FIX: normalize returns column
# # #             if "returns" not in df.columns:
# # #                 if "returns_log" in df.columns:
# # #                     df["returns"] = df["returns_log"]

# # #             cleaned[t] = df

# # #         except Exception as e:
# # #             print(f"⚠️ Cleaning failed for {t}: {e}")

# # #     return cleaned


# # # # =========================
# # # # STAGE 3: TECHNICAL FEATURES
# # # # =========================
# # # def add_technical(data):
# # #     out = {}

# # #     for t, df in data.items():
# # #         df = df.copy()

# # #         if "returns" not in df.columns:
# # #             print(f"⚠️ Missing returns for {t}")
# # #             continue

# # #         # Volatility
# # #         df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)

# # #         # RSI
# # #         delta = df["close"].diff()
# # #         gain = delta.where(delta > 0, 0).rolling(14).mean()
# # #         loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# # #         rs = gain / (loss + 1e-9)
# # #         df["rsi"] = 100 - (100 / (1 + rs))

# # #         # Volume Z-score
# # #         df["volume_zscore"] = (
# # #             (df["volume"] - df["volume"].rolling(20).mean()) /
# # #             (df["volume"].rolling(20).std() + 1e-9)
# # #         )

# # #         # Hurst
# # #         try:
# # #             df["hurst"] = feature_engineer.calculate_hurst(df["close"])
# # #         except:
# # #             df["hurst"] = np.nan

# # #         out[t] = df

# # #     return out


# # # # =========================
# # # # STAGE 4: MODEL FEATURES
# # # # =========================
# # # def add_model_features(data):
# # #     out = {}

# # #     for t, df in data.items():
# # #         df = df.copy()

# # #         try:
# # #             regime = regime_detector.detect_regime(
# # #                 hurst_exponent=df["hurst"].iloc[-1],
# # #                 volatility=df["volatility"].iloc[-1],
# # #                 volume_zscore=df["volume_zscore"].iloc[-1],
# # #                 is_stationary=False
# # #             )

# # #             df["regime"] = regime.regime.value
# # #             df["regime_confidence"] = regime.confidence

# # #         except Exception as e:
# # #             print(f"⚠️ Regime detection failed for {t}: {e}")
# # #             df["regime"] = "UNKNOWN"
# # #             df["regime_confidence"] = 0

# # #         out[t] = df

# # #     return out


# # # # =========================
# # # # EXPORT FUNCTION
# # # # =========================
# # # def export(stage_name, data):
# # #     combined = []

# # #     for t, df in data.items():
# # #         temp = df.copy()

# # #         # ✅ keep date column for EDA / Power BI
# # #         temp["date"] = temp.index

# # #         temp["ticker"] = t
# # #         temp["stage"] = stage_name

# # #         combined.append(temp)

# # #     if not combined:
# # #         print(f"⚠️ No data to save for {stage_name}")
# # #         return

# # #     final = pd.concat(combined)

# # #     path = f"datasets/{stage_name}_dataset.csv"
# # #     final.to_csv(path, index=False)

# # #     print(f"💾 Saved {path} | shape={final.shape}")


# # # # =========================
# # # # RUN PIPELINE
# # # # =========================
# # # if __name__ == "__main__":

# # #     tickers = [
# # #         "AAPL", "MSFT", "NVDA",
# # #         "GOOGL", "AMZN", "META", "TSLA"
# # #     ]

# # #     print("\n🚀 Starting FULL PIPELINE...\n")

# # #     # RAW
# # #     raw = get_raw(tickers)
# # #     export("raw", raw)

# # #     # CLEAN
# # #     clean = clean_data(raw)
# # #     export("clean", clean)

# # #     # TECH
# # #     tech = add_technical(clean)
# # #     export("technical", tech)

# # #     # MODEL
# # #     model = add_model_features(tech)
# # #     export("model", model)

# # #     print("\n✅ DONE: Pipeline executed successfully!\n")
# # # """
# # # FINAL ROBUST MARKET PIPELINE (CORRECTED)

# # # RAW → CLEAN → TECH → FINAL → MODEL

# # # ✔ 10 years data
# # # ✔ multi-stock
# # # ✔ MultiIndex-safe (yfinance fix)
# # # ✔ proper preprocessing
# # # ✔ burn-in removal (NaN handling)
# # # ✔ export at each stage
# # # """

# # # import pandas as pd
# # # import numpy as np
# # # import yfinance as yf
# # # import os
# # # import time

# # # # =========================
# # # # CONFIG
# # # # =========================
# # # TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
# # # OUTPUT_DIR = "datasets"
# # # os.makedirs(OUTPUT_DIR, exist_ok=True)


# # # # =========================
# # # # STAGE 1: DATA COLLECTION
# # # # =========================
# # # def fetch_data(tickers):
# # #     all_data = {}

# # #     for t in tickers:
# # #         print(f"Fetching {t}...")

# # #         df = yf.download(t, period="10y", interval="1d", progress=False)

# # #         if df is None or df.empty:
# # #             print(f"Skipping {t}")
# # #             continue

# # #         # ✅ FIX: Handle MultiIndex columns
# # #         if isinstance(df.columns, pd.MultiIndex):
# # #             df.columns = df.columns.get_level_values(0)

# # #         # Normalize column names
# # #         df.columns = [col.lower() for col in df.columns]

# # #         # Optional consistency fix
# # #         if "adj close" in df.columns:
# # #             df = df.rename(columns={"adj close": "adj_close"})

# # #         # Add metadata
# # #         df["ticker"] = t
# # #         df.index = pd.to_datetime(df.index)
# # #         df = df.sort_index()

# # #         all_data[t] = df
# # #         time.sleep(0.5)

# # #     return all_data


# # # # =========================
# # # # STAGE 2: CLEANING
# # # # =========================
# # # def clean_data(data):
# # #     out = {}

# # #     for t, df in data.items():
# # #         df = df.copy()

# # #         # Returns
# # #         df["returns_log"] = np.log(df["close"] / df["close"].shift(1))
# # #         df["returns"] = df["returns_log"]

# # #         out[t] = df

# # #     return out


# # # # =========================
# # # # STAGE 3: FEATURE ENGINEERING
# # # # =========================
# # # def add_features(data):
# # #     out = {}

# # #     for t, df in data.items():
# # #         df = df.copy()

# # #         # Volatility (annualized)
# # #         df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)

# # #         # RSI
# # #         delta = df["close"].diff()
# # #         gain = delta.where(delta > 0, 0).rolling(14).mean()
# # #         loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# # #         rs = gain / (loss + 1e-9)
# # #         df["rsi"] = 100 - (100 / (1 + rs))

# # #         # Volume Z-score
# # #         df["volume_zscore"] = (
# # #             (df["volume"] - df["volume"].rolling(20).mean()) /
# # #             (df["volume"].rolling(20).std() + 1e-9)
# # #         )

# # #         # Hurst exponent (approx)
# # #         def hurst_calc(x):
# # #             if len(x) < 20:
# # #                 return np.nan
# # #             try:
# # #                 lags = range(2, 20)
# # #                 tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
# # #                 poly = np.polyfit(np.log(lags), np.log(tau), 1)
# # #                 return poly[0]
# # #             except:
# # #                 return np.nan

# # #         df["hurst"] = df["close"].rolling(100).apply(hurst_calc, raw=False)

# # #         out[t] = df

# # #     return out


# # # # =========================
# # # # STAGE 4: HANDLE MISSING (IMPORTANT)
# # # # =========================
# # # def handle_missing(data):
# # #     out = {}

# # #     for t, df in data.items():
# # #         df = df.copy()

# # #         before = len(df)

# # #         # Drop NaNs (burn-in period)
# # #         df = df.dropna()

# # #         after = len(df)

# # #         print(f"{t}: removed {before - after} rows (burn-in period)")

# # #         out[t] = df

# # #     return out


# # # # =========================
# # # # STAGE 5: MODEL (REGIME DETECTION)
# # # # =========================
# # # def add_model(data):
# # #     out = {}

# # #     for t, df in data.items():
# # #         df = df.copy()

# # #         latest = df.iloc[-1]

# # #         # Simple rule-based regime
# # #         if latest["volatility"] > 0.4:
# # #             regime = "HIGH_VOL"
# # #         elif latest["hurst"] > 0.6:
# # #             regime = "TRENDING"
# # #         elif latest["volume_zscore"] > 2:
# # #             regime = "VOLUME_SPIKE"
# # #         else:
# # #             regime = "NORMAL"

# # #         df["regime"] = regime
# # #         df["confidence"] = 0.8

# # #         out[t] = df

# # #     return out


# # # # =========================
# # # # EXPORT FUNCTION
# # # # =========================
# # # def export(stage, data):
# # #     combined = []

# # #     for t, df in data.items():
# # #         temp = df.copy()

# # #         # Keep date column for BI tools
# # #         temp["date"] = temp.index

# # #         temp["ticker"] = t
# # #         temp["stage"] = stage

# # #         combined.append(temp)

# # #     if not combined:
# # #         print(f"No data to save for {stage}")
# # #         return

# # #     final = pd.concat(combined)

# # #     path = f"{OUTPUT_DIR}/{stage}_dataset.csv"
# # #     final.to_csv(path, index=False)

# # #     print(f"Saved {stage} → shape={final.shape}")


# # # # =========================
# # # # RUN PIPELINE
# # # # =========================
# # # if __name__ == "__main__":

# # #     print("\nStarting pipeline...\n")

# # #     # RAW
# # #     raw = fetch_data(TICKERS)
# # #     export("raw", raw)

# # #     # CLEAN
# # #     clean = clean_data(raw)
# # #     export("clean", clean)

# # #     # TECH
# # #     tech = add_features(clean)
# # #     export("technical", tech)

# # #     # FINAL (after NaN removal)
# # #     final = handle_missing(tech)
# # #     export("final", final)

# # #     # MODEL
# # #     model = add_model(final)
# # #     export("model", model)

# # #     print("\nPipeline completed successfully.\n")


# # """
# # FINAL ROBUST MARKET PIPELINE (NO FREEZE VERSION)

# # RAW → CLEAN → TECH → FINAL → MODEL

# # ✔ 10 years data
# # ✔ multi-stock
# # ✔ MultiIndex-safe
# # ✔ missing value handling
# # ✔ fast Hurst (NO rolling apply freeze)
# # ✔ burn-in removal
# # ✔ stable feature pipeline
# # """

# # import pandas as pd
# # import numpy as np
# # import yfinance as yf
# # import os
# # import time

# # # =========================
# # # CONFIG
# # # =========================
# # TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
# # OUTPUT_DIR = "datasets"
# # os.makedirs(OUTPUT_DIR, exist_ok=True)


# # # =========================
# # # STAGE 1: RAW DATA
# # # =========================
# # def fetch_data(tickers):
# #     all_data = {}

# #     for t in tickers:
# #         print(f"Fetching {t}...")

# #         df = yf.download(t, period="10y", interval="1d", progress=False)

# #         if df is None or df.empty:
# #             continue

# #         if isinstance(df.columns, pd.MultiIndex):
# #             df.columns = df.columns.get_level_values(0)

# #         df.columns = [c.lower() for c in df.columns]

# #         df.index = pd.to_datetime(df.index)
# #         df = df.sort_index()

# #         all_data[t] = df
# #         time.sleep(0.3)

# #     return all_data


# # # =========================
# # # STAGE 2: CLEAN + MISSING VALUES
# # # =========================
# # def clean_data(data):
# #     out = {}

# #     for t, df in data.items():
# #         df = df.copy()

# #         # missing value handling FIRST (safe)
# #         df = df.ffill().bfill()

# #         # returns
# #         df["returns"] = np.log(df["close"] / df["close"].shift(1))

# #         out[t] = df

# #     return out


# # # =========================
# # # FAST HURST (NO FREEZE)
# # # =========================
# # def hurst_fast(series):
# #     series = np.array(series.dropna())

# #     if len(series) < 200:
# #         return np.nan

# #     lags = range(2, 20)
# #     tau = []

# #     for lag in lags:
# #         diff = np.subtract(series[lag:], series[:-lag])
# #         tau.append(np.std(diff))

# #     tau = np.array(tau)

# #     if np.any(tau <= 0):
# #         return np.nan

# #     poly = np.polyfit(np.log(lags), np.log(tau), 1)
# #     return poly[0]


# # # =========================
# # # STAGE 3: FEATURES (SAFE)
# # # =========================
# # def add_features(data):
# #     out = {}

# #     for t, df in data.items():
# #         df = df.copy()

# #         # volatility
# #         df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)

# #         # RSI
# #         delta = df["close"].diff()
# #         gain = delta.clip(lower=0).rolling(14).mean()
# #         loss = (-delta.clip(upper=0)).rolling(14).mean()
# #         rs = gain / (loss + 1e-9)
# #         df["rsi"] = 100 - (100 / (1 + rs))

# #         # volume z-score
# #         df["volume_zscore"] = (
# #             (df["volume"] - df["volume"].rolling(20).mean()) /
# #             (df["volume"].rolling(20).std() + 1e-9)
# #         )

# #         # FAST HURST (computed ONCE, NOT rolling)
# #         df["hurst"] = hurst_fast(df["close"])

# #         out[t] = df

# #     return out


# # # =========================
# # # STAGE 4: FINAL CLEAN (burn-in removal)
# # # =========================
# # def handle_missing(data):
# #     out = {}

# #     for t, df in data.items():
# #         df = df.copy()

# #         before = len(df)

# #         # remove NaN-heavy warmup period
# #         df = df.dropna()

# #         after = len(df)

# #         print(f"{t}: removed {before - after} rows")

# #         out[t] = df

# #     return out


# # # =========================
# # # STAGE 5: MODEL
# # # =========================
# # def add_model(data):
# #     out = {}

# #     for t, df in data.items():
# #         df = df.copy()

# #         latest = df.iloc[-1]

# #         if latest["volatility"] > 0.4:
# #             regime = "HIGH_VOL"
# #         elif latest["hurst"] > 0.6:
# #             regime = "TRENDING"
# #         elif latest["volume_zscore"] > 2:
# #             regime = "VOLUME_SPIKE"
# #         else:
# #             regime = "NORMAL"

# #         df["regime"] = regime
# #         df["confidence"] = 0.8

# #         out[t] = df

# #     return out


# # # =========================
# # # EXPORT
# # # =========================
# # def export(stage, data):
# #     all_df = []

# #     for t, df in data.items():
# #         temp = df.copy()
# #         temp["ticker"] = t
# #         temp["stage"] = stage
# #         temp["date"] = temp.index
# #         all_df.append(temp)

# #     final = pd.concat(all_df)

# #     path = f"{OUTPUT_DIR}/{stage}.csv"
# #     final.to_csv(path, index=False)

# #     print(f"Saved {stage} → shape={final.shape}")


# # # =========================
# # # RUN PIPELINE
# # # =========================
# # if __name__ == "__main__":

# #     print("Starting pipeline...\n")

# #     raw = fetch_data(TICKERS)
# #     export("raw", raw)

# #     clean = clean_data(raw)
# #     export("clean", clean)

# #     tech = add_features(clean)
# #     export("technical", tech)

# #     final = handle_missing(tech)
# #     export("final", final)

# #     model = add_model(final)
# #     export("model", model)

# #     print("\nDONE 🚀")


# """
# FINAL ROBUST MARKET PIPELINE

# RAW → CLEAN → TECH → FINAL → MODEL

# ✔ 10 years data
# ✔ multi-stock
# ✔ no freeze (fixed Hurst)
# ✔ proper missing value handling
# ✔ correct datetime export (no “hashed date” issue)
# ✔ clean stage-wise CSV outputs
# """

# import pandas as pd
# import numpy as np
# import yfinance as yf
# import os
# import time

# # =========================
# # CONFIG
# # =========================
# TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
# OUTPUT_DIR = "datasets"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # =========================
# # STAGE 1: RAW DATA
# # =========================
# def fetch_data(tickers):
#     all_data = {}

#     for t in tickers:
#         print(f"Fetching {t}...")

#         df = yf.download(t, period="10y", interval="1d", progress=False)

#         if df is None or df.empty:
#             continue

#         # Fix MultiIndex if present
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = df.columns.get_level_values(0)

#         df.columns = [c.lower() for c in df.columns]

#         df.index = pd.to_datetime(df.index)
#         df = df.sort_index()

#         all_data[t] = df
#         time.sleep(0.3)

#     return all_data


# # =========================
# # STAGE 2: CLEANING
# # =========================
# def clean_data(data):
#     out = {}

#     for t, df in data.items():
#         df = df.copy()

#         # Fill missing values safely
#         df = df.ffill().bfill()

#         # Log returns
#         df["returns"] = np.log(df["close"] / df["close"].shift(1))

#         out[t] = df

#     return out


# # =========================
# # FAST HURST (NO FREEZE)
# # =========================
# def hurst_exponent(series):
#     series = np.array(series.dropna())

#     if len(series) < 200:
#         return np.nan

#     lags = range(2, 20)
#     tau = []

#     for lag in lags:
#         diff = np.subtract(series[lag:], series[:-lag])
#         tau.append(np.std(diff))

#     tau = np.array(tau)

#     if np.any(tau <= 0):
#         return np.nan

#     poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
#     return poly[0]


# # =========================
# # STAGE 3: FEATURES
# # =========================
# def add_features(data):
#     out = {}

#     for t, df in data.items():
#         df = df.copy()

#         # Volatility
#         df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)

#         # RSI
#         delta = df["close"].diff()
#         gain = delta.clip(lower=0).rolling(14).mean()
#         loss = (-delta.clip(upper=0)).rolling(14).mean()
#         rs = gain / (loss + 1e-9)
#         df["rsi"] = 100 - (100 / (1 + rs))

#         # Volume Z-score
#         df["volume_zscore"] = (
#             (df["volume"] - df["volume"].rolling(20).mean()) /
#             (df["volume"].rolling(20).std() + 1e-9)
#         )

#         # Hurst (FAST, single-pass per ticker)
#         df["hurst"] = hurst_exponent(df["close"])

#         out[t] = df

#     return out


# # =========================
# # STAGE 4: FINAL CLEAN (burn-in removal)
# # =========================
# def handle_missing(data):
#     out = {}

#     for t, df in data.items():
#         df = df.copy()

#         before = len(df)

#         # Remove NaNs after feature creation
#         df = df.dropna()

#         after = len(df)

#         print(f"{t}: removed {before - after} rows")

#         out[t] = df

#     return out


# # =========================
# # STAGE 5: MODEL (REGIME)
# # =========================
# def add_model(data):
#     out = {}

#     for t, df in data.items():
#         df = df.copy()

#         latest = df.iloc[-1]

#         if latest["volatility"] > 0.4:
#             regime = "HIGH_VOL"
#         elif latest["hurst"] > 0.6:
#             regime = "TRENDING"
#         elif latest["volume_zscore"] > 2:
#             regime = "VOLUME_SPIKE"
#         else:
#             regime = "NORMAL"

#         df["regime"] = regime
#         df["confidence"] = 0.8

#         out[t] = df

#     return out


# # =========================
# # EXPORT (FIXED DATE HANDLING)
# # =========================
# def export(stage, data):
#     all_df = []

#     for t, df in data.items():
#         temp = df.copy()

#         # ✅ FIX: proper datetime column (NO HASHING ISSUE)
#         temp = temp.reset_index()
#         temp.rename(columns={"index": "date"}, inplace=True)

#         temp["ticker"] = t
#         temp["stage"] = stage

#         all_df.append(temp)

#     final = pd.concat(all_df)

#     path = f"{OUTPUT_DIR}/{stage}.csv"
#     final.to_csv(path, index=False)

#     print(f"Saved {stage} → shape={final.shape}")


# # =========================
# # RUN PIPELINE
# # =========================
# if __name__ == "__main__":

#     print("Starting pipeline...\n")

#     raw = fetch_data(TICKERS)
#     export("raw", raw)

#     clean = clean_data(raw)
#     export("clean", clean)

#     tech = add_features(clean)
#     export("technical", tech)

#     final = handle_missing(tech)
#     export("final", final)

#     model = add_model(final)
#     export("model", model)

#     print("\nDONE 🚀")

import pandas as pd
import numpy as np
import yfinance as yf
import os
import time

# =========================
# CONFIG
# =========================
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
OUTPUT_DIR = "datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STAGE 1: RAW DATA
# =========================
def fetch_data(tickers):
    all_data = {}
    for t in tickers:
        print(f"Fetching {t}...")
        df = yf.download(t, period="10y", interval="1d", progress=False)
        if df is None or df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        all_data[t] = df
        time.sleep(0.3)
    return all_data

# =========================
# STAGE 2: CLEANING
# =========================
def clean_data(data):
    out = {}
    for t, df in data.items():
        df = df.copy()
        df = df.ffill().bfill()
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
        out[t] = df
    return out

# =========================
# FAST HURST
# =========================
def hurst_exponent(series):
    series = np.array(series.dropna())
    if len(series) < 200: return np.nan
    lags = range(2, 20)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    if np.any(np.array(tau) <= 0): return np.nan
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]

# =========================
# STAGE 3: FEATURES
# =========================
def add_features(data):
    out = {}
    for t, df in data.items():
        df = df.copy()
        df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["volume_zscore"] = ((df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9))
        df["hurst"] = hurst_exponent(df["close"])
        out[t] = df
    return out

# =========================
# STAGE 4: CLEAN UP
# =========================
def handle_missing(data):
    out = {t: df.dropna() for t, df in data.items()}
    return out

# =========================
# STAGE 5: MODEL (ROW-BY-ROW)
# =========================
def add_model(data):
    out = {}
    def classify(row):
        if row["volatility"] > 0.4: return "HIGH_VOL"
        elif row["hurst"] > 0.6: return "TRENDING"
        elif row["volume_zscore"] > 2: return "VOLUME_SPIKE"
        else: return "NORMAL"
        
    for t, df in data.items():
        df = df.copy()
        df["regime"] = df.apply(classify, axis=1)
        df["confidence"] = 0.8
        out[t] = df
    return out

# =========================
# EXPORT FUNCTION (STAGE-WISE)
# =========================
def export(stage, data):
    all_df = []
    for t, df in data.items():
        temp = df.copy().reset_index()
        temp.rename(columns={"index": "date"}, inplace=True)
        temp["ticker"] = t
        temp["stage"] = stage
        all_df.append(temp)
    
    final = pd.concat(all_df)
    final.to_csv(f"{OUTPUT_DIR}/{stage}.csv", index=False)
    print(f"Saved {stage}.csv → shape={final.shape}")

# =========================
# RUN PIPELINE
# =========================
if __name__ == "__main__":
    print("Starting pipeline...\n")
    
    raw = fetch_data(TICKERS)
    export("raw", raw)
    
    clean = clean_data(raw)
    export("clean", clean)
    
    tech = add_features(clean)
    export("technical", tech)
    
    final = handle_missing(tech)
    export("final", final)
    
    model = add_model(final)
    export("model", model)
    
    print("\nDONE 🚀")