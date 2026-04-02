# =============================================================================
# backend/ml/ml_engine.py — FIXED (REAL ACCURACY + NO OVERFITTING)
# =============================================================================
import numpy as np
import pandas as pd
import joblib
import logging
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

ML_FEATURES = [
    "ema_score", "ema50_dist",
    "rsi", "rsi_score",
    "macd_hist", "macd_score",
    "bb_pct", "bb_squeeze", "bb_score",
    "atr_pct", "is_volatile",
    "vwap_dist", "above_vwap",
    "vol_ratio", "vol_spike",
    "smc_score", "bos_bull", "bos_bear",
    "swing_high", "swing_low",
]

logger = logging.getLogger(__name__)

class MLEngine:

    def __init__(self):
        self.xgb = None
        self.rf = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self._model_path = MODEL_DIR / "xgb_rf.pkl"
        self._feature_cols = []
        self._load_if_exists()
        self.last_trained_time = None
        self.min_retrain_gap = 7200  # 2 hours

    def label(self, df):
        tp_pct = 0.003
        sl_pct = 0.002

        labels = []

        for i in range(len(df)):
            entry = df["close"].iloc[i]
            future = df.iloc[i+1:i+30]

            tp_hit = any(row["high"] >= entry * (1 + tp_pct) for _, row in future.iterrows())
            sl_hit = any(row["low"] <= entry * (1 - sl_pct) for _, row in future.iterrows())

            if tp_hit and not sl_hit:
                labels.append(1)  # LONG
            elif sl_hit and not tp_hit:
                labels.append(0)  # SHORT
            else:
                labels.append(None)

        df["label"] = labels
        return df

    def _build_X(self, df):
        cols = [c for c in ML_FEATURES if c in df.columns]
        X = df[cols].copy()

        for c in X.select_dtypes(include="bool").columns:
            X[c] = X[c].astype(float)

        X.fillna(0, inplace=True)
        return X, cols

    # Train
    def train(self, dfs, force=False):
        # ✅ Prevent unnecessary retraining
        if self.is_trained and not force:
            if self.last_trained_time and (time.time() - self.last_trained_time < self.min_retrain_gap):
                logger.info("Skipping training — recently trained")
                return {"status": "skipped"}

        logger.info("Training ML models...")

        df = dfs.get("1m", pd.DataFrame())

        if df.empty or len(df) < 1000:
            return {"error": "Not enough data"}

        df = self.label(df)
        df.dropna(inplace=True)

        X, cols = self._build_X(df)
        y = df["label"].values

        self._feature_cols = cols
        X_sc = self.scaler.fit_transform(X)

        # ✅ TIME-SERIES SAFE SPLIT
        split = int(len(X_sc) * 0.8)
        X_train, X_test = X_sc[:split], X_sc[split:]
        y_train, y_test = y[:split], y[split:]

        self.xgb = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )

        self.rf = RandomForestClassifier(n_estimators=150, max_depth=8)

        self.xgb.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)

        y_pred = self.xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.is_trained = True
        self.last_trained_time = time.time()

        self._save()

        logger.info(f"Real Model Accuracy: {acc * 100:.2f}%")

        return {
            "samples": len(df),
            "features": len(cols),
            "accuracy": round(acc * 100, 2)
        }

    def predict(self, df, df_seq=None):

        if not self.is_trained:
            return {"prediction": "FLAT", "confidence": 0}

        X, _ = self._build_X(df.iloc[[-1]])

        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0

        X = X[self._feature_cols]
        X_sc = self.scaler.transform(X)

        xgb_p = self.xgb.predict_proba(X_sc)[0]
        rf_p = self.rf.predict_proba(X_sc)[0]

        blend = 0.6 * xgb_p + 0.4 * rf_p

        pred = int(np.argmax(blend))
        confidence = round(float(blend[pred]) * 100, 1)

        label_map = {0: "SHORT", 1: "LONG"}
        if confidence < 55:
            return {"prediction": "FLAT", "confidence": confidence}

        return {
            "prediction": label_map[pred],
            "confidence": confidence,
            "p_long": round(float(blend[1]) * 100, 1),
            "p_short": round(float(blend[0]) * 100, 1),
        }

    def _save(self):
        joblib.dump({
            "xgb": self.xgb,
            "rf": self.rf,
            "scaler": self.scaler,
            "features": self._feature_cols
        }, str(self._model_path))

    def _load_if_exists(self):
        if self._model_path.exists():
            try:
                data = joblib.load(str(self._model_path))
                self.xgb = data["xgb"]
                self.rf = data["rf"]
                self.scaler = data["scaler"]
                self._feature_cols = data["features"]
                self.is_trained = True
                logger.info("Loaded existing ML models")
            except Exception as e:
                logger.warning(f"Load failed: {e}")

_ml_engine = MLEngine()

def get_ml_engine():
    return _ml_engine