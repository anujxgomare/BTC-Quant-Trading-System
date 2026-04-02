# =============================================================================
# backend/strategies/signal_engine.py — PRO VERSION (HIGH ACCURACY)
# =============================================================================

import config.settings as SETTINGS
from datetime import datetime


class SignalEngine:

    def generate(self, dfs, mtf, ml_pred, sentiment, orderbook):

        # ── SAFETY CHECK ─────────────────────────────────
        if not dfs or len(dfs) == 0:
            return {"direction": "FLAT", "confidence": 0}

        df = dfs.get("1m")
        if df is None or df.empty:
            df = next(iter(dfs.values()), None)

        if df is None or df.empty:
            return {"direction": "FLAT", "confidence": 0}

        price = float(df["close"].iloc[-1])

        # ── 🔥 LIQUIDITY SWEEP LOGIC ─────────────────────
        high_prev = df["high"].rolling(10).max().iloc[-2]
        low_prev = df["low"].rolling(10).min().iloc[-2]

        sweep_high = price > high_prev
        sweep_low = price < low_prev

        # ── CANDLE CONFIRMATION ──────────────────────────
        confirm = df.iloc[-2]

        body = abs(confirm["close"] - confirm["open"])
        range_ = confirm["high"] - confirm["low"]

        strong_candle = (range_ > 0) and (body / range_ > 0.6)

        bullish = confirm["close"] > confirm["open"]
        bearish = confirm["close"] < confirm["open"]

        # ── ML OUTPUT ────────────────────────────────────
        ml_dir = ml_pred.get("prediction", "FLAT")
        ml_conf = ml_pred.get("confidence", 0)

        # ── TREND FILTER (MTF) ───────────────────────────
        trend_bias = mtf.get("bias", "NEUTRAL") if mtf else "NEUTRAL"

        # ── VOLATILITY FILTER ────────────────────────────
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]

        if atr != atr or atr == 0:
            atr = price * 0.002

        low_volatility = atr < price * 0.001

        # ── 🎯 SCORING SYSTEM (KEY UPGRADE) ──────────────
        score = 0
        direction = "FLAT"
        reason = "NO SIGNAL"

        if sweep_low or sweep_high:
            score += 1

        if strong_candle:
            score += 1

        if ml_conf > 60:
            score += 1

        if trend_bias != "NEUTRAL":
            score += 1

        if low_volatility:
            score -= 1  # avoid choppy market

        # ── 🎯 FINAL DECISION ────────────────────────────
        if score >= 3:

            # LONG CONDITIONS
            if sweep_low and bullish and ml_dir == "LONG":

                # Trend filter
                if trend_bias == "BEARISH":
                    return {"direction": "FLAT", "confidence": 0}

                direction = "LONG"
                reason = "HIGH PROBABILITY LONG"

            # SHORT CONDITIONS
            elif sweep_high and bearish and ml_dir == "SHORT":

                if trend_bias == "BULLISH":
                    return {"direction": "FLAT", "confidence": 0}

                direction = "SHORT"
                reason = "HIGH PROBABILITY SHORT"

        # ⚡ ML fallback (RARE)
        elif ml_conf > 80:
            direction = ml_dir
            reason = "ML STRONG CONFIDENCE"

        # ❌ NO TRADE
        if direction == "FLAT":
            return {"direction": "FLAT", "confidence": 0}

        # ── TP/SL (IMPROVED RR) ──────────────────────────
        if direction == "LONG":
            sl = price - atr * SETTINGS.ATR_SL_MULT
            tp = price + atr * (SETTINGS.ATR_TP_MULT * 1.2)  # slightly better RR
        else:
            sl = price + atr * SETTINGS.ATR_SL_MULT
            tp = price - atr * (SETTINGS.ATR_TP_MULT * 1.2)

        rr = abs(tp - price) / abs(price - sl)

        breakeven = price

        # ── FINAL SIGNAL ────────────────────────────────
        return {
            "direction": direction,
            "confidence": ml_conf,
            "entry": price,
            "stop_loss": sl,
            "take_profit": tp,
            "breakeven": breakeven,
            "risk_reward": round(rr, 2),
            "ml_prediction": ml_dir,
            "reason": reason,

            # 🔥 IMPROVED SCORE OUTPUT
            "raw_score": round(score / 4, 2),

            "timestamp": datetime.utcnow().isoformat()
        }


def get_signal_engine():
    return SignalEngine()