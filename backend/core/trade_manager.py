# =============================================================================
# backend/core/trade_manager.py — FINAL (WITH TRADE MANAGEMENT)
# =============================================================================
import uuid
import logging
from datetime import datetime
from typing import List, Optional

from backend.db.db import get_connection

logger = logging.getLogger(__name__)


# =============================================================================
# DB FUNCTIONS (UNCHANGED)
# =============================================================================
def save_trade_open(trade):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO trades (
        id, symbol, timeframe, direction, status,
        entry_price, stop_loss, take_profit, breakeven,
        atr, risk_reward, confidence,
        ml_prediction,
        open_time
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        trade.id,
        trade.symbol,
        trade.timeframe,
        trade.direction,
        trade.status,
        trade.entry,
        trade.stop_loss,
        trade.take_profit,
        trade.breakeven,
        trade.atr,
        trade.risk_reward,
        trade.confidence,
        trade.prediction,
        trade.open_time
    )

    cursor.execute(query, values)
    conn.commit()
    conn.close()


def update_trade_close(trade):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    UPDATE trades
    SET
        status = %s,
        close_time = %s,
        close_price = %s,
        result = %s,
        pnl_pct = %s,
        pnl_usd = %s
    WHERE id = %s
    """

    values = (
        trade.status,
        trade.close_time,
        trade.close_price,
        trade.result,
        trade.pnl_pct,
        trade.pnl_usd,
        trade.id
    )

    cursor.execute(query, values)
    conn.commit()
    conn.close()


# =============================================================================
# TRADE CLASS
# =============================================================================
class Trade:
    def __init__(self, signal: dict):
        self.id = str(uuid.uuid4())[:8]

        self.symbol = signal.get("symbol", "BTCUSDT")
        self.timeframe = signal.get("timeframe", "1m")

        self.direction = signal["direction"]
        self.entry = signal["entry"]
        self.stop_loss = signal["stop_loss"]
        self.take_profit = signal["take_profit"]
        self.breakeven = signal.get("breakeven", self.entry)

        self.risk_reward = signal.get("risk_reward", 0)
        self.atr = signal.get("atr", abs(self.entry - self.stop_loss))
        self.confidence = signal.get("confidence", 0)

        self.prediction = signal.get("ml_prediction", None)

        self.open_time = datetime.utcnow()
        self.close_time = None

        self.status = "OPEN"
        self.close_price = None
        self.close_reason = None

        self.pnl_pct = None
        self.pnl_usd = None
        self.result = None

        # 🔥 NEW: trade management flags
        self.be_moved = False
        self.partial_closed = False

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status,
            "confidence": self.confidence,
            "risk_reward": self.risk_reward,
            "ml_prediction": self.prediction,
            "open_time": str(self.open_time),
            "close_time": str(self.close_time) if self.close_time else None,
            "result": self.result,
            "pnl_pct": self.pnl_pct,
        }


# =============================================================================
# TRADE MANAGER
# =============================================================================
class TradeManager:

    def __init__(self):
        self._trades: List[Trade] = []

    # -------------------------------------------------------------------------
    def open_trade(self, signal: dict) -> Optional[Trade]:

        if signal.get("direction") == "FLAT":
            return None

        if len(self.active_trades()) >= 2:
            return None

        trade = Trade(signal)
        self._trades.append(trade)

        save_trade_open(trade)

        logger.info(f"Trade OPENED: {trade.direction} @ {trade.entry}")

        return trade

    # -------------------------------------------------------------------------
    def update(self, price: float, notifier=None):
        for trade in self.active_trades():
            self._update_trade(trade, price, notifier)

    def _update_trade(self, t: Trade, price: float, notifier=None):

        # ✅ FIX 1: Ignore bad price
        if price is None or price == 0:
            logger.warning(f"Invalid price received for trade {t.id}")
            return

        is_long = t.direction == "LONG"

        # 🔥 CALCULATE R (RISK)
        risk = abs(t.entry - t.stop_loss)

        if risk == 0:
            return

        # ---------------------------------------------------------------------
        # DEBUG LOG (VERY IMPORTANT)
        logger.info(f"[CHECK] {t.id} | Price: {price} | TP: {t.take_profit} | SL: {t.stop_loss}")

        # ---------------------------------------------------------------------
        # ✅ 1. BREAKEVEN AT 1R
        # ---------------------------------------------------------------------
        if not t.be_moved:
            if (is_long and price >= t.entry + risk) or \
            (not is_long and price <= t.entry - risk):
                t.stop_loss = t.entry
                t.be_moved = True
                logger.info(f"BE moved: {t.id}")

        # ---------------------------------------------------------------------
        # ✅ 2. PARTIAL CLOSE AT 1.5R
        # ---------------------------------------------------------------------
        if not t.partial_closed:
            if (is_long and price >= t.entry + 1.5 * risk) or \
            (not is_long and price <= t.entry - 1.5 * risk):
                t.partial_closed = True
                logger.info(f"Partial TP hit: {t.id}")

        # ---------------------------------------------------------------------
        # ✅ 3. TRAILING STOP AT 2R
        # ---------------------------------------------------------------------
        if (is_long and price >= t.entry + 2 * risk):
            t.stop_loss = price - risk

        elif (not is_long and price <= t.entry - 2 * risk):
            t.stop_loss = price + risk

        # ---------------------------------------------------------------------
        # ✅ FIX 2: ROBUST TP CHECK (USE LIVE PRICE)
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # ✅ TP HIT
        # ---------------------------------------------------------------------
        if is_long and price >= t.take_profit:
            logger.info(f"TP HIT (LONG): {t.id}")
            self._close(t, price, "TP_HIT", notifier)
            return

        elif not is_long and price <= t.take_profit:
            logger.info(f"TP HIT (SHORT): {t.id}")
            self._close(t, price, "TP_HIT", notifier)
            return


        # ---------------------------------------------------------------------
        # ✅ SL HIT
        # ---------------------------------------------------------------------
        if is_long and price <= t.stop_loss:
            logger.info(f"SL HIT (LONG): {t.id}")
            self._close(t, price, "SL_HIT", notifier)

        elif not is_long and price >= t.stop_loss:
            logger.info(f"SL HIT (SHORT): {t.id}")
            self._close(t, price, "SL_HIT", notifier)

        # ---------------------------------------------------------------------
        # ✅ 1. BREAKEVEN AT 1R
        # ---------------------------------------------------------------------
        if not t.be_moved:
            if (is_long and price >= t.entry + risk) or \
               (not is_long and price <= t.entry - risk):
                t.stop_loss = t.entry
                t.be_moved = True
                logger.info(f"BE moved: {t.id}")

        # ---------------------------------------------------------------------
        # ✅ 2. PARTIAL CLOSE AT 1.5R
        # ---------------------------------------------------------------------
        if not t.partial_closed:
            if (is_long and price >= t.entry + 1.5 * risk) or \
               (not is_long and price <= t.entry - 1.5 * risk):
                t.partial_closed = True
                logger.info(f"Partial TP hit: {t.id}")

        # ---------------------------------------------------------------------
        # ✅ 3. TRAILING STOP AT 2R
        # ---------------------------------------------------------------------
        if (is_long and price >= t.entry + 2 * risk):
            t.stop_loss = price - risk

        elif (not is_long and price <= t.entry - 2 * risk):
            t.stop_loss = price + risk

        # ---------------------------------------------------------------------
        # TP HIT
        if (is_long and price >= t.take_profit) or \
           (not is_long and price <= t.take_profit):
            self._close(t, t.take_profit, "TP_HIT", notifier)

        # SL HIT
        elif (is_long and price <= t.stop_loss) or \
             (not is_long and price >= t.stop_loss):
            self._close(t, t.stop_loss, "SL_HIT", notifier)

    # -------------------------------------------------------------------------
    def _close(self, t: Trade, price: float, reason: str, notifier=None):

        # 🚨 Prevent double closing
        if t.status == "CLOSED":
            return

        t.close_price = price
        t.close_reason = reason
        t.close_time = datetime.utcnow()
        t.status = "CLOSED"

        # ✅ PnL Calculation (correct)
        if t.direction == "LONG":
            t.pnl_pct = (price - t.entry) / t.entry * 100
        else:
            t.pnl_pct = (t.entry - price) / t.entry * 100

        # ❗ FIX: USD should NOT equal %
        if hasattr(t, "risk_amount"):
            t.pnl_usd = (t.pnl_pct / 100) * t.risk_amount
        else:
            t.pnl_usd = t.pnl_pct  # fallback

        # ✅ Result classification
        if t.pnl_pct > 0.01:
            t.result = "WIN"
        elif t.pnl_pct < -0.01:
            t.result = "LOSS"
        else:
            t.result = "BREAKEVEN"

        # ✅ Save to DB
        update_trade_close(t)

        # ✅ Logging
        logger.info(f"Trade CLOSED: {t.id} | {t.result} | {t.pnl_pct:.2f}% | {reason}")

        # ✅ Telegram / notifier
        if notifier:
            try:
                notifier.send(
                    f"📉 Trade Closed\n"
                    f"ID: {t.id}\n"
                    f"Result: {t.result}\n"
                    f"PnL: {t.pnl_pct:.2f}%\n"
                    f"Reason: {reason}"
                )
            except Exception as e:
                logger.error(f"Notifier failed: {e}")

    # -------------------------------------------------------------------------
    def active_trades(self) -> List[Trade]:
        return [t for t in self._trades if t.status != "CLOSED"]

    def all_trades(self) -> List[Trade]:
        return self._trades

    def summary(self):
        closed = [t for t in self._trades if t.status == "CLOSED"]
        open_trades = [t for t in self._trades if t.status != "CLOSED"]

        wins = [t for t in closed if t.pnl_pct and t.pnl_pct > 0]
        total_pnl = sum(t.pnl_usd or 0 for t in closed)

        return {
            "total_trades": len(self._trades),
            "open_trades": len(open_trades),
            "closed_trades": len(closed),
            "win_rate": round((len(wins) / len(closed)) * 100, 2) if closed else 0,
            "total_pnl": round(total_pnl, 2),
            "active": [t.to_dict() for t in open_trades]
        }
    
    def close_trade_manual(self, trade_id, price, notifier=None):
        for t in self.trades:
            if t.id == trade_id and t.status == "OPEN":
                self._close(t, price, "MANUAL_CLOSE", notifier)
                return {"status": "closed"}

        return {"status": "not_found"}


# =============================================================================
# SINGLETON
# =============================================================================
_manager = TradeManager()

def get_trade_manager():
    return _manager