# =============================================================================
# backend/core/engine.py — Main Orchestrator (Background Engine)
# =============================================================================
import asyncio
import logging
import time
import threading
from datetime import datetime
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import *
from backend.core.data_fetcher import get_fetcher
from backend.core.indicators   import get_indicator_engine, get_mtf_analyzer
from backend.ml.ml_engine      import get_ml_engine
from backend.core.sentiment    import get_sentiment_analyzer
from backend.strategies.signal_engine import get_signal_engine
from backend.core.trade_manager import get_trade_manager
from backend.utils.telegram_notifier  import get_notifier

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Central orchestrator. Runs three loops in background threads:
      1. price_loop  — every PRICE_UPDATE_SEC seconds
      2. signal_loop — every SIGNAL_UPDATE_SEC seconds
      3. trade_loop  — every 5 seconds (SL/TP management)
    """

    def __init__(self):
        self.fetcher   = get_fetcher()
        self.ind_eng   = get_indicator_engine()
        self.mtf_eng   = get_mtf_analyzer()
        self.ml_eng    = get_ml_engine()
        self.sentiment = get_sentiment_analyzer()
        self.signal_eng= get_signal_engine()
        self.trades    = get_trade_manager()
        self.notifier  = get_notifier()
        self.last_trained_time = None
        self.retrain_interval = 7200  # 2 hours

        # Shared state (updated by background threads, read by API)
        self.state = {
            "price"     : None,
            "signal"    : None,
            "sentiment" : None,
            "ml"        : None,
            "mtf"       : None,
            "training"  : False,
            "trained"   : False,
            "error"     : None,
            "started_at": datetime.utcnow().isoformat(),
            "last_signal": None,
            "last_price" : None,
            "price_history": [],   # last 500 price ticks for sparkline
        }
        self._running = False
        self._threads: list = []
        self._dfs_ind: dict = {}

    # ── Start ─────────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        logger.info("Trading engine starting...")

        # Train in background first
        t_train = threading.Thread(target=self._train_loop, daemon=True)
        t_price  = threading.Thread(target=self._price_loop,  daemon=True)
        t_signal = threading.Thread(target=self._signal_loop, daemon=True)
        t_trade  = threading.Thread(target=self._trade_loop,  daemon=True)

        for t in [t_train, t_price, t_signal, t_trade]:
            t.start()
            self._threads.append(t)
        logger.info("All engine threads started")

    # =============================================================================
# backend/core/engine.py — FIXED (NO RETRAIN LOOP + NO CRASH)
# =============================================================================

# ONLY SHOWING CHANGED PARTS

    def _train_loop(self):
        """Train ONLY ONCE at startup"""
        try:
            self.state["training"] = True
            logger.info("Fetching data for training...")

            dfs_raw = self.fetcher.fetch_all_timeframes(force=True)

            if not dfs_raw:
                logger.error("No data for training")
                return

            self._dfs_ind = self.ind_eng.compute_all(dfs_raw)

            metrics = self.ml_eng.train(self._dfs_ind)

            self.state["trained"]  = True
            self.state["training"] = False
            self.last_trained_time = time.time()  # ✅ TRACK TIME

            logger.info(f"Training done: {metrics}")

        except Exception as e:
            logger.error(f"Training error: {e}")
            self.state["error"] = str(e)

    # ── Price loop ────────────────────────────────────────────────────────────
    def _price_loop(self):
        while self._running:
            try:
                price_data = self.fetcher.get_live_price()
                self.state["price"]      = price_data
                self.state["last_price"] = datetime.utcnow().isoformat()
                # Update sparkline
                p = price_data.get("price", 0)
                if p:
                    hist = self.state["price_history"]
                    hist.append({"ts": datetime.utcnow().isoformat(), "price": p})
                    self.state["price_history"] = hist[-500:]
                # Update trade SL/TP
                if p:
                    self.trades.update(p, self.notifier)
            except Exception as e:
                logger.error(f"Price loop error: {e}")
            time.sleep(PRICE_UPDATE_SEC)

    # Signal Loop
    def _signal_loop(self):
        # Wait for training
        while self._running and not self.state["trained"]:
            time.sleep(5)

        while self._running:
            try:

                # 🔁 ✅ ADD IT HERE (TOP OF LOOP)
                if self.last_trained_time:
                    if time.time() - self.last_trained_time > self.retrain_interval:
                        logger.info("Retraining model after interval...")
                        self.ml_eng.is_trained = False   # ✅ reset flag
                        self._train_loop()

                # -----------------------------
                # Refresh data
                dfs_raw = self.fetcher.fetch_all_timeframes()

                if not dfs_raw:
                    logger.warning("No data fetched — skipping cycle")
                    time.sleep(SIGNAL_UPDATE_SEC)
                    continue

                self._dfs_ind = self.ind_eng.compute_all(dfs_raw)

                # MTF analysis
                mtf = self.mtf_eng.analyze(self._dfs_ind)
                self.state["mtf"] = mtf

                # ML prediction
                df_primary = self._dfs_ind.get("15m")
                ml_pred = {}
                if df_primary is not None and not df_primary.empty:
                    ml_pred = self.ml_eng.predict(df_primary, df_primary)

                self.state["ml"] = ml_pred

                # Sentiment
                sent = self.sentiment.get_sentiment()
                self.state["sentiment"] = sent

                # Order book
                ob = self.fetcher.fetch_orderbook()

                # Generate signal
                sig = self.signal_eng.generate(
                    self._dfs_ind, mtf, ml_pred, sent, ob
                )

                self.state["signal"] = sig
                self.state["last_signal"] = datetime.utcnow().isoformat()

                # Auto-open trade
                if (sig.get("direction") != "FLAT" and
                    sig.get("confidence", 0) >= SIGNAL_CONFIDENCE_MIN and
                    sig.get("entry") is not None):

                    new_trade = self.trades.open_trade(sig)

                    if new_trade:
                        logger.info(f"Auto-opened trade: {new_trade.direction}")

                        msg = f"""
                        NEW {sig['direction']} SIGNAL
                        ━━━━━━━━━━━━━━━━━━━
                        Entry:      ${sig['entry']:.2f}
                        Stop Loss:  ${sig['stop_loss']:.2f}
                        Take Profit:${sig['take_profit']:.2f}
                        Breakeven:  ${sig['breakeven']:.2f}
                        R:R Ratio:  1:{sig['risk_reward']}
                        Confidence: {sig['confidence']:.1f}%
                        Time:       {datetime.utcnow().strftime("%H:%M UTC")}
                        """

                        self.notifier.send(msg)

            except Exception as e:
                logger.error(f"Signal loop error: {e}", exc_info=True)
                self.state["error"] = str(e)

            time.sleep(SIGNAL_UPDATE_SEC)

    # ── Trade management loop ─────────────────────────────────────────────────
    def _trade_loop(self):
        """Dedicated high-frequency loop just for trade updates."""
        while self._running:
            try:
                price_data = self.state.get("price")
                if price_data and price_data.get("price"):
                    self.trades.update(price_data["price"], self.notifier)
            except Exception as e:
                logger.error(f"Trade loop error: {e}")
            time.sleep(3)

    def stop(self):
        self._running = False


def _tf_to_sec(tf: str) -> int:
    m = {"1m": 60, "3m":180,"5m": 300, "15m": 900,"30m":1800, "1h": 3600, "4h": 14400}
    return m.get(tf, 900)


_engine: Optional[TradingEngine] = None

def get_engine() -> TradingEngine:
    global _engine
    if _engine is None:
        _engine = TradingEngine()
    return _engine
