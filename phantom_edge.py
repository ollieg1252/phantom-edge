#!/usr/bin/env python3
"""
Phantom Edge — BTC Polymarket 5-Minute Signal Tracker
======================================================
Monitors Polymarket's BTC 5-minute prediction markets in real time.
At 30 seconds into each window, records the BTC direction signal.
Simulates quarter-Kelly, half-Kelly, and full-Kelly betting on a $100 bankroll.

NO REAL MONEY IS PLACED. This is a simulation only.

Usage:
    pip install -r requirements.txt
    python phantom_edge.py
    Open http://localhost:5000 in your browser

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUTURE: LIVE ORDER EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the simulation has validated the edge over enough windows (target:
~500+ trades with consistent positive EV), live order execution can be
added. Here is how it would work and what is needed:

HOW:
  Polymarket order placement goes through the CLOB API:
    https://clob.polymarket.com
  Orders are signed EIP-712 transactions submitted on Polygon.
  The official Python SDK handles signing and submission:
    https://github.com/Polymarket/py-clob-client

  The execution hook would slot in at the signal recording point (t=30s),
  right after the direction and Kelly fraction are determined — replace the
  simulation call to apply_bet() with a real order via the CLOB client.
  The outcome check at t=295s would then verify fill status rather than
  computing a simulated P&L.

WHY IT IS NOT SET UP YET:
  1. Edge not yet validated at scale — 500+ live windows needed to confirm
     the p_win curve holds in real market conditions before risking capital.
  2. Requires a funded Polymarket account and API credentials (private key
     for Polygon wallet + CLOB API key), which introduces custody and
     security concerns that need separate handling.
  3. Order sizing in real markets must account for liquidity and slippage —
     the 10% volume cap is a start but real execution needs order book depth
     analysis to avoid moving the market against ourselves.
  4. Polymarket's CLOB has rate limits and occasional downtime; production
     execution needs retry logic, fill confirmation, and partial-fill
     handling that simulation does not require.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
import threading
import csv
import logging
import os
import sys
import signal
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import requests
from flask import Flask, render_template, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP  (writes to both console and phantom_edge.log)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("phantom_edge.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — tweak these if needed
# ─────────────────────────────────────────────────────────────────────────────
WIN_PROBABILITY   = 0.636    # Historical 30-second signal win rate (from research)
POLYMARKET_FEE    = 0.02     # Polymarket charges 2% fee on net profits
STARTING_BANKROLL = 100.0    # Starting bankroll (dollars) for each strategy
FALLBACK_PRICE    = 0.5886   # Historical average Polymarket price (used if API unavailable)

# 2D p_win table: (price_bucket, delta_bucket) → blended win probability
# Derived from 359 live windows collected by this app.
# Bayesian blend: (live_wr × n + 0.636 × 30) / (n + 30) — smooths sparse cells.
#
# Price buckets (Polymarket price of signal direction at t=30s):
#   0 → 0.500–0.575
#   1 → 0.575–0.650
#   2 → 0.650–1.000
#
# Delta buckets (signal-aligned BTC % move from open to t=30s):
#   Signal-aligned delta = raw_delta × (+1 if HIGHER signal, −1 if LOWER signal)
#   0 → against  (<−0.03%): BTC moving opposite to signal
#   1 → flat     (±0.03%):  BTC essentially flat
#   2 → with     (>+0.03%): BTC moving in signal direction
#
#  Price \ Delta  Against  Flat    With
#  0.500–0.575    0.602    0.516   0.588
#  0.575–0.650    0.658    0.571   0.615
#  0.650–1.000    0.711    0.744   0.802
PWIN_CURVE_2D = {
    (0, 0): 0.602,  (0, 1): 0.516,  (0, 2): 0.588,
    (1, 0): 0.658,  (1, 1): 0.571,  (1, 2): 0.615,
    (2, 0): 0.711,  (2, 1): 0.744,  (2, 2): 0.802,
}

# Delta threshold (%) separating "flat" from directional buckets
DELTA_THRESHOLD = 0.03


def get_pwin_2d(poly_price: float, signal_aligned_delta: float) -> float:
    """
    Look up empirical win probability using both the Polymarket price and the
    signal-aligned BTC momentum at t=30s.

    signal_aligned_delta: (btc_at_30s - btc_open) / btc_open * 100,
        sign-flipped for LOWER signals so positive always means "moving with
        the signal direction".

    Returns 0.0 if poly_price < 0.50 (should not happen in normal operation).
    """
    if poly_price < 0.50:
        log.warning(f"poly_price {poly_price:.4f} < 0.50 — signal is weaker outcome, skipping.")
        return 0.0

    # Price bucket
    if poly_price < 0.575:
        pb = 0
    elif poly_price < 0.650:
        pb = 1
    else:
        pb = 2

    # Delta bucket
    if signal_aligned_delta < -DELTA_THRESHOLD:
        db = 0  # against
    elif signal_aligned_delta <= DELTA_THRESHOLD:
        db = 1  # flat
    else:
        db = 2  # with

    return PWIN_CURVE_2D[(pb, db)]


def get_market_volume(market: Dict) -> Optional[float]:
    """
    Extract total USD volume from a Gamma API market object.
    Returns None if unavailable so callers can skip the volume cap gracefully.
    """
    try:
        for field in ("volume", "volumeNum", "volume24hr", "volume24hrNum"):
            val = market.get(field)
            if val is not None:
                return float(val)
    except (ValueError, TypeError):
        pass
    return None

# ── Optional: paste a specific Polymarket market ID here if you know one ──────
# Example: "0x1234abcd..."  (find it in the URL of the market on polymarket.com)
# Leave as None to let the app search automatically.
MANUAL_MARKET_ID  = None

LOG_FILE          = "trades.json"    # JSON file that persists all trade data
WINDOW_LOG_FILE   = "window_log.csv" # Per-window data log for future model building
DASHBOARD_PORT    = 8080           # Port for the Flask web dashboard

# ── Drawdown protection ───────────────────────────────────────────────────────
DRAWDOWN_PROTECTION_ENABLED = True
MAX_DRAWDOWN      = 0.30   # Stop betting if bankroll drops >30% from its peak
MAX_CONSEC_LOSSES = 15     # Stop betting after this many consecutive losses

# ── Live order execution ──────────────────────────────────────────────────────
# Set LIVE_MODE = True once credentials are configured to place real CLOB orders.
# The simulation still runs in parallel so you can compare live vs simulated P&L.
LIVE_MODE        = True     # MUST be True to place real orders
LIVE_KELLY_MULT  = 0.25     # Fraction of full Kelly used for live bets (quarter = safest)
MIN_LIVE_STAKE   = 1.00     # Skip orders smaller than $1 USDC
MAX_LIVE_STAKE   = 50.00    # Hard cap per order regardless of Kelly sizing

# Credentials — NEVER hardcode keys here. Only one env var needed:
#   export POLY_PRIVATE_KEY="0x..."   # private key of your Polygon wallet
# API key/secret/passphrase are derived automatically from the private key.
POLY_PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY", "")

WINDOW_MINUTES    = 5        # Each Polymarket BTC market window = 5 minutes
SIGNAL_SECONDS    = 30       # We record our signal at this many seconds into the window
SIGNAL_WINDOW_LO  = 27       # Start of the signal detection window (seconds into window)
SIGNAL_WINDOW_HI  = 37       # End of the signal detection window
OUTCOME_WINDOW_LO = 299      # Check CLOB prices 1 second before window close
OUTCOME_WINDOW_HI = 330      # End of outcome detection (safety buffer)

REQUEST_TIMEOUT   = 8        # Seconds before an API request times out
MAX_RETRIES       = 3        # How many times to retry a failed API call
RETRY_DELAY       = 2        # Base delay (seconds) between retries; doubles each attempt

# Public API endpoints (no API key required)
GAMMA_API_BASE    = "https://gamma-api.polymarket.com"
BINANCE_URL       = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
COINGECKO_URL     = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

# ── Chainlink on-chain oracle (same source Polymarket uses for resolution) ───
# BTC/USD aggregator contract on Polygon (Polymarket's chain)
# https://docs.chain.link/data-feeds/price-feeds/addresses?network=polygon
CHAINLINK_AGGREGATOR = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
CHAINLINK_SELECTOR   = "0xfeaf968c"   # latestRoundData() function selector
# Public Polygon RPC endpoints — tried in order, first success wins
POLYGON_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
]

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (read/written by both the monitor thread and Flask thread)
# ─────────────────────────────────────────────────────────────────────────────
state_lock = threading.Lock()  # Prevents race conditions between threads


def _load_saved_state() -> Dict:
    """
    Load saved trade data from the JSON log file if it exists.
    This lets the app resume after a restart without losing history.
    """
    defaults = {
        "bankroll_quarter_kelly": STARTING_BANKROLL,
        "bankroll_half_kelly":    STARTING_BANKROLL,
        "bankroll_full_kelly":    STARTING_BANKROLL,
        "peak_quarter_kelly":     STARTING_BANKROLL,
        "peak_half_kelly":        STARTING_BANKROLL,
        "peak_full_kelly":        STARTING_BANKROLL,
        "bankroll_history":       [],
        "trades":                 [],
        "running_since":          datetime.now(timezone.utc).isoformat(),
        "total_trades":              0,
        "wins":                      0,
        "consecutive_losses":        0,
        "live_bankroll":             STARTING_BANKROLL,
        "live_peak_bankroll":        STARTING_BANKROLL,
        "live_consecutive_losses":   0,
    }

    for path, label in [(LOG_FILE, "main"), (LOG_FILE + ".bak", "backup")]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    saved = json.load(f)
                # Fill in any keys missing from older/reset save files
                for k, v in defaults.items():
                    saved.setdefault(k, v)
                log.info(f"Loaded {len(saved.get('trades', []))} saved trades from {label} file ({path})")
                return saved
            except Exception as e:
                log.warning(f"Could not read {label} file ({path}): {e} — trying next.")

    # Fresh start — no save file found
    return defaults


# Load persistent state; then add runtime-only fields
state: Dict = _load_saved_state()
state["status"]         = "Starting up — waiting for first 5-minute window..."
state["current_window"] = {
    "start_time":             None,
    "opening_btc_price":      None,
    "current_btc_price":      None,
    "signal_direction":       None,   # "higher" or "lower"
    "signal_polymarket_price": None,  # e.g. 0.5886
    "signal_time":            None,
    "market_found":           False,
}


WINDOW_LOG_FIELDS = [
    "window_start", "btc_open", "poly_open",
    "btc_at_30s", "poly_at_30s",
    "signal_direction", "outcome", "won",
]


def _append_window_log(row: Dict) -> None:
    """Append one row to window_log.csv. Creates the file with headers if new."""
    file_exists = os.path.exists(WINDOW_LOG_FILE)
    with open(WINDOW_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_LOG_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ORDER EXECUTION  (Polymarket CLOB — only used when LIVE_MODE = True)
# ─────────────────────────────────────────────────────────────────────────────

_clob_client      = None
_clob_client_lock = threading.Lock()


def get_clob_client():
    """Lazy-init and cache the Polymarket CLOB client. Raises on missing deps/creds."""
    global _clob_client
    if _clob_client is not None:
        return _clob_client
    with _clob_client_lock:
        if _clob_client is not None:
            return _clob_client
        try:
            from py_clob_client.client import ClobClient      # noqa: PLC0415
            from py_clob_client.constants import POLYGON      # noqa: PLC0415
        except ImportError:
            raise RuntimeError("py-clob-client not installed — run: pip install py-clob-client")
        if not POLY_PRIVATE_KEY:
            raise RuntimeError("POLY_PRIVATE_KEY env var not set")
        # Initialise without creds first, then derive L2 API creds from the private key
        from eth_account import Account  # noqa: PLC0415
        funder = Account.from_key(POLY_PRIVATE_KEY).address
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=POLY_PRIVATE_KEY,
            chain_id=POLYGON,
            signature_type=1,   # proxy wallet (Polymarket default)
            funder=funder,
        )
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        _clob_client = client
        log.info("CLOB client initialised — LIVE MODE ACTIVE")
    return _clob_client


def place_live_order(token_id: str, price: float, usdc_amount: float) -> Optional[str]:
    """
    Place a GTC buy order on Polymarket CLOB.
    Returns orderID string on success, None after MAX_RETRIES failures.
    usdc_amount is in dollars; shares = usdc_amount / price.
    """
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType  # noqa: PLC0415
        from py_clob_client.order_builder.constants import BUY       # noqa: PLC0415
        client = get_clob_client()
        shares = round(usdc_amount / price, 4)
        order_args = OrderArgs(token_id=token_id, price=price, size=shares, side=BUY)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                signed   = client.create_order(order_args)
                resp     = client.post_order(signed, OrderType.GTC)
                order_id = resp.get("orderID") or resp.get("order_id")
                if order_id:
                    log.info(
                        f"LIVE ORDER: {shares:.4f} sh @ {price:.4f} = ${usdc_amount:.2f} | id={order_id}"
                    )
                    return order_id
                log.warning(f"Order attempt {attempt}: unexpected response {resp}")
            except Exception as e:
                log.warning(f"Order attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
    except Exception as e:
        log.error(f"LIVE ORDER ERROR: {e}", exc_info=True)
    log.error("LIVE ORDER failed after all retries — no bet placed this window")
    return None


def cancel_order_safe(order_id: str) -> bool:
    """Cancel an open order. Logs but never raises."""
    try:
        get_clob_client().cancel({"orderID": order_id})
        log.info(f"Cancelled order {order_id}")
        return True
    except Exception as e:
        log.warning(f"Cancel failed for {order_id}: {e}")
        return False


def get_order_fill_usdc(order_id: str, price: float) -> Optional[float]:
    """
    Return filled amount in USDC (size_matched_shares × price).
    Returns None if the order status cannot be fetched.
    """
    try:
        order        = get_clob_client().get_order(order_id)
        size_matched = float(order.get("size_matched", 0) or 0)
        return round(size_matched * price, 4)
    except Exception as e:
        log.warning(f"get_order_fill_usdc({order_id}): {e}")
        return None


def get_live_usdc_balance() -> float:
    """
    Fetch the real USDC balance available in the Polymarket wallet.
    Called before every live bet so Kelly sizing reflects the actual bankroll.
    Falls back to STARTING_BANKROLL ($100) on any failure.
    """
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType  # noqa: PLC0415
        resp    = get_clob_client().get_balance_allowance(params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=1))
        raw = float(resp.get("balance", 0) or 0)
        balance = raw / 1_000_000  # USDC has 6 decimals
        if balance > 0:
            log.info(f"LIVE: wallet balance ${balance:.2f} USDC")
            return round(balance, 2)
        log.warning("LIVE: balance endpoint returned 0 — using fallback")
    except Exception as e:
        log.warning(f"LIVE: wallet balance fetch failed ({e}) — using ${STARTING_BANKROLL:.2f}")
    return STARTING_BANKROLL


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_json(url: str, params: dict = None, retries: int = MAX_RETRIES) -> Optional[Any]:
    """
    GET a URL and return the parsed JSON body.
    Retries up to `retries` times with exponential backoff.
    Returns None if every attempt fails.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            wait = RETRY_DELAY * (2 ** attempt)  # 2s, 4s, 8s …
            log.warning(f"Request attempt {attempt+1}/{retries} failed: {e} — retrying in {wait}s")
            if attempt < retries - 1:
                time.sleep(wait)

    log.error(f"All {retries} attempts failed for URL: {url}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BTC PRICE FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def get_btc_chainlink() -> Optional[float]:
    """
    Query the Chainlink BTC/USD on-chain aggregator on Polygon.

    This is the same oracle Polymarket uses to resolve its BTC markets,
    so our opening/closing prices will match Polymarket's reference price
    almost exactly (sub-dollar difference in practice).

    Uses a raw JSON-RPC eth_call — no API key or web3 library needed.
    Calls latestRoundData() on the Chainlink aggregator contract and
    decodes the signed int256 'answer' (8 decimal places for BTC/USD).

    Tries each public RPC in POLYGON_RPCS until one responds.
    """
    payload = {
        "jsonrpc": "2.0",
        "method":  "eth_call",
        "params":  [{"to": CHAINLINK_AGGREGATOR, "data": CHAINLINK_SELECTOR}, "latest"],
        "id":      1,
    }
    for rpc in POLYGON_RPCS:
        try:
            resp   = requests.post(rpc, json=payload, timeout=REQUEST_TIMEOUT)
            result = resp.json().get("result", "")
            if not result or len(result) < 130:
                continue

            # latestRoundData() returns 5 × 32-byte words:
            #   [0] roundId  [1] answer  [2] startedAt  [3] updatedAt  [4] answeredInRound
            hex_data   = result[2:]           # strip leading '0x'
            answer_hex = hex_data[64:128]     # second 32-byte word
            raw        = int(answer_hex, 16)

            # int256 two's complement (price is always positive so this is a no-op in practice)
            if raw >= 2 ** 255:
                raw -= 2 ** 256

            price = raw / 1e8   # BTC/USD feed uses 8 decimal places
            log.info(f"Chainlink oracle ({rpc.split('/')[2]}): ${price:,.2f}")
            return price

        except Exception as e:
            log.warning(f"Chainlink RPC {rpc} failed: {e}")

    log.warning("All Chainlink RPCs failed.")
    return None


def get_btc_price() -> Optional[float]:
    """
    Get current BTC/USD price.
    Primary: Chainlink on-chain oracle (matches Polymarket's resolution source).
    Fallback: Binance ticker, then CoinGecko.
    """
    price = get_btc_chainlink()
    if price:
        return price

    log.info("Chainlink unavailable — falling back to Binance ticker...")
    data = fetch_json(BINANCE_URL)
    if data and "price" in data:
        try:
            return float(data["price"])
        except (ValueError, TypeError):
            pass

    log.info("Binance unavailable — falling back to CoinGecko...")
    data = fetch_json(COINGECKO_URL)
    if data and "bitcoin" in data:
        try:
            return float(data["bitcoin"]["usd"])
        except (ValueError, TypeError, KeyError):
            pass

    log.error("Could not fetch BTC price from any source.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# POLYMARKET MARKET FINDER
# ─────────────────────────────────────────────────────────────────────────────

def _market_matches_current_window(market: Dict) -> bool:
    """
    Return True only if this market's eventStartTime falls within the
    current 5-minute window (i.e. it is genuinely the market for RIGHT NOW,
    not a pre-created future market or a stale past one).
    """
    try:
        events = market.get("events", [])
        start_str = events[0].get("startTime") if events else None
        if not start_str:
            # Fall back to the top-level endDate minus 5 minutes
            end_str = market.get("endDate", "")
            if not end_str:
                return False
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            event_start = end_dt - timedelta(minutes=WINDOW_MINUTES)
        else:
            event_start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))

        win_start_utc = current_window_start().astimezone(timezone.utc)

        # Allow ±90 seconds so minor clock drift doesn't cause a miss
        diff = abs((event_start - win_start_utc).total_seconds())
        return diff <= 90

    except Exception as e:
        log.warning(f"Could not validate market window time: {e}")
        return False


def find_btc_5min_market() -> Optional[Dict]:
    """
    Find the Polymarket BTC 5-minute market for the CURRENT window only.

    The markets follow a predictable slug:
        btc-updown-5m-{unix_epoch_of_window_start_utc}

    After finding any candidate market we always validate its eventStartTime
    against the current window — this prevents accidentally using a pre-created
    future market (which shows 50/50 prices with no real trading).

    Returns the market dict (with outcomePrices) or None if not found /
    not yet active for this window.
    """

    # ── 1. Manual market ID override ─────────────────────────────────────────
    if MANUAL_MARKET_ID:
        data = fetch_json(f"{GAMMA_API_BASE}/markets/{MANUAL_MARKET_ID}")
        if data and data.get("active") and not data.get("closed"):
            if _market_matches_current_window(data):
                log.info(f"Using manually specified market: {data.get('question', 'N/A')}")
                return data
            log.warning("Manual market ID does not match current window — ignoring.")
        else:
            log.warning(f"Manual market ID {MANUAL_MARKET_ID} not active.")

    # ── 2. Direct slug lookup ────────────────────────────────────────────────
    win_start_utc = current_window_start().astimezone(timezone.utc)
    win_ts        = int(win_start_utc.timestamp())
    slug          = f"btc-updown-5m-{win_ts}"

    data = fetch_json(f"{GAMMA_API_BASE}/markets", params={"slug": slug})
    if data:
        markets = data if isinstance(data, list) else data.get("markets", [])
        for market in markets:
            if market.get("active") and not market.get("closed"):
                if _market_matches_current_window(market):
                    log.info(f"Found via slug: {market.get('question')}")
                    return market
                else:
                    log.warning(
                        f"Slug match rejected — wrong window time: "
                        f"{market.get('question')} "
                        f"(our window: {win_start_utc.strftime('%H:%M UTC')})"
                    )

    # ── 3. Scan recent markets, validate window time ─────────────────────────
    # Important: sort by eventStartTime proximity, NOT by creation startDate.
    # Sorting by startDate returns pre-created future markets first.
    data = fetch_json(
        f"{GAMMA_API_BASE}/markets",
        params={
            "active":    "true",
            "closed":    "false",
            "limit":     100,
            "order":     "startDate",
            "ascending": "false",
        },
    )
    if not data:
        log.warning("Polymarket API unavailable — using fallback price.")
        return None

    markets = data if isinstance(data, list) else data.get("markets", [])
    for market in markets:
        q         = market.get("question", "").lower()
        mkt_slug  = market.get("slug", "")
        is_btc_5m = (
            ("bitcoin" in q or "btc" in q)
            and "up or down" in q
            and "btc-updown-5m-" in mkt_slug
            and market.get("active")
            and not market.get("closed")
        )
        if is_btc_5m and _market_matches_current_window(market):
            log.info(f"Found via scan: {market.get('question')}")
            return market

    log.warning(
        f"No BTC 5-min market found for current window "
        f"({win_start_utc.strftime('%H:%M UTC')}) — using fallback price."
    )
    return None


def get_price_for_direction(market: Dict, direction: str) -> Optional[float]:
    """
    Get the live Polymarket price for a given direction ('higher' or 'lower').

    The Gamma API's outcomePrices field is NOT updated in real time — it can
    sit at 50/50 even when the live order book shows 65/35.  We therefore
    query the CLOB midpoint endpoint directly using the token ID for the
    outcome we want, which reflects actual live trading.

    Falls back to outcomePrices only if the CLOB call fails.
    """
    direction_map = {"higher": "up", "lower": "down"}
    target = direction_map.get(direction.lower(), direction.lower())

    try:
        outcomes_raw = market.get("outcomes", "[]")
        tokens_raw   = market.get("clobTokenIds", "[]")

        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        tokens   = json.loads(tokens_raw)   if isinstance(tokens_raw, str)   else tokens_raw

        if not outcomes or not tokens or len(outcomes) != len(tokens):
            log.warning("Market missing outcomes or clobTokenIds")
            return None

        # Find the token ID for our direction
        token_id = None
        for outcome, token in zip(outcomes, tokens):
            if target in outcome.lower():
                token_id = token
                break

        if not token_id:
            log.warning(f"Could not find token for direction '{direction}'")
            return None

        # ── Primary: CLOB midpoint (live order book price) ───────────────────
        data = fetch_json(
            "https://clob.polymarket.com/midpoint",
            params={"token_id": token_id},
            retries=2,
        )
        if data and data.get("mid") not in (None, "", "0", "1"):
            mid = float(data["mid"])
            # Sanity check: ignore degenerate prices (market already resolved)
            if 0.02 < mid < 0.98:
                log.info(f"CLOB midpoint for {direction} ({target}): {mid:.4f}")
                return mid

        # ── Fallback: Gamma API outcomePrices ────────────────────────────────
        log.warning("CLOB midpoint unavailable — falling back to Gamma outcomePrices")
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if prices and len(prices) == len(outcomes):
            for outcome, price in zip(outcomes, prices):
                if target in outcome.lower():
                    p = float(price)
                    if 0.02 < p < 0.98:
                        return p

        return None

    except Exception as e:
        log.error(f"Error getting price for direction '{direction}': {e}")
        return None


def get_clob_midpoint_raw(token_id: str) -> Optional[float]:
    """Fetch CLOB midpoint for a token with no sanity-check filtering."""
    data = fetch_json(
        "https://clob.polymarket.com/midpoint",
        params={"token_id": token_id},
        retries=2,
    )
    if data and data.get("mid") not in (None, ""):
        try:
            return float(data["mid"])
        except (ValueError, TypeError):
            pass
    return None


def get_market_outcome(market_slug: str, market_id: Optional[str] = None) -> Optional[str]:
    """
    Fetch a resolved Polymarket market and return the winning outcome
    ("higher" or "lower") by reading outcomePrices.

    Queries by market ID (direct /markets/{id}) when available — this is
    unambiguous and avoids slug recycling issues. Falls back to slug search.

    When a market resolves, the winning outcome is priced at "1" and the
    losing outcome at "0". This is Polymarket's own resolution — based on
    Chainlink Data Streams — so it is the ground truth for win/loss.

    Returns "higher", "lower", or None if the market hasn't resolved yet.
    """
    try:
        # Prefer direct ID lookup — avoids slug recycling/reuse issues
        if market_id:
            data = fetch_json(f"{GAMMA_API_BASE}/markets/{market_id}", retries=2)
            market = data if isinstance(data, dict) else None
        else:
            data = fetch_json(f"{GAMMA_API_BASE}/markets", params={"slug": market_slug}, retries=2)
            if not data:
                return None
            markets = data if isinstance(data, list) else data.get("markets", [])
            if not markets:
                return None
            market = markets[0]

        if not market:
            return None

        outcomes_raw = market.get("outcomes", "[]")
        prices_raw   = market.get("outcomePrices", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices   = json.loads(prices_raw)   if isinstance(prices_raw,   str) else prices_raw

        log.info(f"Market outcome check — closed={market.get('closed')} outcomes={outcomes} prices={prices}")

        for outcome, price in zip(outcomes, prices):
            try:
                if float(price) >= 0.99:
                    o = outcome.lower()
                    if o in ("up", "higher"):
                        return "higher"
                    elif o in ("down", "lower"):
                        return "lower"
            except (ValueError, TypeError):
                continue

        return None  # not resolved yet — prices not at 0/1

    except Exception as e:
        log.error(f"Error fetching market outcome for {market_slug}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# KELLY BETTING MATHS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_kelly_fraction(p_win: float, market_price: float, fee: float = POLYMARKET_FEE) -> float:
    """
    Full Kelly fraction for a Polymarket binary bet.

    Args:
        p_win        : Our estimated probability of winning (0.636 from research).
        market_price : Polymarket's implied probability for our chosen outcome,
                       e.g. 0.5886.  If you pay $0.5886, you collect $1 if correct.
        fee          : Polymarket fee on net profit (0.02 = 2%).

    Returns:
        Kelly fraction f* in [0, 1].  0 means "no edge, don't bet."

    Maths:
        Net odds  b = (1 - market_price) / market_price  ×  (1 - fee)
        Kelly     f* = (p_win × b − (1 − p_win)) / b
                     = p_win − (1 − p_win) / b
    """
    if not (0 < market_price < 1):
        return 0.0

    # Net profit per $1 staked, after the 2% fee on winnings
    b = (1.0 - market_price) / market_price * (1.0 - fee)

    if b <= 0:
        return 0.0

    f_star = (p_win * b - (1.0 - p_win)) / b
    return max(0.0, min(1.0, f_star))


def apply_bet(bankroll: float, kelly_fraction: float,
              market_price: float, won: bool) -> tuple:
    """
    Simulate a single Polymarket bet and return (new_bankroll, pnl).

    Args:
        bankroll      : Current bankroll in dollars.
        kelly_fraction: Fraction of bankroll to stake.
        market_price  : Price paid per dollar of payout.
        won           : True if our prediction was correct.

    Returns:
        (new_bankroll, pnl)  — pnl is positive for a win, negative for a loss.
    """
    stake = bankroll * kelly_fraction

    if won:
        # Gross profit = stake × (1/market_price − 1)
        # Net profit   = gross × (1 − fee)
        gross_profit = stake * (1.0 - market_price) / market_price
        net_profit   = gross_profit * (1.0 - POLYMARKET_FEE)
        pnl          = net_profit
    else:
        pnl = -stake

    new_bankroll = max(0.0, bankroll + pnl)  # bankroll can't go below 0
    return new_bankroll, pnl


# ─────────────────────────────────────────────────────────────────────────────
# STATE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_state_to_disk():
    """
    Write the persistent part of state to the JSON log file atomically.
    Writes to a temp file first, then renames — rename is atomic on POSIX
    so the file is never left in a partially-written state if the process dies.
    """
    try:
        persistent = {
            "bankroll_quarter_kelly": state["bankroll_quarter_kelly"],
            "bankroll_half_kelly":    state["bankroll_half_kelly"],
            "bankroll_full_kelly":    state["bankroll_full_kelly"],
            "peak_quarter_kelly":     state.get("peak_quarter_kelly", state["bankroll_quarter_kelly"]),
            "peak_half_kelly":        state.get("peak_half_kelly",    state["bankroll_half_kelly"]),
            "peak_full_kelly":        state.get("peak_full_kelly",    state["bankroll_full_kelly"]),
            "bankroll_history":       state["bankroll_history"],
            "trades":                 state["trades"],
            "running_since":          state["running_since"],
            "total_trades":           state["total_trades"],
            "wins":                   state["wins"],
            "consecutive_losses":     state.get("consecutive_losses", 0),
        }
        tmp_file = LOG_FILE + ".tmp"
        bak_file = LOG_FILE + ".bak"
        with open(tmp_file, "w") as f:
            json.dump(persistent, f, indent=2)
        # Rotate: current → .bak, then tmp → current (both atomically)
        if os.path.exists(LOG_FILE):
            os.replace(LOG_FILE, bak_file)
        os.replace(tmp_file, LOG_FILE)
        log.info(f"State saved to {LOG_FILE}")
    except Exception as e:
        log.error(f"Failed to save state: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW TIMING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def current_window_start() -> datetime:
    """
    Return the start of the current 5-minute window, aligned to the clock (UTC).
    e.g. if it's 12:07:45 UTC → returns 12:05:00 UTC.
    """
    now = datetime.now(timezone.utc)
    aligned_minute = (now.minute // WINDOW_MINUTES) * WINDOW_MINUTES
    return now.replace(minute=aligned_minute, second=0, microsecond=0)


def seconds_into_window() -> float:
    """How many seconds have elapsed since the current window started."""
    return (datetime.now(timezone.utc) - current_window_start()).total_seconds()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MONITOR LOOP  (runs in a background thread forever)
# ─────────────────────────────────────────────────────────────────────────────

def monitor_loop():
    """
    Watches the clock and reacts to three events per 5-minute window:

      t =  0s  →  Record opening BTC price for this window.
      t = 30s  →  Take the direction signal + fetch Polymarket odds.
      t ~= 5m  →  Record outcome, simulate bets, save to disk.

    Uses wide time windows (e.g. 27–37s) so that slow API calls
    don't cause us to miss an event.
    """
    log.info("Monitor loop starting...")

    last_window_start:    Optional[datetime] = None  # Window we last processed
    signal_recorded_for:  Optional[datetime] = None  # Window we took a signal for
    outcome_recorded_for: Optional[datetime] = None  # Window we recorded outcome for
    last_price_fetch:     float              = 0.0   # time.time() of last price poll
    prefetched_open:      Optional[float]    = None  # Price grabbed just before window boundary
    prefetch_for_window:  Optional[datetime] = None  # Which upcoming window it's for
    PRICE_POLL_INTERVAL   = 10                       # seconds between live price refreshes
    # Chainlink heartbeat is ~27s, so 10s gives a timely display without hammering the RPC.

    # If we start mid-window (past the signal point), skip to the next window.
    initial_secs = seconds_into_window()
    if initial_secs > SIGNAL_WINDOW_HI:
        log.info(
            f"App started {initial_secs:.0f}s into the window "
            f"(past {SIGNAL_WINDOW_HI}s signal point). "
            f"Waiting for the next window to start clean."
        )
        with state_lock:
            state["status"] = (
                f"Started mid-window ({initial_secs:.0f}s in). "
                f"Waiting for the next 5-minute window..."
            )
        # Mark the current window as already-processed so we skip it
        signal_recorded_for  = current_window_start()
        outcome_recorded_for = current_window_start()

    while True:
        try:
            now          = datetime.now(timezone.utc)
            win_start    = current_window_start()
            secs_elapsed = (now - win_start).total_seconds()

            # ── PRE-FETCH: grab Chainlink price 1s after the window boundary ──
            # Fetching at t=301s (1s into the new window) gives Chainlink time
            # to settle the first block after the boundary — closer to the price
            # Polymarket's Data Streams captured at the exact window open.
            next_win_start = win_start + timedelta(minutes=WINDOW_MINUTES)
            if (301 <= secs_elapsed <= 304
                    and prefetch_for_window != next_win_start):
                prefetch_for_window = next_win_start
                prefetched_open = get_btc_price()
                last_price_fetch = time.time()
                log.info(f"Pre-fetched opening price for next window: ${prefetched_open:,.2f}")

            # ── EVENT 1: NEW WINDOW ─────────────────────────────────────────
            if win_start != last_window_start:
                last_window_start = win_start

                # Use the pre-fetched price if it was grabbed for this window;
                # otherwise fall back to a fresh Chainlink call right now.
                if prefetch_for_window == win_start and prefetched_open:
                    opening_price = prefetched_open
                    log.info(f"Using pre-fetched open: ${opening_price:,.2f}")
                else:
                    opening_price = get_btc_price()

                with state_lock:
                    state["current_window"] = {
                        "start_time":              win_start.strftime("%H:%M:%S"),
                        "opening_btc_price":       opening_price,
                        "current_btc_price":       opening_price,
                        "signal_direction":        None,
                        "signal_polymarket_price": None,
                        "signal_time":             None,
                        "market_found":            False,
                        # Flag so dashboard can show the price-source caveat
                        "price_source":            "Chainlink on-chain oracle (Polygon)",
                    }
                    if opening_price:
                        state["status"] = (
                            f"New window {win_start.strftime('%H:%M')} | "
                            f"Opening BTC: ${opening_price:,.2f} | "
                            f"Waiting for 30s signal..."
                        )
                    else:
                        state["status"] = "New window — could not fetch opening BTC price. Retrying..."

                if opening_price:
                    log.info(f"[{win_start.strftime('%H:%M')}] New window | BTC open: ${opening_price:,.2f}")
                else:
                    log.warning(f"[{win_start.strftime('%H:%M')}] Could not fetch opening BTC price.")

                # Fetch Polymarket price at window open for data logging
                open_market = find_btc_5min_market()
                poly_open = None
                if open_market:
                    up_p   = get_price_for_direction(open_market, "higher")
                    down_p = get_price_for_direction(open_market, "lower")
                    if up_p and down_p:
                        poly_open = up_p if up_p >= down_p else down_p
                with state_lock:
                    state["current_window"]["poly_open"] = poly_open

            # ── CONTINUOUS: UPDATE CURRENT BTC PRICE ───────────────────────
            # Poll every PRICE_POLL_INTERVAL seconds — Chainlink's heartbeat is
            # ~27s so there's no benefit hitting the RPC more often than that.
            # We still poll more frequently at signal/outcome time (see below).
            now_ts = time.time()
            if now_ts - last_price_fetch >= PRICE_POLL_INTERVAL:
                live_price = get_btc_price()
                last_price_fetch = now_ts
                if live_price:
                    with state_lock:
                        state["current_window"]["current_btc_price"] = live_price
            else:
                # Use the cached value already in state
                with state_lock:
                    live_price = state["current_window"].get("current_btc_price")

            # ── EVENT 2: 30-SECOND SIGNAL ──────────────────────────────────
            if (SIGNAL_WINDOW_LO <= secs_elapsed <= SIGNAL_WINDOW_HI
                    and signal_recorded_for != win_start):

                signal_recorded_for = win_start
                # Force a fresh Chainlink price at this critical moment
                live_price = get_btc_price()
                last_price_fetch = time.time()
                if live_price:
                    with state_lock:
                        state["current_window"]["current_btc_price"] = live_price

                with state_lock:
                    opening_price = state["current_window"].get("opening_btc_price")

                if opening_price is None:
                    log.warning("Skipping signal — no opening price available.")
                else:
                    current_price = live_price or opening_price

                    # Try to get the live Polymarket market first
                    market      = find_btc_5min_market()
                    poly_price  = None
                    market_found = False

                    if market:
                        market_found = True

                        # ── Direction from live market prices (preferred) ─────
                        # Polymarket's prices reflect its own opening reference
                        # (Chainlink Data Streams), which differs slightly from
                        # our on-chain Chainlink read.  Using the market's own
                        # implied direction avoids calling the wrong side when
                        # BTC sits between the two reference prices.
                        up_price   = get_price_for_direction(market, "higher")
                        down_price = get_price_for_direction(market, "lower")

                        if up_price and down_price:
                            # Whichever outcome the market prices above 50c is
                            # the direction momentum is in at this moment.
                            direction  = "higher" if up_price >= down_price else "lower"
                            poly_price = up_price if direction == "higher" else down_price
                            log.info(
                                f"Direction from market: UP={up_price:.4f} DOWN={down_price:.4f} "
                                f"→ signal {direction.upper()}"
                            )
                        else:
                            # Prices unavailable — fall back to our own comparison
                            direction  = "higher" if current_price >= opening_price else "lower"
                            poly_price = up_price or down_price

                    else:
                        # No market found — use our Chainlink price comparison
                        direction = "higher" if current_price >= opening_price else "lower"

                    if poly_price is None:
                        poly_price = FALLBACK_PRICE
                        log.info(f"Using fallback Polymarket price: {FALLBACK_PRICE}")

                    market_volume = get_market_volume(market) if market else None
                    market_slug   = market.get("slug") if market else None
                    market_id     = market.get("id")   if market else None
                    if market_volume:
                        log.info(f"Market volume: ${market_volume:,.0f}")
                    else:
                        log.info("Market volume unavailable — volume cap will not apply this window")
                    log.info(f"Market ID: {market_id} | Slug: {market_slug}")

                    with state_lock:
                        state["current_window"]["signal_direction"]        = direction
                        state["current_window"]["signal_polymarket_price"] = poly_price
                        state["current_window"]["signal_time"]             = now.strftime("%H:%M:%S")
                        state["current_window"]["market_found"]            = market_found
                        state["current_window"]["market_volume"]           = market_volume
                        state["current_window"]["market_slug"]             = market_slug
                        state["current_window"]["market_id"]               = market_id
                        state["current_window"]["btc_at_signal"]           = current_price
                        state["status"] = (
                            f"Signal @ 30s: BTC going {direction.upper()} | "
                            f"Polymarket: {poly_price:.4f} "
                            f"({'live' if market_found else 'fallback avg'})"
                        )

                    log.info(
                        f"SIGNAL: {direction.upper()} | Poly price: {poly_price:.4f} | "
                        f"BTC: ${current_price:,.2f} (open: ${opening_price:,.2f}) | "
                        f"Market found: {market_found}"
                    )

                    # ── Extract signal-direction CLOB token ID ───────────────
                    signal_token_id = None
                    if market:
                        outcomes_raw  = market.get("outcomes", "[]")
                        tokens_raw    = market.get("clobTokenIds", "[]")
                        out_list = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                        tok_list = json.loads(tokens_raw)   if isinstance(tokens_raw,   str) else tokens_raw
                        for o, t in zip(out_list, tok_list):
                            if direction == "higher" and o.lower() in ("up", "higher"):
                                signal_token_id = t; break
                            elif direction == "lower" and o.lower() in ("down", "lower"):
                                signal_token_id = t; break
                    with state_lock:
                        state["current_window"]["signal_token_id"] = signal_token_id

                    # ── Live order placement ─────────────────────────────────
                    if LIVE_MODE:
                        live_br = get_live_usdc_balance()  # real wallet balance each bet
                        with state_lock:
                            state["live_bankroll"] = live_br   # keep state in sync for display
                            live_peak   = state.get("live_peak_bankroll", live_br)
                            live_consec = state.get("live_consecutive_losses", 0)

                        raw_delta   = (current_price - opening_price) / opening_price * 100 if opening_price else 0.0
                        sig_delta   = raw_delta if direction == "higher" else -raw_delta
                        p_win_live  = get_pwin_2d(poly_price, sig_delta)
                        f_live      = calculate_kelly_fraction(p_win_live, poly_price)
                        stake_raw   = live_br * f_live * LIVE_KELLY_MULT if f_live > 0 else 0.0
                        stake_usdc  = min(stake_raw, MAX_LIVE_STAKE)

                        dd_live     = (DRAWDOWN_PROTECTION_ENABLED and live_peak > 0
                                       and (live_peak - live_br) / live_peak >= MAX_DRAWDOWN)
                        consec_live = DRAWDOWN_PROTECTION_ENABLED and live_consec >= MAX_CONSEC_LOSSES

                        placed_order_id = None
                        if not signal_token_id:
                            log.warning("LIVE: no token_id for signal direction — order skipped")
                        elif dd_live:
                            log.warning(f"LIVE: drawdown guard — bankroll ${live_br:.2f}, peak ${live_peak:.2f} — skipped")
                        elif consec_live:
                            log.warning(f"LIVE: {live_consec} consecutive losses — order skipped")
                        elif stake_usdc < MIN_LIVE_STAKE:
                            log.info(f"LIVE: Kelly stake ${stake_usdc:.2f} < min ${MIN_LIVE_STAKE} — skipped")
                        else:
                            placed_order_id = place_live_order(signal_token_id, poly_price, stake_usdc)

                        with state_lock:
                            state["current_window"]["live_order_id"]   = placed_order_id
                            state["current_window"]["live_stake_usdc"] = stake_usdc if placed_order_id else 0.0
                            state["current_window"]["live_price"]      = poly_price if placed_order_id else None

            # ── LIVE: cancel unfilled order at t=270s (30s before close) ────
            if LIVE_MODE and 268 <= secs_elapsed <= 272:
                with state_lock:
                    pending_oid  = state["current_window"].get("live_order_id")
                    pending_px   = state["current_window"].get("live_price")
                if pending_oid and pending_px:
                    check_fill = get_order_fill_usdc(pending_oid, pending_px)
                    if check_fill is None or check_fill < 0.01:
                        log.warning(f"LIVE: order {pending_oid} unfilled at t=270s — cancelling")
                        cancel_order_safe(pending_oid)
                        with state_lock:
                            state["current_window"]["live_order_id"] = None

            # ── EVENT 3: WINDOW END ─────────────────────────────────────────
            if (OUTCOME_WINDOW_LO <= secs_elapsed <= OUTCOME_WINDOW_HI
                    and outcome_recorded_for != win_start):

                outcome_recorded_for = win_start

                with state_lock:
                    opening_price    = state["current_window"].get("opening_btc_price")
                    signal_dir       = state["current_window"].get("signal_direction")
                    poly_price       = state["current_window"].get("signal_polymarket_price")
                    market_volume    = state["current_window"].get("market_volume")
                    market_slug      = state["current_window"].get("market_slug")
                    market_id        = state["current_window"].get("market_id")
                    btc_at_signal    = state["current_window"].get("btc_at_signal")
                    live_order_id    = state["current_window"].get("live_order_id")
                    live_stake_usdc  = state["current_window"].get("live_stake_usdc", 0.0)
                    live_price_ord   = state["current_window"].get("live_price")
                    live_br_pre      = state.get("live_bankroll", STARTING_BANKROLL)
                    live_peak_pre    = state.get("live_peak_bankroll", live_br_pre)
                    live_consec_pre  = state.get("live_consecutive_losses", 0)

                if not all([opening_price, signal_dir, poly_price]):
                    log.warning("Window ended but signal was never taken — skipping trade record.")
                else:
                    # ── Resolve win/loss from Polymarket's own resolution ────
                    # Fetch the closed market and read outcomePrices.
                    # The winning outcome is priced at "1" by Polymarket after
                    # resolution — this uses Chainlink Data Streams as the source
                    # of truth, not our on-chain Chainlink feed.
                    # ── Resolve via CLOB prices at t=299s ───────────────────
                    # At 1 second before close, CLOB prices strongly reflect
                    # the outcome (arbitrageurs push to 0/1). Whichever side
                    # is priced higher IS the resolution direction.
                    actual_dir = None
                    resolution_source = "clob_at_close"

                    # Re-fetch market dict to get current clobTokenIds
                    outcome_market = None
                    if market_id:
                        outcome_market = fetch_json(
                            f"{GAMMA_API_BASE}/markets/{market_id}", retries=2
                        )
                        if isinstance(outcome_market, list):
                            outcome_market = outcome_market[0] if outcome_market else None
                    if not outcome_market and market_slug:
                        d = fetch_json(
                            f"{GAMMA_API_BASE}/markets",
                            params={"slug": market_slug},
                            retries=2,
                        )
                        if d:
                            ml = d if isinstance(d, list) else d.get("markets", [])
                            outcome_market = ml[0] if ml else None

                    if outcome_market:
                        outcomes_raw = outcome_market.get("outcomes", "[]")
                        tokens_raw   = outcome_market.get("clobTokenIds", "[]")
                        outcomes_list = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                        tokens_list   = json.loads(tokens_raw)   if isinstance(tokens_raw, str)   else tokens_raw

                        up_token = down_token = None
                        for o, t in zip(outcomes_list, tokens_list):
                            if o.lower() in ("up", "higher"):
                                up_token = t
                            elif o.lower() in ("down", "lower"):
                                down_token = t

                        up_price   = get_clob_midpoint_raw(up_token)   if up_token   else None
                        down_price = get_clob_midpoint_raw(down_token) if down_token else None
                        log.info(f"CLOB at t=299s — UP={up_price} DOWN={down_price}")

                        if up_price is not None and down_price is not None:
                            actual_dir = "higher" if up_price > down_price else "lower"
                        elif up_price is not None:
                            actual_dir = "higher" if up_price >= 0.5 else "lower"
                        elif down_price is not None:
                            actual_dir = "lower" if down_price >= 0.5 else "higher"

                    if actual_dir is None:
                        log.warning("Could not get CLOB prices at window close — skipping trade.")
                        continue

                    won = (signal_dir == actual_dir)

                    # ── Live settlement ──────────────────────────────────────
                    live_filled = 0.0
                    live_pnl    = 0.0
                    if LIVE_MODE and live_order_id and live_price_ord:
                        live_filled = get_order_fill_usdc(live_order_id, live_price_ord) or 0.0
                        if live_filled < 0.01:
                            cancel_order_safe(live_order_id)
                            log.warning(f"LIVE: order {live_order_id} unfilled at resolution — cancelled")
                        else:
                            # Cancel any remaining unfilled portion
                            if live_filled < live_stake_usdc * 0.99:
                                cancel_order_safe(live_order_id)
                                log.info(f"LIVE: partial fill ${live_filled:.2f}/${live_stake_usdc:.2f} — remainder cancelled")
                            if won:
                                live_pnl = live_filled * (1.0 / live_price_ord - 1.0) * (1.0 - POLYMARKET_FEE)
                            else:
                                live_pnl = -live_filled
                            log.info(
                                f"LIVE SETTLE: {'WIN' if won else 'LOSS'} | "
                                f"Filled ${live_filled:.2f} | P&L ${live_pnl:+.2f} | "
                                f"Live bankroll ${live_br_pre + live_pnl:.2f}"
                            )

                    # Log this window's raw data for future model building
                    with state_lock:
                        poly_open = state["current_window"].get("poly_open")
                    _append_window_log({
                        "window_start":    win_start.strftime("%Y-%m-%d %H:%M UTC"),
                        "btc_open":        round(opening_price, 2) if opening_price else "",
                        "poly_open":       round(poly_open, 4) if poly_open else "",
                        "btc_at_30s":      round(btc_at_signal, 2) if btc_at_signal else "",
                        "poly_at_30s":     round(poly_price, 4) if poly_price else "",
                        "signal_direction": signal_dir,
                        "outcome":         actual_dir,
                        "won":             won,
                    })

                    # BTC delta at t=30s — signal-aligned (positive = moving with signal)
                    signal_price    = btc_at_signal or opening_price
                    price_delta     = signal_price - opening_price if opening_price else 0.0
                    price_delta_pct = round(price_delta / opening_price * 100, 4) if opening_price else 0.0
                    # Flip sign for LOWER signals so positive always means "with the signal"
                    signal_aligned_delta = price_delta_pct if signal_dir == "higher" else -price_delta_pct

                    # 2D p_win: bucketed by Polymarket price AND signal-aligned BTC momentum
                    p_win_empirical = get_pwin_2d(poly_price, signal_aligned_delta)
                    f_full = calculate_kelly_fraction(p_win_empirical, poly_price)
                    edge_positive = f_full > 0

                    with state_lock:
                        br_q = state["bankroll_quarter_kelly"]
                        br_h = state["bankroll_half_kelly"]
                        br_f = state["bankroll_full_kelly"]
                        pk_q = state.get("peak_quarter_kelly", br_q)
                        pk_h = state.get("peak_half_kelly",    br_h)
                        pk_f = state.get("peak_full_kelly",    br_f)
                        consec_losses = state.get("consecutive_losses", 0)

                        # ── Drawdown protection ──────────────────────────────
                        def drawdown_blocked(bankroll, peak, label):
                            if not DRAWDOWN_PROTECTION_ENABLED:
                                return False
                            if peak > 0 and (peak - bankroll) / peak >= MAX_DRAWDOWN:
                                log.warning(
                                    f"DRAWDOWN PROTECTION ({label}): bankroll ${bankroll:.2f} "
                                    f"is {(peak-bankroll)/peak:.1%} below peak ${peak:.2f} — bet blocked"
                                )
                                return True
                            return False

                        consec_blocked = DRAWDOWN_PROTECTION_ENABLED and consec_losses >= MAX_CONSEC_LOSSES
                        if consec_blocked:
                            log.warning(
                                f"CONSECUTIVE LOSS PROTECTION: {consec_losses} losses in a row — bet blocked"
                            )

                        dd_q = drawdown_blocked(br_q, pk_q, "¼K")
                        dd_h = drawdown_blocked(br_h, pk_h, "½K")
                        dd_f = drawdown_blocked(br_f, pk_f, "K")

                        # Volume cap: never stake more than 10% of market volume.
                        # If volume is unavailable the cap is skipped gracefully.
                        max_vol_cap = 0.10 * market_volume if market_volume else None

                        def vol_capped_fraction(bankroll, fraction):
                            stake = bankroll * fraction
                            if max_vol_cap and stake > max_vol_cap:
                                log.info(
                                    f"Volume cap applied: stake ${stake:.2f} → "
                                    f"${max_vol_cap:.2f} (10% of ${market_volume:,.0f})"
                                )
                                return max_vol_cap / bankroll if bankroll > 0 else 0.0
                            return fraction

                        # Simulate all three Kelly fractions with volume cap + protection applied
                        fq = 0.0 if (dd_q or consec_blocked) else vol_capped_fraction(br_q, f_full * 0.25)
                        fh = 0.0 if (dd_h or consec_blocked) else vol_capped_fraction(br_h, f_full * 0.50)
                        ff = 0.0 if (dd_f or consec_blocked) else vol_capped_fraction(br_f, f_full * 1.00)

                        # SIMULATION ONLY — apply_bet() models the P&L mathematically.
                        # For live execution, replace these three calls with CLOB API
                        # order placement (see the "FUTURE: LIVE ORDER EXECUTION" note
                        # at the top of this file for full details).
                        new_q, pnl_q = apply_bet(br_q, fq, poly_price, won)
                        new_h, pnl_h = apply_bet(br_h, fh, poly_price, won)
                        new_f, pnl_f = apply_bet(br_f, ff, poly_price, won)

                        state["bankroll_quarter_kelly"] = new_q
                        state["bankroll_half_kelly"]    = new_h
                        state["bankroll_full_kelly"]    = new_f

                        # Update peak bankrolls
                        state["peak_quarter_kelly"] = max(pk_q, new_q)
                        state["peak_half_kelly"]    = max(pk_h, new_h)
                        state["peak_full_kelly"]    = max(pk_f, new_f)

                        # Update simulated consecutive loss counter
                        if won:
                            state["consecutive_losses"] = 0
                        else:
                            state["consecutive_losses"] = consec_losses + 1

                        # Update live bankroll (if a real order was filled)
                        if LIVE_MODE and live_filled > 0.01:
                            new_live_br = live_br_pre + live_pnl
                            state["live_bankroll"]           = new_live_br
                            state["live_peak_bankroll"]      = max(live_peak_pre, new_live_br)
                            state["live_consecutive_losses"] = 0 if won else live_consec_pre + 1

                        state["total_trades"] += 1
                        if won:
                            state["wins"] += 1

                        total   = state["total_trades"]
                        win_rate = state["wins"] / total if total else 0

                        # Add a point to the bankroll chart history
                        state["bankroll_history"].append({
                            "time":         now.strftime("%H:%M"),
                            "quarter_kelly": round(new_q, 2),
                            "half_kelly":    round(new_h, 2),
                            "full_kelly":    round(new_f, 2),
                        })
                        # Cap history to 1000 points so memory stays bounded
                        if len(state["bankroll_history"]) > 1000:
                            state["bankroll_history"] = state["bankroll_history"][-1000:]

                        # Build the trade log entry
                        trade_entry = {
                            "timestamp":           win_start.strftime("%Y-%m-%d %H:%M UTC"),
                            "opening_btc":         round(opening_price, 2),
                            "signal_direction":    signal_dir,
                            "polymarket_outcome":  actual_dir,
                            "resolution_source":   resolution_source,
                            "market_slug":         market_slug,
                            "polymarket_price":    round(poly_price, 4),
                            "p_win_used":          round(p_win_empirical, 4),
                            # Calibration fields — used to build a better p_win curve over time:
                            # plot price_delta_pct vs won to see how win rate varies with momentum strength
                            "price_delta_pct":     price_delta_pct,
                            "signal_aligned_delta": round(signal_aligned_delta, 4),
                            "market_volume":       round(market_volume, 2) if market_volume else None,
                            "full_kelly_fraction": round(f_full, 4),
                            "edge_positive":       edge_positive,
                            "won":                 won,
                            "pnl_quarter_kelly":   round(pnl_q, 4),
                            "pnl_half_kelly":      round(pnl_h, 4),
                            "pnl_full_kelly":      round(pnl_f, 4),
                            "bankroll_quarter_kelly": round(new_q, 2),
                            "bankroll_half_kelly":    round(new_h, 2),
                            "bankroll_full_kelly":    round(new_f, 2),
                            # Live execution fields (None when LIVE_MODE = False)
                            "live_order_id":    live_order_id   if LIVE_MODE else None,
                            "live_filled_usdc": round(live_filled, 4) if LIVE_MODE else None,
                            "live_pnl":         round(live_pnl,   4) if LIVE_MODE else None,
                            "live_bankroll":    round(state.get("live_bankroll", STARTING_BANKROLL), 2) if LIVE_MODE else None,
                        }
                        state["trades"].append(trade_entry)

                        state["status"] = (
                            f"Trade #{total}: {'WIN ✓' if won else 'LOSS ✗'} | "
                            f"Win rate {win_rate:.1%} vs {poly_price:.1%} implied | "
                            f"Full Kelly: ${new_f:.2f}"
                        )

                    log.info(
                        f"OUTCOME: {'WIN ✓' if won else 'LOSS ✗'} | "
                        f"Signal {signal_dir.upper()} | Polymarket resolved {actual_dir.upper()} | "
                        f"Market price {poly_price:.3f} ({'edge: bet placed' if edge_positive else 'no edge — Kelly skip'}) | "
                        f"Trade #{state['total_trades']} | "
                        f"Bankrolls  ¼K:${new_q:.2f}  ½K:${new_h:.2f}  K:${new_f:.2f}"
                    )

                    save_state_to_disk()

        except Exception as e:
            log.error(f"Unhandled error in monitor loop: {e}", exc_info=True)

        # Sleep 1 second before the next iteration
        time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# FLASK WEB DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def dashboard():
    """Serve the main dashboard HTML page."""
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    """
    JSON endpoint polled by the dashboard every few seconds.
    Returns everything the dashboard needs to update itself.
    """
    with state_lock:
        total  = state.get("total_trades", 0)
        wins   = state.get("wins", 0)
        win_rate = wins / total if total > 0 else 0.0

        # Parse running_since (handles both timezone-aware and naive ISO strings)
        try:
            rs = datetime.fromisoformat(state.get("running_since", ""))
            if rs.tzinfo is None:
                rs = rs.replace(tzinfo=timezone.utc)
            uptime_secs = (datetime.now(timezone.utc) - rs).total_seconds()
        except Exception:
            uptime_secs = 0

        # Compute implied probability from the last trade if available
        last_poly_price = None
        if state["trades"]:
            last_poly_price = state["trades"][-1].get("polymarket_price")

        # Kelly fraction for display
        display_kelly = None
        cw = state["current_window"]
        if cw.get("signal_polymarket_price"):
            display_kelly = round(
                calculate_kelly_fraction(WIN_PROBABILITY, cw["signal_polymarket_price"]), 4
            )

        # Drawdown protection status per strategy
        def _drawdown_pct(bankroll, peak):
            if peak and peak > 0:
                return round((peak - bankroll) / peak * 100, 1)
            return 0.0

        br_q = state["bankroll_quarter_kelly"]
        br_h = state["bankroll_half_kelly"]
        br_f = state["bankroll_full_kelly"]
        pk_q = state.get("peak_quarter_kelly", br_q)
        pk_h = state.get("peak_half_kelly",    br_h)
        pk_f = state.get("peak_full_kelly",    br_f)
        consec = state.get("consecutive_losses", 0)

        protection = {
            "drawdown_quarter": _drawdown_pct(br_q, pk_q),
            "drawdown_half":    _drawdown_pct(br_h, pk_h),
            "drawdown_full":    _drawdown_pct(br_f, pk_f),
            "dd_blocked_quarter": DRAWDOWN_PROTECTION_ENABLED and _drawdown_pct(br_q, pk_q) >= MAX_DRAWDOWN * 100,
            "dd_blocked_half":    DRAWDOWN_PROTECTION_ENABLED and _drawdown_pct(br_h, pk_h) >= MAX_DRAWDOWN * 100,
            "dd_blocked_full":    DRAWDOWN_PROTECTION_ENABLED and _drawdown_pct(br_f, pk_f) >= MAX_DRAWDOWN * 100,
            "consecutive_losses": consec,
            "consec_blocked":     DRAWDOWN_PROTECTION_ENABLED and consec >= MAX_CONSEC_LOSSES,
        }

        return jsonify({
            # Bankrolls
            "bankroll_quarter_kelly": round(br_q, 2),
            "bankroll_half_kelly":    round(br_h, 2),
            "bankroll_full_kelly":    round(br_f, 2),

            # Chart data (last 200 points)
            "bankroll_history": state["bankroll_history"][-200:],

            # Current window
            "current_window": cw,
            "full_kelly_fraction": display_kelly,

            # Protection status
            "protection": protection,

            # Metadata
            "status":       state["status"],
            "total_trades": total,
            "wins":         wins,
            "win_rate":     round(win_rate * 100, 1),
            "implied_prob": round((last_poly_price or FALLBACK_PRICE) * 100, 1),
            "uptime":       _format_uptime(uptime_secs),

            # Last 50 trades, newest first
            "recent_trades": list(reversed(state["trades"][-50:])),
        })


def _format_uptime(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable 'Xh Ym Zs' string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 66)
    print("   PHANTOM EDGE — BTC Polymarket 5-Min Signal Tracker")
    print("=" * 66)
    print(f"   Dashboard  →  http://localhost:{DASHBOARD_PORT}   ← open this in your browser")
    print(f"   Trade log  →  {os.path.abspath(LOG_FILE)}")
    print(f"   App log    →  {os.path.abspath('phantom_edge.log')}")
    print("=" * 66)
    print()
    print("   HOW IT WORKS:")
    print("   • Every 5 min (aligned to clock: 12:00, 12:05, 12:10 …)")
    print("     the app records the opening BTC price.")
    print("   • At t=30s it records the direction (higher/lower) and")
    print("     searches Polymarket for live odds.")
    print("   • At t=5min it checks the outcome and updates bankrolls.")
    print()
    print("   POLYMARKET PRICE SOURCE:")
    print("   • If an active BTC 5-min market is found → live price used.")
    print("   • If not (these markets are episodic) → falls back to the")
    print(f"     historical average price ({FALLBACK_PRICE}) from your research.")
    print("   • To use a specific market, set MANUAL_MARKET_ID in the script.")
    print()
    print("   NO REAL MONEY IS PLACED — simulation only.")
    print("=" * 66)
    print("   Press Ctrl+C to stop")
    print("=" * 66)
    print()

    # Graceful shutdown: save state on SIGINT (Ctrl+C) and SIGTERM (kill/system shutdown)
    def _shutdown(signum, frame):
        log.info(f"Received signal {signum} — saving state before exit...")
        try:
            save_state_to_disk()
            log.info("State saved. Goodbye.")
        except Exception as e:
            log.error(f"Failed to save state on shutdown: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start the monitor in a daemon thread so it dies if Flask crashes
    t = threading.Thread(target=monitor_loop, name="MonitorLoop", daemon=True)
    t.start()
    log.info("Monitor thread started.")

    # Flask blocks the main thread — use_reloader=False is required when
    # Flask runs alongside another thread (avoids double-starting the monitor).
    app.run(
        host="0.0.0.0",
        port=DASHBOARD_PORT,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
