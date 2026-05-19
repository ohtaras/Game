import json
import os
from datetime import datetime, timezone, timedelta

GR_TZ = timezone(timedelta(hours=3))


def now_gr() -> str:
    return datetime.now(GR_TZ).strftime("%H:%M (Ελλάδος)")

DATA_DIR = "data/candles"
OUTPUT_DIR = "data/signals"

PAIRS_PRIMARY = ["BTC", "ETH"]
PAIRS_SECONDARY = ["SOL", "BNB", "LINK", "DOGE", "SUI"]

FVG_MIN_PCT = 0.001       # minimum FVG size (0.1% of price)
OB_IMPULSE_PCT = 0.005    # minimum impulse after OB (0.5%)
LIQ_TOLERANCE = 0.002     # equal highs/lows tolerance (0.2%)
MIN_RR = 2.0


# ── Data ────────────────────────────────────────────────────────────────────

def load(pair: str, tf: str) -> list[dict]:
    path = f"{DATA_DIR}/{pair}USDT_{tf}.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ── Market Structure ─────────────────────────────────────────────────────────

def bias(candles: list[dict]) -> str:
    """Compare last 10 vs prior 10 candles for HH/HL or LL/LH."""
    if len(candles) < 20:
        return "neutral"
    prev = candles[-20:-10]
    last = candles[-10:]
    ph = max(c["high"] for c in prev)
    pl = min(c["low"]  for c in prev)
    lh = max(c["high"] for c in last)
    ll = min(c["low"]  for c in last)
    if lh > ph and ll > pl:
        return "bullish"
    if lh < ph and ll < pl:
        return "bearish"
    return "neutral"


# ── Order Blocks ─────────────────────────────────────────────────────────────

def order_blocks(candles: list[dict], direction: str) -> list[dict]:
    """Last opposite-colour candle before a strong impulse move."""
    result = []
    for i in range(2, min(len(candles) - 1, 80)):
        c = candles[-(i + 1)]
        nxt = candles[-i]
        if direction == "bullish":
            if c["close"] < c["open"]:                          # bearish candle
                move = (nxt["close"] - c["low"]) / c["low"]
                if move >= OB_IMPULSE_PCT:
                    result.append({"top": c["open"], "bottom": c["low"],
                                   "mid": (c["open"] + c["low"]) / 2})
        else:
            if c["close"] > c["open"]:                          # bullish candle
                move = (c["high"] - nxt["close"]) / c["high"]
                if move >= OB_IMPULSE_PCT:
                    result.append({"top": c["high"], "bottom": c["close"],
                                   "mid": (c["high"] + c["close"]) / 2})
        if len(result) == 3:
            break
    return result


# ── Fair Value Gaps ──────────────────────────────────────────────────────────

def fvgs(candles: list[dict], direction: str) -> list[dict]:
    """3-candle imbalances."""
    result = []
    for i in range(2, len(candles)):
        c1, _, c3 = candles[i - 2], candles[i - 1], candles[i]
        if direction == "bullish" and c3["low"] > c1["high"]:
            size = (c3["low"] - c1["high"]) / c1["high"]
            if size >= FVG_MIN_PCT:
                result.append({"top": c3["low"], "bottom": c1["high"],
                                "mid": (c3["low"] + c1["high"]) / 2})
        elif direction == "bearish" and c3["high"] < c1["low"]:
            size = (c1["low"] - c3["high"]) / c1["low"]
            if size >= FVG_MIN_PCT:
                result.append({"top": c1["low"], "bottom": c3["high"],
                                "mid": (c1["low"] + c3["high"]) / 2})
    return result[-5:]


# ── Liquidity Levels ─────────────────────────────────────────────────────────

def liquidity(candles: list[dict]) -> dict:
    """Cluster equal highs (resistance) and equal lows (support)."""
    recent = candles[-120:]
    all_h = [c["high"] for c in recent]
    all_l = [c["low"]  for c in recent]

    def clusters(levels):
        seen = []
        for lvl in sorted(levels, reverse=True):
            group = [x for x in levels if abs(x - lvl) / lvl < LIQ_TOLERANCE]
            if len(group) >= 2:
                avg = sum(group) / len(group)
                if not any(abs(s - avg) / avg < LIQ_TOLERANCE for s in seen):
                    seen.append(round(avg, 8))
        return seen

    return {
        "resistance": sorted(clusters(all_h), reverse=True)[:4],
        "support":    sorted(clusters(all_l))[:4],
    }


# ── Signal Builder ───────────────────────────────────────────────────────────

def watch(pair: str) -> dict | None:
    """Return watch info for pairs with aligned bias, even if not in zone yet."""
    c4h  = load(pair, "4h")
    c1h  = load(pair, "1h")
    c15m = load(pair, "15m")
    c5m  = load(pair, "5m")
    if not all([c4h, c1h, c15m, c5m]):
        return None
    b4h = bias(c4h)
    b1h = bias(c1h)
    if b4h != b1h or b4h == "neutral":
        return None
    direction = "LONG" if b4h == "bullish" else "SHORT"
    price = c5m[-1]["close"]
    obs  = order_blocks(c15m, b4h)
    gaps = fvgs(c15m, b4h)
    liq  = liquidity(c1h)
    # Pick nearest zone regardless of whether price is in it
    candidates = []
    if direction == "LONG":
        for ob in obs:
            if ob["top"] <= price:  # zone below price
                candidates.append(ob)
        for g in gaps:
            if g["top"] <= price:
                candidates.append(g)
        zone = max(candidates, key=lambda z: z["top"]) if candidates else (obs[0] if obs else None)
        if not zone:
            return None
        entry = zone["mid"]
        sl    = zone["bottom"] * 0.998
        tgts  = [t for t in liq["resistance"] if t > entry * 1.005]
        if not tgts:
            return None
        tp1 = tgts[-1]
        tp2 = tgts[0] if len(tgts) > 1 else round(tp1 * 1.01, 8)
    else:
        for ob in obs:
            if ob["bottom"] >= price:
                candidates.append(ob)
        for g in gaps:
            if g["bottom"] >= price:
                candidates.append(g)
        zone = min(candidates, key=lambda z: z["bottom"]) if candidates else (obs[0] if obs else None)
        if not zone:
            return None
        entry = zone["mid"]
        sl    = zone["top"] * 1.002
        tgts  = [t for t in liq["support"] if t < entry * 0.995]
        if not tgts:
            return None
        tp1 = tgts[0]
        tp2 = tgts[-1] if len(tgts) > 1 else round(tp1 * 0.99, 8)
    risk   = abs(entry - sl)
    reward = abs(tp1 - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0
    dist   = round(abs(price - entry), 4)
    dist_pct = round(abs(price - entry) / price * 100, 2)
    sl_dist_pct = abs(price - sl) / price * 100
    if sl_dist_pct < 0.5:           # SL too close to current price — wick risk
        return None
    return {
        "pair":      f"{pair}USDT",
        "direction": direction,
        "price":     round(price, 6),
        "entry":     round(entry, 6),
        "sl":        round(sl, 6),
        "tp1":       round(tp1, 6),
        "tp2":       round(tp2, 6),
        "rr":        rr,
        "dist":      dist,
        "dist_pct":  dist_pct,
        "bias_4h":   b4h,
        "bias_1h":   b1h,
        "time":      datetime.now(GR_TZ).strftime("%Y-%m-%d %H:%M (Ελλάδος)"),
    }


def analyze(pair: str) -> dict | None:
    c4h  = load(pair, "4h")
    c1h  = load(pair, "1h")
    c15m = load(pair, "15m")
    c5m  = load(pair, "5m")
    if not all([c4h, c1h, c15m, c5m]):
        return None

    b4h = bias(c4h)
    b1h = bias(c1h)
    if b4h != b1h or b4h == "neutral":
        return None                         # timeframes disagree → no trade

    direction = "LONG" if b4h == "bullish" else "SHORT"
    price = c5m[-1]["close"]

    obs  = order_blocks(c15m, b4h)
    gaps = fvgs(c15m, b4h)
    liq  = liquidity(c1h)

    # Find which zone price is currently trading into
    zone = None
    tags = []
    if direction == "LONG":
        for ob in obs:
            if ob["bottom"] <= price <= ob["top"] * 1.005:
                zone = ob; tags.append("OB"); break
        for g in reversed(gaps):
            if g["bottom"] <= price <= g["top"] * 1.005:
                tags.append("FVG")
                if not zone:
                    zone = g
                break
        if not zone:
            return None
        entry = zone["mid"]
        sl    = zone["bottom"] * 0.998
        tgts  = [t for t in liq["resistance"] if t > entry * 1.005]
        if not tgts:
            return None
        tp1 = tgts[-1]
        tp2 = tgts[0] if len(tgts) > 1 else round(tp1 * 1.01, 8)
    else:
        for ob in obs:
            if ob["bottom"] * 0.995 <= price <= ob["top"]:
                zone = ob; tags.append("OB"); break
        for g in reversed(gaps):
            if g["bottom"] * 0.995 <= price <= g["top"]:
                tags.append("FVG")
                if not zone:
                    zone = g
                break
        if not zone:
            return None
        entry = zone["mid"]
        sl    = zone["top"] * 1.002
        tgts  = [t for t in liq["support"] if t < entry * 0.995]
        if not tgts:
            return None
        tp1 = tgts[0]
        tp2 = tgts[-1] if len(tgts) > 1 else round(tp1 * 0.99, 8)

    risk   = abs(entry - sl)
    reward = abs(tp1 - entry)
    if risk == 0 or reward / risk < MIN_RR:
        return None

    return {
        "pair":      f"{pair}USDT",
        "direction": direction,
        "entry":     round(entry, 6),
        "sl":        round(sl, 6),
        "tp1":       round(tp1, 6),
        "tp2":       round(tp2, 6),
        "rr":        round(reward / risk, 2),
        "setup":     " + ".join(tags),
        "bias_4h":   b4h,
        "bias_1h":   b1h,
        "price":     round(price, 6),
        "time":      datetime.now(GR_TZ).strftime("%Y-%m-%d %H:%M (Ελλάδος)"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    signals = []
    watches = []

    for pair in PAIRS_PRIMARY + PAIRS_SECONDARY:
        sig = analyze(pair)
        if sig:
            signals.append(sig)
            print(f"SIGNAL  {sig['pair']:12} {sig['direction']:5} "
                  f"entry={sig['entry']}  sl={sig['sl']}  "
                  f"tp1={sig['tp1']}  RR={sig['rr']}")
        else:
            w = watch(pair)
            if w:
                watches.append(w)
                print(f"WATCH   {w['pair']:12} {w['direction']:5} "
                      f"entry={w['entry']}  dist={w['dist_pct']}%  RR={w['rr']}")
            else:
                print(f"NO SETUP {pair}USDT")

    # Radar: pairs with 4h trend but 1h not yet confirmed
    radar = []
    for pair in PAIRS_PRIMARY + PAIRS_SECONDARY:
        c4h = load(pair, "4h"); c1h = load(pair, "1h"); c5m = load(pair, "5m")
        if not all([c4h, c1h, c5m]):
            continue
        b4h = bias(c4h); b1h = bias(c1h)
        if b4h == "neutral":
            continue
        if b4h == b1h:
            continue  # already in signals or watches
        price = c5m[-1]["close"]
        radar.append({
            "pair":    f"{pair}USDT",
            "price":   round(price, 6),
            "bias_4h": b4h,
            "bias_1h": b1h,
        })

    signals.sort(key=lambda x: x["rr"], reverse=True)
    watches.sort(key=lambda x: x["dist_pct"])

    with open(f"{OUTPUT_DIR}/signals.json", "w") as f:
        json.dump({
            "updated":     datetime.now(GR_TZ).strftime("%Y-%m-%d %H:%M (Ελλάδος)"),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "count":       len(signals),
            "signals":     signals,
            "watches":     watches,
            "radar":       radar,
        }, f, indent=2)

    print(f"\n→ {len(signals)} signal(s), {len(watches)} watch(es), {len(radar)} radar  |  {now_gr()}")


if __name__ == "__main__":
    main()
