import json
import os
import time
import requests

# Tier 1: BTC ETH SOL XRP
# Tier 2: BNB LINK AVAX ADA DOGE ARB OP APT TRX
# New:    SUI TON WIF JUP INJ TIA
PAIRS = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "LINK", "AVAX", "ADA", "DOGE", "ARB",
    "OP", "APT", "TRX", "SUI", "TON", "WIF", "JUP", "INJ", "TIA",
]

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h"]

FUTURES_INTERVAL = {
    "1m": "Min1",
    "5m": "Min5",
    "15m": "Min15",
    "30m": "Min30",
    "1h": "Min60",
    "4h": "Hour4",
}

LIMIT = 500
FUTURES_BASE = "https://contract.mexc.com/api/v1/contract"
OUTPUT_DIR = "data/candles"


def fetch_futures(symbol: str, tf: str) -> list[dict]:
    url = f"{FUTURES_BASE}/kline/{symbol}_USDT"
    r = requests.get(url, params={"interval": FUTURES_INTERVAL[tf], "limit": LIMIT}, timeout=15)
    r.raise_for_status()
    body = r.json()
    if not body.get("success"):
        raise ValueError(body)
    d = body["data"]
    return [
        {
            "time": d["time"][i],
            "open": float(d["open"][i]),
            "high": float(d["high"][i]),
            "low": float(d["low"][i]),
            "close": float(d["close"][i]),
            "volume": float(d["vol"][i]),
        }
        for i in range(len(d["time"]))
    ]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []

    for pair in PAIRS:
        for tf in TIMEFRAMES:
            path = f"{OUTPUT_DIR}/{pair}USDT_{tf}.json"
            try:
                candles = fetch_futures(pair, tf)
                with open(path, "w") as f:
                    json.dump(candles, f, separators=(",", ":"))
                print(f"OK  {pair}USDT_{tf}: {len(candles)} bars")
            except Exception as exc:
                print(f"ERR {pair}USDT_{tf}: {exc}")
                errors.append(f"{pair}USDT_{tf}")
            time.sleep(0.15)

    if errors:
        print(f"\nFailed ({len(errors)}): {', '.join(errors)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
