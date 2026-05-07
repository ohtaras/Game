import json
import os
import time
import requests

# Primary: full timeframe stack — BTC/ETH are the market makers
# Secondary: skip 1m and 30m — enough resolution without noise
PAIRS = {
    "primary":   ["BTC", "ETH"],
    "secondary": ["SOL", "BNB", "LINK", "DOGE", "SUI"],
}

PRIMARY_TF   = ["1m", "5m", "15m", "30m", "1h", "4h"]
SECONDARY_TF = ["5m", "15m", "1h", "4h"]

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
    for attempt in range(3):
        try:
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
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []

    schedule = (
        [(p, tf) for p in PAIRS["primary"]   for tf in PRIMARY_TF] +
        [(p, tf) for p in PAIRS["secondary"] for tf in SECONDARY_TF]
    )

    for pair, tf in schedule:
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
