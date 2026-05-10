#!/usr/bin/env python3
"""
Fetch KINO (game 1100) draw results from OPAP API.
Usage: python fetch_kino.py [YYYY-MM-DD]
Default date: today
"""

import json
import sys
import os
from datetime import date

import requests

GAME_ID = 1100
BASE_URL = "https://api.opap.gr"
OUTPUT_DIR = "kino/data"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def fetch_draws(draw_date: str) -> dict:
    url = f"{BASE_URL}/draws/v3.0/{GAME_ID}/draw-date/{draw_date}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_draws(raw: dict) -> list:
    draws = []
    for item in raw.get("content", []):
        draw = {
            "draw_id": item.get("drawId"),
            "draw_time": item.get("drawTime"),
            "status": item.get("drawBreakdown", {}).get("wagerStatistics", {}).get("status"),
            "winning_numbers": item.get("winningNumbers", {}).get("list", []),
            "bonus": item.get("winningNumbers", {}).get("bonus", []),
        }
        draws.append(draw)
    return draws


def main():
    draw_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    print(f"Fetching KINO draws for {draw_date} ...")
    raw = fetch_draws(draw_date)
    draws = parse_draws(raw)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/kino_{draw_date.replace('-', '')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"date": draw_date, "total_draws": len(draws), "draws": draws}, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(draws)} draws to {filename}")


if __name__ == "__main__":
    main()
