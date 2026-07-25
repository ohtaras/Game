#!/usr/bin/env python3
"""Fetch latest KINO draws from OPAP and merge into data/raw/kino_raw_YYYY_MM.json"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import requests

GAME_ID = 1100
OPAP_BASE = "https://api.opap.gr"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Origin": "https://www.opap.gr",
    "Referer": "https://www.opap.gr/",
}

ATHENS = timezone(timedelta(hours=3))


def fetch_draws_for_date(date_str):
    url = f"{OPAP_BASE}/draws/v3.0/{GAME_ID}/draw-date/{date_str}/{date_str}?page=0&size=300"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    draws = []
    for item in data.get("content", []):
        nums = item.get("winningNumbers", {}).get("list", [])
        bonus = item.get("winningNumbers", {}).get("bonus", [])
        draws.append({
            "id": item["drawId"],
            "n": sorted(nums),
            "b": bonus[0] if bonus else None,
        })
    return sorted(draws, key=lambda d: d["id"])


def main():
    now_athens = datetime.now(ATHENS)
    date_str = now_athens.strftime("%Y-%m-%d")
    month_str = now_athens.strftime("%Y-%m")
    month_file = f"data/raw/kino_raw_{now_athens.strftime('%Y_%m')}.json"

    print(f"Fetching KINO for {date_str} ...")
    new_draws = fetch_draws_for_date(date_str)
    print(f"OPAP returned {len(new_draws)} draws for today")

    if not new_draws:
        print("No draws found, exiting.")
        return

    # Load existing file
    existing_draws = []
    existing_ids = set()
    if os.path.exists(month_file):
        with open(month_file, encoding="utf-8") as f:
            existing = json.load(f)
        existing_draws = existing.get("draws", [])
        existing_ids = {d["id"] for d in existing_draws}

    # Merge only new draws
    added = [d for d in new_draws if d["id"] not in existing_ids]
    if not added:
        print("No new draws to add.")
        return

    all_draws = sorted(existing_draws + added, key=lambda d: d["id"])
    out = {"month": month_str, "draws": all_draws}

    os.makedirs("data/raw", exist_ok=True)
    with open(month_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Added {len(added)} new draws. Total: {len(all_draws)}. Last id: {all_draws[-1]['id']}")


if __name__ == "__main__":
    main()
