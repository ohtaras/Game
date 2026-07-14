#!/usr/bin/env python3
"""Fetch Power Spin (1110) draws from OPAP and store in data/raw/ps_raw_YYYY_MM.json.

Usage:
  python3 data/fetch_ps.py                  # fetch today
  python3 data/fetch_ps.py 2026-07-14       # fetch a specific date
  python3 data/fetch_ps.py 2026-01 2026-07  # backfill months (YYYY-MM range)
"""

import json
import os
import sys
from datetime import date, timedelta, datetime, timezone

import requests

GAME_ID   = 1110
OPAP_BASE = "https://api.opap.gr"
HEADERS   = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://www.opap.gr",
    "Referer": "https://www.opap.gr/",
}
ATHENS = timezone(timedelta(hours=3))


def athens_now():
    return datetime.now(ATHENS)


def fetch_draws_for_date(date_str):
    url = f"{OPAP_BASE}/draws/v3.0/{GAME_ID}/draw-date/{date_str}/{date_str}?page=0&size=300"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    draws = []
    for item in data.get("content", []):
        dt_ms = item.get("drawTime", 0)
        if not dt_ms:
            continue
        d = datetime.utcfromtimestamp(dt_ms / 1000) + timedelta(hours=3)
        day_str   = d.strftime("%Y-%m-%d")
        mins      = d.hour * 60 + d.minute
        nums_list = item.get("winningNumbers", {}).get("list", [])
        if len(nums_list) < 1:
            continue
        # Map values: 1-24 = number, anything else = symbol (25-27)
        w = []
        for v in nums_list[:3]:
            try:
                n = int(v)
                w.append(n if 1 <= n <= 24 else 25)
            except (ValueError, TypeError):
                w.append(25)
        while len(w) < 3:
            w.append(None)
        draws.append({
            "id": item["drawId"],
            "d": day_str,
            "m": mins,
            "w": w,
        })
    return sorted(draws, key=lambda x: x["id"])


def merge_into_file(month_file, new_draws, month_str):
    existing_draws, existing_ids = [], set()
    if os.path.exists(month_file):
        with open(month_file, encoding="utf-8") as f:
            existing = json.load(f)
        existing_draws = existing.get("draws", [])
        existing_ids   = {d["id"] for d in existing_draws}

    added = [d for d in new_draws if d["id"] not in existing_ids]
    if not added:
        print(f"  No new draws for {month_str}.")
        return 0

    all_draws = sorted(existing_draws + added, key=lambda x: x["id"])
    os.makedirs(os.path.dirname(month_file), exist_ok=True)
    with open(month_file, "w", encoding="utf-8") as f:
        json.dump({"game": "power_spin", "game_id": GAME_ID, "month": month_str, "draws": all_draws},
                  f, ensure_ascii=False, separators=(",", ":"))
    print(f"  +{len(added)} draws → {month_file} (total {len(all_draws)})")
    return len(added)


def fetch_date(date_str):
    print(f"Fetching Power Spin {date_str}…")
    draws = fetch_draws_for_date(date_str)
    print(f"  OPAP returned {len(draws)} draws")
    if not draws:
        return
    month_str  = date_str[:7]
    month_file = f"data/raw/ps_raw_{month_str.replace('-','_')}.json"
    merge_into_file(month_file, draws, month_str)


def fetch_month_range(start_ym, end_ym):
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    cur = date(sy, sm, 1)
    end = date(ey, em, 1)
    while cur <= end:
        # iterate each day in this month
        day = cur
        while day.month == cur.month:
            fetch_date(day.strftime("%Y-%m-%d"))
            day += timedelta(days=1)
        # next month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        fetch_date(athens_now().strftime("%Y-%m-%d"))
    elif len(args) == 1:
        arg = args[0]
        if len(arg) == 7:  # YYYY-MM
            fetch_month_range(arg, arg)
        else:
            fetch_date(arg)
    elif len(args) == 2:
        fetch_month_range(args[0], args[1])
    else:
        print("Usage: fetch_ps.py [date|YYYY-MM] [end-YYYY-MM]")
