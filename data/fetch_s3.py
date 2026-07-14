#!/usr/bin/env python3
"""Fetch Super 3 (2100) draws from OPAP and store in data/raw/s3_raw_YYYY_MM.json.

Usage:
  python3 data/fetch_s3.py                  # fetch today
  python3 data/fetch_s3.py 2026-07-14       # fetch a specific date
  python3 data/fetch_s3.py 2026-01 2026-07  # backfill months (YYYY-MM range)

Super 3 result: a 3-digit number 000-999.
OPAP API typically returns it in winningNumbers.list as [d1, d2, d3] (three single digits).
"""

import json
import os
import sys
from datetime import date, timedelta, datetime, timezone

import requests

GAME_ID   = 2100
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


def parse_s3_number(nums_list):
    """Extract the 3-digit number from winningNumbers.list."""
    if not nums_list:
        return None
    if len(nums_list) == 3:
        try:
            d0, d1, d2 = int(nums_list[0]), int(nums_list[1]), int(nums_list[2])
            if all(0 <= x <= 9 for x in (d0, d1, d2)):
                return d0 * 100 + d1 * 10 + d2
        except (ValueError, TypeError):
            pass
        # Maybe they're stored as three separate values not necessarily single digits
        try:
            combined = int("".join(str(x) for x in nums_list))
            if 0 <= combined <= 999:
                return combined
        except (ValueError, TypeError):
            pass
    if len(nums_list) == 1:
        try:
            n = int(nums_list[0])
            if 0 <= n <= 999:
                return n
        except (ValueError, TypeError):
            pass
    return None


def fetch_draws_for_date(date_str):
    url = f"{OPAP_BASE}/draws/v3.0/{GAME_ID}/draw-date/{date_str}/{date_str}?page=0&size=500"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    draws = []
    for item in data.get("content", []):
        dt_ms = item.get("drawTime", 0)
        if not dt_ms:
            continue
        d = datetime.utcfromtimestamp(dt_ms / 1000) + timedelta(hours=3)
        day_str = d.strftime("%Y-%m-%d")
        mins    = d.hour * 60 + d.minute
        nums    = item.get("winningNumbers", {}).get("list", [])
        n = parse_s3_number(nums)
        if n is None:
            print(f"  ⚠ Could not parse nums_list={nums} for drawId={item.get('drawId')}")
            continue
        draws.append({
            "id": item["drawId"],
            "d": day_str,
            "m": mins,
            "n": n,
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
        json.dump({"game": "super3", "game_id": GAME_ID, "month": month_str, "draws": all_draws},
                  f, ensure_ascii=False, separators=(",", ":"))
    print(f"  +{len(added)} draws → {month_file} (total {len(all_draws)})")
    return len(added)


def fetch_date(date_str):
    print(f"Fetching Super 3 {date_str}…")
    draws = fetch_draws_for_date(date_str)
    print(f"  OPAP returned {len(draws)} draws")
    if not draws:
        return
    month_str  = date_str[:7]
    month_file = f"data/raw/s3_raw_{month_str.replace('-','_')}.json"
    merge_into_file(month_file, draws, month_str)


def fetch_month_range(start_ym, end_ym):
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    cur = date(sy, sm, 1)
    end = date(ey, em, 1)
    while cur <= end:
        day = cur
        while day.month == cur.month:
            fetch_date(day.strftime("%Y-%m-%d"))
            day += timedelta(days=1)
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
        print("Usage: fetch_s3.py [date|YYYY-MM] [end-YYYY-MM]")
