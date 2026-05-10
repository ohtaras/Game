#!/usr/bin/env python3
"""
Fetch KINO draw results from OPAP API and push to GitHub automatically.

Usage:
    python fetch_kino.py [YYYY-MM-DD] [--token YOUR_GITHUB_TOKEN]

Requires:
    pip install requests

Set token via env var to avoid passing it as argument:
    export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
    python fetch_kino.py 2026-05-09
"""

import json
import sys
import os
import base64
import argparse
from datetime import date

import requests

# --- Config ---
GAME_ID = 1100
OPAP_BASE = "https://api.opap.gr"
GITHUB_REPO = "ohtaras/game"
GITHUB_BRANCH = "main"
# --------------

OPAP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Origin": "https://www.opap.gr",
    "Referer": "https://www.opap.gr/",
}


def fetch_draws(draw_date: str) -> list:
    url = f"{OPAP_BASE}/draws/v3.0/{GAME_ID}/draw-date/{draw_date}"
    resp = requests.get(url, headers=OPAP_HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    draws = []
    for item in raw.get("content", []):
        draws.append({
            "draw_id": item.get("drawId"),
            "draw_time": item.get("drawTime"),
            "winning_numbers": item.get("winningNumbers", {}).get("list", []),
            "bonus": item.get("winningNumbers", {}).get("bonus", []),
        })
    return draws


def push_to_github(token: str, filepath: str, content: str, draw_date: str):
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filepath}"
    gh_headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    # Check if file already exists (need sha to update)
    sha = None
    resp = requests.get(api_url, headers=gh_headers, params={"ref": GITHUB_BRANCH})
    if resp.status_code == 200:
        sha = resp.json()["sha"]

    payload = {
        "message": f"KINO results for {draw_date}",
        "content": base64.b64encode(content.encode()).decode(),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(api_url, headers=gh_headers, json=payload)
    resp.raise_for_status()
    print(f"Pushed to GitHub: {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?", default=date.today().isoformat(),
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub personal access token (or set GITHUB_TOKEN env var)")
    args = parser.parse_args()

    draw_date = args.date
    print(f"Fetching KINO draws for {draw_date} ...")

    draws = fetch_draws(draw_date)
    print(f"Found {len(draws)} draws.")

    result = {"date": draw_date, "total_draws": len(draws), "draws": draws}
    content = json.dumps(result, ensure_ascii=False, indent=2)

    # Save locally
    local_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(local_dir, exist_ok=True)
    local_file = os.path.join(local_dir, f"kino_{draw_date.replace('-', '')}.json")
    with open(local_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved locally: {local_file}")

    # Push to GitHub if token available
    if args.token:
        github_path = f"kino/data/kino_{draw_date.replace('-', '')}.json"
        push_to_github(args.token, github_path, content, draw_date)
    else:
        print("No GITHUB_TOKEN found — skipping GitHub push.")
        print("Set GITHUB_TOKEN env var or use --token flag to enable auto-push.")


if __name__ == "__main__":
    main()
