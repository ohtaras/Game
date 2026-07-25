#!/usr/bin/env python3
"""
Fetch ΚΙΝΟ (1100) historical draws from OPAP API and save as monthly JSON files.
Usage:
    python3 data/fetch_kino.py 2026-05 2026-07
    (fetches from May 2026 to July 2026 inclusive)
"""

import json, sys, time, urllib.request, urllib.error
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

GAME_ID  = 1100
OUT_DIR  = Path(__file__).parent / 'raw'
BASE_URL = 'https://api.opap.gr/draws/v3.0'

def fetch_day(day_str):
    url = f'{BASE_URL}/{GAME_ID}/draw-date/{day_str}/{day_str}?page=0&size=500'
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def parse_item(item):
    wn  = item.get('winningNumbers') or {}
    lst = [int(n) for n in (wn.get('list') or []) if 1 <= int(n) <= 80]
    if len(lst) < 20:
        return None
    b = None
    bonus = wn.get('bonus') or []
    if bonus:
        b = int(bonus[0])
    dt = (item.get('drawTime') or '')[:10]  # YYYY-MM-DD
    return {'id': item['drawId'], 'n': lst[:20], 'b': b, 'd': dt}

def date_range(start_str, end_str):
    d = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    while d <= end:
        yield d
        d += timedelta(days=1)

def month_key(d):
    return f'{d.year}_{d.month:02d}'

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 data/fetch_kino.py 2026-05 2026-07')
        print('  (provide first day of start month and last day of end month)')
        sys.exit(1)

    # Accept YYYY-MM or YYYY-MM-DD
    raw_start = sys.argv[1]
    raw_end   = sys.argv[2]
    if len(raw_start) == 7: raw_start += '-01'
    if len(raw_end)   == 7:
        y, m = int(raw_end[:4]), int(raw_end[5:7])
        # last day of that month
        if m == 12: raw_end = f'{y+1}-01-01'
        else:       raw_end = f'{y}-{m+1:02d}-01'
        raw_end = str(date.fromisoformat(raw_end) - timedelta(days=1))

    start_date = date.fromisoformat(raw_start)
    end_date   = min(date.fromisoformat(raw_end), date.today())

    print(f'Fetching ΚΙΝΟ {start_date} → {end_date}')

    # Load existing monthly data to avoid re-fetching
    existing = defaultdict(dict)  # month_key -> {day_str -> [draws]}
    for f in OUT_DIR.glob('kino_raw_*.json'):
        mk = f.stem.replace('kino_raw_', '')
        try:
            data = json.loads(f.read_text())
            for draw in (data.get('draws') or []):
                d = draw.get('d', '')
                if d:
                    existing[mk].setdefault(d, []).append(draw)
        except Exception:
            pass

    monthly = defaultdict(list)

    for day in date_range(str(start_date), str(end_date)):
        mk  = month_key(day)
        day_str = str(day)

        if day_str in existing.get(mk, {}):
            draws = existing[mk][day_str]
            print(f'  {day_str}: cached ({len(draws)} draws)')
            monthly[mk].extend(draws)
            continue

        retries = 3
        for attempt in range(retries):
            try:
                raw = fetch_day(day_str)
                items = raw.get('content') or []
                draws = [p for p in (parse_item(it) for it in items) if p]
                monthly[mk].extend(draws)
                print(f'  {day_str}: {len(draws)} draws fetched')
                time.sleep(0.3)
                break
            except Exception as e:
                print(f'  {day_str}: ERROR ({e})', end='')
                if attempt < retries - 1:
                    print(' retrying…')
                    time.sleep(2 ** attempt)
                else:
                    print(' skipped')

    # Write monthly files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for mk, draws in sorted(monthly.items()):
        # Merge with existing
        prev_draws = []
        out_path = OUT_DIR / f'kino_raw_{mk}.json'
        if out_path.exists():
            try:
                prev_draws = json.loads(out_path.read_text()).get('draws', [])
            except Exception:
                pass
        # Deduplicate by id
        all_draws = {d['id']: d for d in prev_draws}
        all_draws.update({d['id']: d for d in draws})
        sorted_draws = sorted(all_draws.values(), key=lambda d: d['id'])
        y, m = mk.split('_')
        out_path.write_text(json.dumps({'month': f'{y}-{m}', 'draws': sorted_draws}, ensure_ascii=False))
        saved.append(f'{out_path.name} ({len(sorted_draws)} draws)')
        print(f'Saved: {out_path.name} — {len(sorted_draws)} draws total')

    print('\nΈτοιμο! Αρχεία:')
    for s in saved:
        print(f'  data/raw/{s}')
    print('\nΤώρα τρέξε:')
    print('  git add data/raw/kino_raw_2026_*.json')
    print('  git commit -m "ΚΙΝΟ: monthly raw data Jun-Jul 2026"')
    print('  git push origin main')

if __name__ == '__main__':
    main()
