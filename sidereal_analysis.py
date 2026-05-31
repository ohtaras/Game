#!/usr/bin/env python3
"""
Sidereal time correlation analysis for KINO draws.

Anchor: Draw #1303293 at 23:55 EEST (UTC+3) on 2026-05-31
Minutes per draw: computed from first draw estimate (~Jan 1, 2024)
Athens: 37.97°N, 23.73°E

Local Sidereal Time (LST) = Greenwich Mean Sidereal Time + 23.73°
"""
import json, time, math, numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

DATA_DIR = Path('/home/user/Game/data/raw')

# ── Anchor ─────────────────────────────────────────────────────────────────
ANCHOR_ID   = 1303293
ANCHOR_DATE = datetime(2026, 5, 31, 23, 55, 0,
                       tzinfo=timezone(timedelta(hours=3)))  # EEST
ANCHOR_UTC  = ANCHOR_DATE.astimezone(timezone.utc)
ATHENS_LON  = 23.73   # degrees East

# ── Astronomical utilities ─────────────────────────────────────────────────
def julian_date(dt_utc: datetime) -> float:
    """Convert UTC datetime to Julian Date (standard formula)."""
    y, mo, d = dt_utc.year, dt_utc.month, dt_utc.day
    h = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    if mo <= 2:
        y -= 1; mo += 12
    A = int(y/100); B = 2 - A + int(A/4)
    JD = int(365.25*(y+4716)) + int(30.6001*(mo+1)) + d + h/24 + B - 1524.5
    return JD

def lst_hours(dt_utc: datetime, lon_deg: float) -> float:
    """Local Sidereal Time in decimal hours for given UTC datetime and longitude."""
    JD = julian_date(dt_utc)
    T  = (JD - 2451545.0) / 36525.0
    # Greenwich Mean Sidereal Time (degrees)
    GMST = (280.46061837
            + 360.98564736629 * (JD - 2451545.0)
            + 0.000387933 * T**2
            - T**3 / 38710000.0) % 360.0
    LST_deg = (GMST + lon_deg) % 360.0
    return LST_deg / 15.0   # degrees → hours

# Verify anchor
jd_anchor = julian_date(ANCHOR_UTC)
lst_anchor = lst_hours(ANCHOR_UTC, ATHENS_LON)
print(f"Anchor: Draw #{ANCHOR_ID}  UTC {ANCHOR_UTC.strftime('%Y-%m-%d %H:%M')}  "
      f"JD={jd_anchor:.4f}")
print(f"  LST at Athens: {int(lst_anchor):02d}h {int((lst_anchor%1)*60):02d}m")

# ── Load draws ─────────────────────────────────────────────────────────────
print("\nLoading draws...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], frozenset(d['n'])))

all_draws.sort(key=lambda x: x[0])
draw_ids  = [d[0] for d in all_draws]
draw_nums = [d[1] for d in all_draws]
N = len(all_draws)
print(f"  {N} draws  ids {draw_ids[0]}..{draw_ids[-1]}  in {time.time()-t0:.1f}s")

# ── Estimate minutes-per-draw ──────────────────────────────────────────────
# First draw id is draw_ids[0]. Anchor is ANCHOR_ID / ANCHOR_UTC.
id_diff = ANCHOR_ID - draw_ids[0]
# Estimate start = Jan 1, 2024 00:35 UTC (from prior analysis ≈5.28 min/draw)
# Compute min/draw so that first draw lands near Jan 1, 2024 00:00
min_per_draw = 5.28  # minutes between draws

# Verify: first draw estimated time
td_first = timedelta(minutes=(draw_ids[0] - ANCHOR_ID) * min_per_draw)
first_utc = ANCHOR_UTC + td_first
print(f"\n  min/draw = {min_per_draw:.3f}")
print(f"  First draw #{draw_ids[0]} estimated UTC: {first_utc.strftime('%Y-%m-%d %H:%M')}")
print(f"  Last draw  #{draw_ids[-1]} estimated UTC: "
      f"{(ANCHOR_UTC + timedelta(minutes=(draw_ids[-1]-ANCHOR_ID)*min_per_draw)).strftime('%Y-%m-%d %H:%M')}")

# Sidereal period in draws
LST_PERIOD_MIN  = 1436.068  # sidereal day in minutes
LST_PERIOD_DRAWS = LST_PERIOD_MIN / min_per_draw
print(f"  Sidereal period: {LST_PERIOD_MIN:.1f} min = {LST_PERIOD_DRAWS:.1f} draws")

# ── Compute LST for each draw ──────────────────────────────────────────────
print("\nComputing LST for each draw...")
t0 = time.time()

# LST changes by 360.98564736629°/day = 24.06571 hours/day
# or equivalently, LST changes by (24.06571/24)*15° per hour of solar time
# Simpler: compute anchor LST, then for each draw:
# LST[i] = (lst_anchor + (draw_ids[i] - ANCHOR_ID) * min_per_draw * 360.98564736629 / 1440) % 360 / 15

SIDEREAL_RATE = 360.98564736629  # degrees per solar day

lst_anchor_deg = lst_anchor * 15.0
lst_all = np.array([
    (lst_anchor_deg + (did - ANCHOR_ID) * min_per_draw * SIDEREAL_RATE / 1440.0) % 360.0
    for did in draw_ids
])
lst_hours_all = lst_all / 15.0  # 0..24

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  LST range: {lst_hours_all.min():.3f} .. {lst_hours_all.max():.3f} h")

# ── Analysis 1: per-number frequency by LST bin (24 bins × 60 min) ────────
N_BINS = 24
print(f"\n── Analysis 1: Number frequency by LST hour ({N_BINS} bins) ──")
t0 = time.time()

bin_total = np.zeros(N_BINS, dtype=np.int64)   # draws in each bin
bin_counts = np.zeros((81, N_BINS), dtype=np.int64)  # appearances per number per bin

for i, (draw, lst_h) in enumerate(zip(draw_nums, lst_hours_all)):
    b = int(lst_h) % N_BINS
    bin_total[b] += 1
    for n in draw:
        bin_counts[n][b] += 1

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Draws per LST bin: min={bin_total.min()}  max={bin_total.max()}  "
      f"expected={N//N_BINS}")

# Find numbers with significant LST bias
print(f"\n  Numbers with max |z| > 3.5 across LST bins:")
print(f"  {'Num':>4}  {'LST_peak':>9}  {'Rate_peak':>10}  {'LST_trough':>11}  {'Rate_trough':>12}  {'max_z':>7}")
outliers = []
for n in range(1, 81):
    row = bin_counts[n]
    expected = bin_total * (20/80)
    var      = expected * (1 - 20/80)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(var > 0, (row - expected) / np.sqrt(var), 0)
    max_z  = np.max(np.abs(z))
    peak_b = np.argmax(z)
    trough_b = np.argmin(z)
    if max_z > 3.5:
        rate_peak   = row[peak_b] / bin_total[peak_b]
        rate_trough = row[trough_b] / bin_total[trough_b]
        outliers.append((max_z, n, peak_b, rate_peak, trough_b, rate_trough, z[peak_b], z[trough_b]))

outliers.sort(reverse=True)
if outliers:
    for max_z, n, pb, rp, tb, rt, zp, zt in outliers[:20]:
        rg, cg = (n-1)//10, (n-1)%10
        print(f"  {n:>4} (r{rg}c{cg})  LST {pb:02d}:xx  rate={rp:.4f} (z={zp:+.2f})  "
              f"LST {tb:02d}:xx  rate={rt:.4f} (z={zt:+.2f})  max_z={max_z:.2f}")
else:
    print("  None found.")

# ── Analysis 2: Global LST bias — any hour better than random for all 20? ─
print(f"\n── Analysis 2: Global hit rate by LST hour ──")
print(f"  (if LST affects all numbers equally, one hour might have more total hits)")
# For this we look at the NEXT draw: how many numbers from draw i appear in draw i+1
# grouped by LST of draw i
print(f"  {'LST_h':>6}  {'Draws':>7}  {'Total_hits':>11}  {'E_hits':>8}  {'Rate':>7}  {'z':>7}")

next_hits_by_bin   = np.zeros(N_BINS, dtype=np.int64)
next_draws_by_bin  = np.zeros(N_BINS, dtype=np.int64)
for i in range(N-1):
    b = int(lst_hours_all[i]) % N_BINS
    hits = len(draw_nums[i] & draw_nums[i+1])
    next_hits_by_bin[b]  += hits
    next_draws_by_bin[b] += 1

global_rate = next_hits_by_bin.sum() / next_draws_by_bin.sum() / 20  # per number

best_bin = -1; best_rate = 0
for b in range(N_BINS):
    nd = next_draws_by_bin[b]
    if nd == 0: continue
    rate = next_hits_by_bin[b] / nd / 20
    exp  = nd * 20 * 0.25
    z    = (next_hits_by_bin[b] - exp) / math.sqrt(exp * 0.75)
    flag = " ★" if abs(z) > 3 else ""
    print(f"  {b:02d}:xx  {nd:>7}  {next_hits_by_bin[b]:>11}  {exp:>8.0f}  {rate:>7.4f}  {z:>+7.2f}{flag}")
    if rate > best_rate:
        best_rate = rate; best_bin = b

print(f"\n  Overall rate: {global_rate:.4f}  (baseline: 0.2500)")
print(f"  Best LST hour: {best_bin:02d}:xx  rate={best_rate:.4f}")

# ── Analysis 3: LST "hot zone" for 7+/20 overlap ──────────────────────────
print(f"\n── Analysis 3: Does LST predict 7+ overlap with next draw? ──")
hits7_by_bin = np.zeros(N_BINS, dtype=np.int64)
for i in range(N-1):
    b = int(lst_hours_all[i]) % N_BINS
    if len(draw_nums[i] & draw_nums[i+1]) >= 7:
        hits7_by_bin[b] += 1

total_7plus = hits7_by_bin.sum()
print(f"  Total draws with 7+ overlap to next: {total_7plus}  "
      f"({total_7plus/(N-1)*100:.2f}%)")
exp_per_bin = total_7plus / N_BINS
print(f"  Expected per LST bin: {exp_per_bin:.1f}")
print(f"  {'LST_h':>6}  {'7+_hits':>8}  {'z':>7}")
for b in range(N_BINS):
    z = (hits7_by_bin[b] - exp_per_bin) / math.sqrt(exp_per_bin)
    flag = " ★" if abs(z) > 2.5 else ""
    print(f"  {b:02d}:xx  {hits7_by_bin[b]:>8}  {z:>+7.2f}{flag}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n══════════════════════════════════════════")
print(f"ΣΥΜΠΕΡΑΣΜΑ — Αστρική Ώρα vs KINO")
print(f"══════════════════════════════════════════")
print(f"  Αγκύρωση: Draw #{ANCHOR_ID} → {first_utc.strftime('%Y-%m-%d %H:%M')} UTC")
print(f"  Κλήρωση κάθε {min_per_draw:.2f} λεπτά")
print(f"  Αστρική περίοδος: {LST_PERIOD_DRAWS:.1f} κληρώσεις (~{LST_PERIOD_MIN/60:.1f}h)")
print(f"  Τιμές LST με z>3.5: {len(outliers)} αριθμοί")
