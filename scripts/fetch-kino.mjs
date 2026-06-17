// Server-side fetch of recent Kino draws from OPAP, merged into data/raw/*.json.
// Runs on a schedule via .github/workflows/fetch-kino.yml so draws keep getting
// saved even when nobody has kino_live.html open in a browser.
import fs from 'fs';
import path from 'path';

const OPAP_BASE = 'https://api.opap.gr/draws/v3.0/1100';
const LIVE_BASE_ID = 1155022; // draw id at 2025-01-01 00:00 EET
const LIVE_BASE_MS = Date.UTC(2024, 11, 31, 22, 0, 0);
const DRAWS_PER_DAY = 288; // every 5 minutes
const DATA_DIR = path.join(process.cwd(), 'data', 'raw');
const DAYS_BACK = 3; // re-check the last few days to fill any gaps

const pad = n => String(n).padStart(2, '0');

function athensDateStr(d) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Europe/Athens', year: 'numeric', month: '2-digit', day: '2-digit'
  }).formatToParts(d);
  const m = {};
  for (const p of parts) m[p.type] = p.value;
  return `${m.year}-${m.month}-${m.day}`;
}

function monthKeyForId(id) {
  const dayN = Math.floor((id - LIVE_BASE_ID) / DRAWS_PER_DAY);
  const dt = new Date(Date.UTC(2025, 0, 1) + dayN * 86400000);
  return `${dt.getUTCFullYear()}_${pad(dt.getUTCMonth() + 1)}`;
}

async function fetchDaySeq(dayStr, size = 50) {
  let items = [], page = 0;
  while (true) {
    const url = `${OPAP_BASE}/draw-date/${dayStr}/${dayStr}?page=${page}&size=${size}`;
    let raw;
    try {
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!res.ok) break;
      raw = await res.json();
    } catch (e) { console.error('fetch failed', dayStr, page, e.message); break; }
    const pg = Array.isArray(raw) ? raw : (raw.content || raw.draws || []);
    items = items.concat(pg);
    if (Array.isArray(raw) || raw.last === true || pg.length === 0 || pg.length < size) break;
    page++;
    if (page > 50) break;
  }
  return items;
}

function toRecord(item) {
  if (!item?.winningNumbers?.list?.length) return null;
  return {
    id: item.drawId,
    n: [...item.winningNumbers.list].sort((a, b) => a - b),
    b: (item.winningNumbers.bonus || [])[0] || null
  };
}

async function main() {
  const now = new Date();
  const byMonth = {};
  for (let i = 0; i < DAYS_BACK; i++) {
    const day = athensDateStr(new Date(now.getTime() - i * 86400000));
    const items = await fetchDaySeq(day);
    for (const item of items) {
      const rec = toRecord(item);
      if (!rec) continue;
      const ym = monthKeyForId(rec.id);
      (byMonth[ym] ||= []).push(rec);
    }
  }

  let totalNew = 0;
  for (const [ym, draws] of Object.entries(byMonth)) {
    const file = path.join(DATA_DIR, `kino_raw_${ym}.json`);
    const mStr = ym.replace('_', '-');
    let existing = { month: mStr, draws: [] };
    if (fs.existsSync(file)) existing = JSON.parse(fs.readFileSync(file, 'utf8'));
    const exIds = new Set(existing.draws.map(d => d.id));
    const newOnes = draws.filter(d => !exIds.has(d.id));
    if (!newOnes.length) continue;
    const merged = [...existing.draws, ...newOnes].sort((a, b) => a.id - b.id);
    fs.writeFileSync(file, JSON.stringify({ month: mStr, draws: merged }));
    totalNew += newOnes.length;
    console.log(`${ym}: +${newOnes.length} new draws (total ${merged.length})`);
  }

  console.log(`Done. Total new draws: ${totalNew}`);
}

main().catch(e => { console.error(e); process.exit(1); });
