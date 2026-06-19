// Server-side fetch of Joker (Τζόκερ) draws from OPAP, merged into data/raw/joker_raw.json.
// Joker draws 3x/week (Tue, Thu, Sun ~23:00 Athens time), so unlike Kino we keep all draws
// in one file instead of splitting by month.
//
// Run modes:
//   node scripts/fetch-joker.mjs            -> incremental: fetch only the latest draw
//   node scripts/fetch-joker.mjs --full      -> full backfill: paginate the entire history

import fs from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data', 'raw');
const DATA_FILE = path.join(DATA_DIR, 'joker_raw.json');
const JOKER_PRODUCT_ID = 5104;

function toRecord(item) {
  const list = item?.winningNumbers?.list;
  const bonus = item?.winningNumbers?.bonus;
  if (!Array.isArray(list) || list.length !== 5 || !Array.isArray(bonus) || bonus.length !== 1) return null;
  return { id: item.drawId, n: [...list].sort((a, b) => a - b), b: bonus[0] };
}

async function fetchRange(base, fromStr, toStr) {
  const url = `${base}/draw-date/${fromStr}/${toStr}`;
  try {
    const res = await fetch(url, { headers: { Accept: 'application/json' } });
    if (!res.ok) { console.error('fetch failed', fromStr, toStr, 'HTTP ' + res.status); return []; }
    const raw = await res.json();
    return Array.isArray(raw) ? raw : (raw.content || raw.draws || []);
  } catch (e) { console.error('fetch failed', fromStr, toStr, e.message); return []; }
}

// Even a 1-year draw-date range trips OPAP's API with HTTP 400, so the backfill instead
// queries one specific draw date at a time — Joker only draws Tue(2)/Thu(4)/Sun(0), so we
// only need to hit those dates, not every single day.
function jokerDrawDates(fromStr, toStr) {
  const dates = [];
  const day = new Date(fromStr + 'T00:00:00Z');
  const end = new Date(toStr + 'T00:00:00Z');
  while (day <= end) {
    const dow = day.getUTCDay();
    if (dow === 0 || dow === 2 || dow === 4) dates.push(day.toISOString().slice(0, 10));
    day.setUTCDate(day.getUTCDate() + 1);
  }
  return dates;
}

async function fetchAllDays(base) {
  const dates = jokerDrawDates('1997-01-01', new Date().toISOString().slice(0, 10));
  let items = [];
  for (let i = 0; i < dates.length; i++) {
    const dayItems = await fetchRange(base, dates[i], dates[i]);
    items = items.concat(dayItems);
    if ((i + 1) % 100 === 0 || i === dates.length - 1) console.log(`  ${dates[i]}: ${i + 1}/${dates.length} dates checked, ${items.length} draws found so far`);
  }
  return items;
}

function loadExisting() {
  if (!fs.existsSync(DATA_FILE)) return [];
  const data = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  return data?.draws || [];
}

function saveDraws(draws) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const sorted = [...draws].sort((a, b) => a.id - b.id);
  fs.writeFileSync(DATA_FILE, JSON.stringify({ draws: sorted }));
  return sorted.length;
}

async function runFull(base) {
  const existing = loadExisting();
  const items = await fetchAllDays(base);
  const fresh = items.map(toRecord).filter(Boolean);
  const merged = [...existing.filter(d => !fresh.some(f => f.id === d.id)), ...fresh];
  const total = saveDraws(merged);
  const newCount = merged.length - existing.length;
  console.log(`Full backfill: ${items.length} fetched, ${newCount} new, ${total} total.`);
}

async function runIncremental(base) {
  const existing = loadExisting();
  const last = existing[existing.length - 1];
  const res = await fetch(`${base}/last-result-and-active`, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const raw = await res.json();
  const rec = toRecord(raw?.last);
  if (!rec) { console.log('No valid latest draw in response.'); return; }
  if (last && rec.id <= last.id) { console.log('No new draw.'); return; }
  const merged = saveDraws([...existing, rec]);
  console.log(`+1 new draw (#${rec.id}), ${merged} total.`);
}

async function main() {
  const base = `https://api.opap.gr/draws/v3.0/${JOKER_PRODUCT_ID}`;
  if (process.argv.includes('--full')) await runFull(base);
  else await runIncremental(base);
}

main().catch(e => { console.error(e); process.exit(1); });
