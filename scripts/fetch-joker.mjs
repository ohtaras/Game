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

// item.drawTime is a Unix epoch in milliseconds. Converted in the Europe/Athens timezone
// so the calendar date matches what was actually on the OPAP ticket that evening.
function extractDate(item) {
  if (typeof item?.drawTime !== 'number') return null;
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: 'Europe/Athens', year: 'numeric', month: '2-digit', day: '2-digit' })
    .formatToParts(new Date(item.drawTime));
  const y = parts.find(p => p.type === 'year').value;
  const m = parts.find(p => p.type === 'month').value;
  const d = parts.find(p => p.type === 'day').value;
  return `${y}-${m}-${d}`;
}

function toRecord(item) {
  const list = item?.winningNumbers?.list;
  const bonus = item?.winningNumbers?.bonus;
  if (!Array.isArray(list) || list.length !== 5 || !Array.isArray(bonus) || bonus.length !== 1) return null;
  const dt = extractDate(item);
  return { id: item.drawId, n: [...list].sort((a, b) => a - b), b: bonus[0], ...(dt ? { dt } : {}) };
}

async function fetchDrawId(base, id) {
  const url = `${base}/draw-id/${id}/${id}`;
  try {
    const res = await fetch(url, { headers: { Accept: 'application/json' } });
    if (!res.ok) return [];
    const raw = await res.json();
    return Array.isArray(raw) ? raw : (raw.content || raw.draws || []);
  } catch (e) { return []; }
}

async function getLatestDrawId(base) {
  const res = await fetch(`${base}/last-result-and-active`, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const raw = await res.json();
  const id = raw?.last?.drawId || raw?.active?.drawId;
  if (!id) throw new Error('Could not determine latest Joker draw ID.');
  return id;
}

// draw-date ranges (even a single day) return HTTP 400 for this game ID. Querying by
// draw-id/{id}/{id} one number at a time works instead — scan every id up to the latest.
async function fetchAllByDrawId(base) {
  const latestId = await getLatestDrawId(base);
  let items = [];
  for (let id = 1; id <= latestId; id++) {
    const idItems = await fetchDrawId(base, id);
    items = items.concat(idItems);
    if (id % 200 === 0 || id === latestId) console.log(`  drawId ${id}/${latestId}, ${items.length} draws found so far`);
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
  const byId = new Map(existing.map(d => [d.id, d]));
  const items = await fetchAllByDrawId(base);
  let newCount = 0, datesFilled = 0;
  for (const item of items) {
    const rec = toRecord(item);
    if (!rec) continue;
    const prior = byId.get(rec.id);
    if (!prior) { byId.set(rec.id, rec); newCount++; }
    else if (rec.dt && !prior.dt) { prior.dt = rec.dt; datesFilled++; }
  }
  const total = saveDraws([...byId.values()]);
  console.log(`Full backfill: ${items.length} fetched, ${newCount} new, ${datesFilled} dates filled, ${total} total.`);
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
