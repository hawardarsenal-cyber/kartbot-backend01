// KartingCentral Chatbot Backend (patched router build)
// - Adds routing for tracking codes (in-chat ticket checker)
// - Ensures HTML-only output
// - Keeps existing /api/faq-response for compatibility

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const fsp = fs.promises;

const OpenAI = require('openai');

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ---------------------------
// Config
// ---------------------------
const PORT = process.env.PORT || 3001;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

// Knowledge sources (MD)
const FAQ_PATH = process.env.KC_FAQ_PATH || path.join(__dirname, 'FAQ.md');
const INSTR_PATH = process.env.KC_INSTRUCTIONS_PATH || path.join(__dirname, 'BOT_INSTRUCTIONS.md');

// Ticket / tracking record search paths (relative to repo root by default)
// You can override with env KC_TRACKING_ROOT.
const DEFAULT_TRACKING_ROOT = process.env.KC_TRACKING_ROOT || path.resolve(__dirname, '..', '..');
const TRACKING_SEARCH_FOLDERS = (process.env.KC_TRACKING_FOLDERS || 'logs,splits,tpr/data').split(',').map(s => s.trim()).filter(Boolean);

// ---------------------------
// OpenAI
// ---------------------------
if (!process.env.OPENAI_API_KEY) {
  console.warn('[WARN] OPENAI_API_KEY is not set. /api/chat will fail until configured.');
}
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------------------------
// Utilities
// ---------------------------
function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function stripCodeFences(s) {
  return String(s ?? '').replace(/```[\s\S]*?```/g, (m) => {
    // keep inner content if it's already HTML-ish; otherwise drop fences
    const inner = m.replace(/^```[a-zA-Z0-9_-]*\n?/, '').replace(/```$/, '');
    return inner;
  });
}

function enforceHtmlOnly(s) {
  // Goal: always return valid HTML (no Markdown).
  let out = stripCodeFences(String(s ?? '')).trim();

  // If model returned Markdown-style links, try a very small conversion:
  // [Label](url) -> <a href="url">Label</a>
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2">$1</a>');

  // Remove stray markdown bullets into <br> (keep simple)
  // Leading "- " or "• " lines -> "• ...<br>"
  out = out
    .split(/\r?\n/)
    .map(line => line.replace(/^\s*[-*]\s+/,'• '))
    .join('\n');

  // If it doesn't look like HTML, wrap it.
  const looksLikeHtml = /^\s*<\w[\s\S]*>\s*$/.test(out);
  if (!looksLikeHtml) {
    // Convert double newlines to paragraph breaks
    const parts = out.split(/\n\n+/).map(p => p.trim()).filter(Boolean);
    if (parts.length === 0) return '<div></div>';
    const html = parts.map(p => `<p>${escapeHtml(p).replace(/\n/g,'<br>')}</p>`).join('');
    return `<div>${html}</div>`;
  }

  // Still ensure we don't output markdown headers accidentally
  out = out.replace(/^\s*#+\s*/gm, '');
  return out;
}

function normalizeCodeInput(raw) {
  let s = String(raw ?? '').trim().toUpperCase();
  s = s.replace(/[^A-Z0-9\-]/g, '');

  // AA00000000 -> AA00-00-00-00
  const m1 = s.match(/^([A-Z]{2})(\d{8})$/);
  if (m1) {
    const digits = m1[2];
    return `${m1[1]}${digits.slice(0,2)}-${digits.slice(2,4)}-${digits.slice(4,6)}-${digits.slice(6,8)}`;
  }

  // AA00-00-00-00 or AA00000000 (strip and reformat)
  const stripped = s.replace(/-/g, '');
  const m2 = stripped.match(/^([A-Z]{2})(\d{8})$/);
  if (m2) {
    const digits = m2[2];
    return `${m2[1]}${digits.slice(0,2)}-${digits.slice(2,4)}-${digits.slice(4,6)}-${digits.slice(6,8)}`;
  }

  return s;
}

function looksLikeTrackingCode(text) {
  const t = String(text ?? '').trim();

  // Common formats seen in this project:
  // - Voucher/pack codes like rd12-010-070126-01 (prefix + numbers + date + sequence)
  // - Corporate codes like AA00-00-00-00
  const re1 = /\b([a-z]{1,4}\d{1,3})-(\d{2,3})-(\d{6})-(\d{2})\b/i;
  const re2 = /\b([A-Z]{2}\d{2}-\d{2}-\d{2}-\d{2})\b/;

  return re1.test(t) || re2.test(normalizeCodeInput(t));
}

function extractTrackingCandidate(text) {
  const t = String(text ?? '').trim();
  const m = t.match(/\b([a-z]{1,4}\d{1,3}-\d{2,3}-\d{6}-\d{2})\b/i);
  if (m) return m[1];
  const m2 = t.match(/\b([A-Z]{2}\d{2}-\d{2}-\d{2}-\d{2})\b/);
  if (m2) return m2[1];

  // last resort: if entire message looks like a code
  if (looksLikeTrackingCode(t)) return t;
  return '';
}

async function readTextIfExists(filePath) {
  try {
    return await fsp.readFile(filePath, 'utf8');
  } catch {
    return '';
  }
}

// ---------------------------
// Simple retrieval: chunk FAQ.md and pick top matches
// ---------------------------
function tokenize(s) {
  return String(s ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function scoreOverlap(queryTokens, chunkTokens) {
  if (!queryTokens.length || !chunkTokens.length) return 0;
  const set = new Set(chunkTokens);
  let hit = 0;
  for (const w of queryTokens) if (set.has(w)) hit++;
  return hit;
}

function chunkText(text, maxChars = 1200) {
  const paras = String(text ?? '').split(/\n\n+/).map(p => p.trim()).filter(Boolean);
  const chunks = [];
  let buf = '';
  for (const p of paras) {
    if ((buf + '\n\n' + p).length > maxChars) {
      if (buf.trim()) chunks.push(buf.trim());
      buf = p;
    } else {
      buf = buf ? (buf + '\n\n' + p) : p;
    }
  }
  if (buf.trim()) chunks.push(buf.trim());
  return chunks;
}

async function retrieveRelevantFaq(query, k = 4) {
  const faqText = await readTextIfExists(FAQ_PATH);
  if (!faqText) return '';

  const chunks = chunkText(faqText, 1200);
  const qTok = tokenize(query);

  const scored = chunks
    .map((c) => ({ c, s: scoreOverlap(qTok, tokenize(c)) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, k)
    .filter(x => x.s > 0)
    .map(x => x.c);

  return scored.join('\n\n---\n\n');
}

// ---------------------------
// Tracking lookup (JSON scan)
// ---------------------------
async function listJsonFilesRecursively(rootDir, maxFiles = 5000) {
  const out = [];
  async function walk(dir) {
    if (out.length >= maxFiles) return;
    let entries = [];
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of entries) {
      if (out.length >= maxFiles) return;
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        // skip node_modules and vendor-ish
        if (ent.name === 'node_modules' || ent.name === '.git') continue;
        await walk(p);
      } else if (ent.isFile() && ent.name.endsWith('.json')) {
        out.push(p);
      }
    }
  }
  await walk(rootDir);
  return out;
}

function candidateCodesFromRecord(obj) {
  const c = [];
  if (!obj || typeof obj !== 'object') return c;

  const keys = ['track_code', 'tracking_code', 'paidFullCode', 'freeFullCode', 'code', 'ticket_code', 'tracking'];
  for (const k of keys) {
    if (obj[k]) c.push(String(obj[k]));
  }
  return c;
}

async function findTrackingRecord(trackCodeRaw) {
  const norm = normalizeCodeInput(trackCodeRaw);

  for (const folder of TRACKING_SEARCH_FOLDERS) {
    const folderPath = path.join(DEFAULT_TRACKING_ROOT, folder);

    // Fast path: if it doesn't exist, continue
    try {
      const st = await fsp.stat(folderPath);
      if (!st.isDirectory()) continue;
    } catch {
      continue;
    }

    const jsonFiles = await listJsonFilesRecursively(folderPath, 3000);

    for (const filePath of jsonFiles) {
      let data;
      try {
        const raw = await fsp.readFile(filePath, 'utf8');
        data = JSON.parse(raw);
      } catch {
        continue;
      }

      // Support: object or array of objects
      const records = Array.isArray(data) ? data : [data];
      for (const rec of records) {
        const candidates = candidateCodesFromRecord(rec);
        for (const stored of candidates) {
          if (normalizeCodeInput(stored) === norm) {
            return { record: rec, filePath };
          }
        }
      }
    }
  }
  return null;
}

function renderTicketCheckerHtml({ code, found, record, filePath }) {
  const codeNice = escapeHtml(normalizeCodeInput(code));

  if (!found) {
    return `
<div>
  <p><strong>Got it — I can help check that code.</strong><br>
  I couldn’t find a match instantly in our local records for <strong>${codeNice}</strong>.</p>

  <p>Please double-check the characters (you can paste it again), or tell me the <strong>purchase email</strong> used at checkout / on the promo stand.</p>

  <p>If you’d rather do this on the booking page, use:<br>
  <a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php">Book Experience</a></p>
</div>`;
  }

  // Heuristic fields (don’t assume they exist)
  const buyerEmail = record?.buyer_email || record?.buyerEmail || record?.email || '';
  const buyerName  = record?.buyer_name || record?.buyerName || '';
  const qty        = record?.qty ?? record?.quantity ?? record?.tickets ?? '';
  const status     = record?.status ?? '';
  const createdAt  = record?.created_at || record?.createdAt || '';
  const redeemed   = record?.redeemed ?? record?.used ?? record?.tickets_used ?? '';
  const remaining  = record?.remaining ?? record?.tickets_remaining ?? '';

  const rows = [];
  const pushRow = (k, v) => {
    if (v === '' || v === null || typeof v === 'undefined') return;
    rows.push(`<tr><td style="padding:6px 10px;border:1px solid rgba(255,255,255,.12);"><strong>${escapeHtml(k)}</strong></td><td style="padding:6px 10px;border:1px solid rgba(255,255,255,.12);">${escapeHtml(v)}</td></tr>`);
  };

  pushRow('Ticket code', normalizeCodeInput(code));
  pushRow('Name', buyerName);
  pushRow('Email', buyerEmail);
  pushRow('Tickets / Qty', String(qty));
  pushRow('Redeemed', String(redeemed));
  pushRow('Remaining', String(remaining));
  pushRow('Status', String(status));
  pushRow('Created', String(createdAt));

  const fileHint = filePath ? `<p style="opacity:.75;font-size:.9em;">(Record source: ${escapeHtml(path.relative(DEFAULT_TRACKING_ROOT, filePath))})</p>` : '';

  return `
<div>
  <p><strong>✅ Code found:</strong> <strong>${codeNice}</strong></p>
  <table style="border-collapse:collapse;width:100%;max-width:720px;">
    <tbody>
      ${rows.join('') || `<tr><td style="padding:6px 10px;border:1px solid rgba(255,255,255,.12);"><strong>Ticket code</strong></td><td style="padding:6px 10px;border:1px solid rgba(255,255,255,.12);">${codeNice}</td></tr>`}
    </tbody>
  </table>
  ${fileHint}
  <p>If you want to <strong>book</strong> using this code, go to:<br>
  <a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php">Book Experience</a></p>
</div>`;
}

// ---------------------------
// Chat router
// ---------------------------
// =========================
// Bundles / Race Day / Corporate canned flows (HTML)
// =========================
const BUNDLE_PACKS = [
  { qty: 4,  price: 80,   validity: "6 months",  name: "Duo Inception Pack" },
  { qty: 8,  price: 160,  validity: "6 months",  name: "Spin Shield Pack" },
  { qty: 12, price: 240,  validity: "6 months",  name: "Target Tracker Pack" },
  { qty: 20, price: 400,  validity: "1 year",    name: "Sky Seeker Pack" },
  { qty: 32, price: 640,  validity: "1 year",    name: "Speed Token Pack" },
  { qty: 40, price: 800,  validity: "18 months", name: "Phase Drift Pack" },
  { qty: 52, price: 1040, validity: "18 months", name: "Aqua Rush Pack" },
  { qty: 60, price: 1200, validity: "2 years",   name: "Storm Pulse Pack" },
  { qty: 72, price: 1440, validity: "2 years",   name: "Nova Core Pack" },
];

function renderBundlesHtml() {
  const rows = BUNDLE_PACKS.map(p => {
    const ppt = p.qty ? Math.round(p.price / p.qty) : 0;
    return `<tr>
      <td style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.10)"><strong>${escapeHtml(p.name)}</strong><br><span style="opacity:.85">x${p.qty} tickets</span></td>
      <td style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.10);text-align:right"><strong>£${p.price}</strong><br><span style="opacity:.85">£${ppt}/ticket</span></td>
      <td style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.10);text-align:right"><span style="opacity:.9">${escapeHtml(p.validity)}</span></td>
    </tr>`;
  }).join("");

  return `
    <div>
      <strong>Ticket bundles</strong><br>
      You can buy a pack and share tickets across friends &amp; family (one buyer can cover the whole group). Tickets can be used across multiple visits until expiry.<br><br>

      <div style="overflow:auto;border:1px solid rgba(255,255,255,.12);border-radius:12px">
        <table style="width:100%;border-collapse:collapse;min-width:520px">
          <thead>
            <tr>
              <th style="text-align:left;padding:10px;border-bottom:1px solid rgba(255,255,255,.14)">Pack</th>
              <th style="text-align:right;padding:10px;border-bottom:1px solid rgba(255,255,255,.14)">Price</th>
              <th style="text-align:right;padding:10px;border-bottom:1px solid rgba(255,255,255,.14)">Validity</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div><br>

      Each ticket includes <a href="https://www.kartingcentral.co.uk/session-info">one full session</a> (gear included). To use Karting Central tickets for an hour slot, Session 2 &amp; 3 must be pre-booked at the same time.<br><br>

      Buy / book using your tracking code: <a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php">Book Experience</a><br>
      Manage gifting / sharing / PIN: <a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/cdashlogin.php">Customer Dashboard</a><br><br>

      Already have a pack? Paste your <strong>ticket code</strong> here and I’ll open the in-chat ticket checker.
    </div>
  `.trim();
}

function renderRaceDayHtml() {
  return `
    <div>
      <strong>Open Race Sessions (Race Day)</strong><br>
      Race Day is <strong>3 sessions</strong> (about <strong>1 hour on-site</strong> with short breaks) and is <strong>£29 per person</strong>.<br>
      Each ticket covers <a href="https://www.kartingcentral.co.uk/session-info">one full session</a> — Session 2 &amp; 3 must be pre-booked together to secure the hour slot and multi-session pricing.<br><br>

      <strong>Next step:</strong> choose track + date/time on <a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php">Book Experience</a>.<br>
      If you already have tickets, paste your <strong>ticket code</strong> here first and I’ll check what you have remaining.
    </div>
  `.trim();
}

function renderCorporateHtml() {
  return `
    <div>
      <strong>Corporate / Private Hire — 4-Race Grand Prix</strong><br>
      Corporate GP is <strong>£52 pp</strong> and includes food + room hire, plus private hire booking structure.<br><br>

      <strong>How checkout works (new flow):</strong><br>
      • <strong>No tracking code:</strong> you pay a <strong>£5 per driver deposit</strong> <em>and</em> a discounted ticket bundle at <strong>£5 per ticket</strong> (bundle size is auto-matched to your driver count).<br>
      • <strong>Tracking code entered &amp; valid:</strong> deposit + bundle add-on are skipped and you pay <strong>£52 × drivers</strong> via checkout.<br><br>

      Corporate bookings must be at least <strong>48 hours</strong> in advance for online deposit flows.<br><br>

      Start here: <a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php">Book Experience</a><br>
      Manage tickets / gifting / referrals: <a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/cdashlogin.php">Customer Dashboard</a><br><br>

      If you have an existing code, paste your <strong>ticket code</strong> into this chat and I’ll check it.
    </div>
  `.trim();
}

function looksLikeBundleIntent(text) {
  const t = (text || "").toLowerCase();
  return /\b(bundle|bundles|pack|package|packages|buy\s+tickets|purchase\s+tickets|ticket\s+bundles|voucher\s+pack|voucher\s+bundle)\b/i.test(t);
}
function looksLikeRaceDayIntent(text) {
  const t = (text || "").toLowerCase();
  return /\b(race\s*day|open\s*race|open\s*racing|open\s*session|open\s*sessions|book\s*a\s*session)\b/i.test(t);
}
function looksLikeCorporateIntent(text) {
  const t = (text || "").toLowerCase();
  return /\b(corporate|private\s*hire|grand\s*prix|team\s*building|company\s*event)\b/i.test(t);
}

async function routeChat({ message, context = {} }) {
  const text = String(message ?? '').trim();
  const candidate = extractTrackingCandidate(text);

  // Routing: tracking code in message => in-chat ticket checker
  if (candidate && looksLikeTrackingCode(candidate)) {
    const found = await findTrackingRecord(candidate);
    return {
      route: 'ticket_checker',
      responseHtml: renderTicketCheckerHtml({
        code: candidate,
        found: !!found,
        record: found?.record,
        filePath: found?.filePath,
      }),
      data: {
        code: normalizeCodeInput(candidate),
        found: !!found,
      },
    };
  }


  // Routing: bundles / race day / corporate (new bundles page logic)
  if (looksLikeCorporateIntent(text)) {
    return { route: 'corporate', responseHtml: renderCorporateHtml(), data: {} };
  }
  if (looksLikeRaceDayIntent(text)) {
    return { route: 'race_day', responseHtml: renderRaceDayHtml(), data: {} };
  }
  if (looksLikeBundleIntent(text)) {
    return { route: 'bundles', responseHtml: renderBundlesHtml(), data: {} };
  }

  
  // Routing: bundles / race day / corporate (new bundles page logic)
  if (looksLikeCorporateIntent(text)) {
    return { route: 'corporate', responseHtml: renderCorporateHtml(), data: {} };
  }
  if (looksLikeRaceDayIntent(text)) {
    return { route: 'race_day', responseHtml: renderRaceDayHtml(), data: {} };
  }
  if (looksLikeBundleIntent(text)) {
    return { route: 'bundles', responseHtml: renderBundlesHtml(), data: {} };
  }

// Otherwise: FAQ / assistant response
  const faqContext = await retrieveRelevantFaq(text, 4);
  const instr = await readTextIfExists(INSTR_PATH);

  const system = `You are Karting Central’s customer support bot.\n\n` +
    `IMPORTANT OUTPUT RULE: Return VALID HTML ONLY. Do not output Markdown.\n` +
    `Use <br> for short line breaks. Keep paragraphs short.\n` +
    `Use proper anchors for ALL links: <a href=\"URL\">Label</a>. Never print bare URLs.\n\n` +
    (instr ? `\n---\nBOT INSTRUCTIONS (authoritative):\n${instr}\n` : '');

  const user = `User message:\n${text}\n\n` +
    (faqContext ? `Relevant FAQ context (scan & use as source of truth):\n${faqContext}\n` : '');

  const completion = await client.chat.completions.create({
    model: OPENAI_MODEL,
    temperature: 0.2,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
  });

  const raw = completion?.choices?.[0]?.message?.content ?? '';
  const html = enforceHtmlOnly(raw);

  return {
    route: 'answer',
    responseHtml: html,
  };
}

// ---------------------------
// Routes
// ---------------------------
app.get('/health', (req, res) => res.json({ ok: true }));

// New unified route
app.post('/api/chat', async (req, res) => {
  try {
    const message = req.body?.message ?? req.body?.prompt ?? '';
    if (!String(message).trim()) {
      return res.status(400).json({ error: 'Missing message' });
    }

    const result = await routeChat({ message, context: req.body?.context || {} });
    return res.json(result);
  } catch (err) {
    console.error('[/api/chat] error:', err);
    return res.status(500).json({ error: 'Server error' });
  }
});

// Backwards-compatible endpoint
app.post('/api/faq-response', async (req, res) => {
  try {
    const prompt = req.body?.prompt || '';
    const result = await routeChat({ message: prompt, context: req.body?.context || {} });
    return res.json({ answer: result.responseHtml, route: result.route, data: result.data || null });
  } catch (err) {
    console.error('[/api/faq-response] error:', err);
    return res.status(500).json({ error: 'Server error' });
  }
});

app.listen(PORT, () => {
  console.log(`KC chatbot backend listening on :${PORT}`);
  console.log(`FAQ_PATH: ${FAQ_PATH}`);
  console.log(`INSTR_PATH: ${INSTR_PATH}`);
  console.log(`TRACKING_ROOT: ${DEFAULT_TRACKING_ROOT}`);
  console.log(`TRACKING_FOLDERS: ${TRACKING_SEARCH_FOLDERS.join(', ')}`);
});
