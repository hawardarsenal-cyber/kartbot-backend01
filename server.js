// server.js — GPT + remote KB (JSON) + external instructions (.md) + promo leads
// Adds: conversation memory + slot logic (track/day) + fast-routes that never default to Gillingham for "where is the track"

import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import fs from "fs/promises";
import { existsSync, mkdirSync } from "fs";
import path from "path";
import crypto from "crypto";

const app = express();
app.use(express.json({ limit: "1mb" }));

app.use(
  cors({
    origin: [
      "https://pos.kartingcentral.co.uk",
      "https://www.kartingcentral.co.uk",
      "http://localhost:3000",
    ],
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// ---------- OpenAI ----------
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini";
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

// ---------- Remote KB ----------
const KB_URL = process.env.KB_URL;
const PROMPT_URL = process.env.PROMPT_URL; // external Markdown instructions
const REFRESH_MS = 5 * 60 * 1000; // 5 minutes

let KB = null;
let DOCS = [];
let VECTORS = [];
let etag = null,
  lastModified = null;

let PROMPT_TEXT = null;
let promptEtag = null,
  promptLastMod = null;

// ---------- Utils ----------
const enforceGBP = (t) => (t || "").replace(/\$/g, "£");

function sha1(s = "") {
  return crypto.createHash("sha1").update(String(s)).digest("hex").slice(0, 10);
}

function cosine(a, b) {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i],
      y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

function normQ(s = "") {
  return String(s || "").toLowerCase().trim();
}
function hasAny(hay, needles = []) {
  return needles.some((n) => hay.includes(n));
}
function stripHtmlToText(html = "") {
  return String(html)
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/li>/gi, "\n")
    .replace(/<\/p>/gi, "\n")
    .replace(/<[^>]*>/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// ---------- Conversation memory (in-memory) ----------
// NOTE: resets on deploy/restart; good enough for now.
const SESS_TTL_MS = 30 * 60 * 1000; // 30 mins
const SESS_MAX_TURNS = 16; // last 8 pairs
const SESS = new Map(); // id -> { updatedAt, turns: [{role, content}], slots: {trackId, day} }

function getSess(id) {
  if (!id) return null;
  const s = SESS.get(id);
  if (!s) return null;
  if (Date.now() - s.updatedAt > SESS_TTL_MS) {
    SESS.delete(id);
    return null;
  }
  return s;
}

function upsertSess(id) {
  if (!id) return null;
  let s = getSess(id);
  if (!s) {
    s = { updatedAt: Date.now(), turns: [], slots: {} };
    SESS.set(id, s);
  }
  s.updatedAt = Date.now();
  return s;
}

function pushTurn(sess, role, content) {
  if (!sess) return;
  sess.turns.push({ role, content });
  if (sess.turns.length > SESS_MAX_TURNS) {
    sess.turns = sess.turns.slice(sess.turns.length - SESS_MAX_TURNS);
  }
}

function inferSessionId(req) {
  // If frontend doesn’t send sessionId, we derive a "best effort" stable-ish id.
  // Prefer x-forwarded-for (first IP) + UA.
  const xf = String(req.headers["x-forwarded-for"] || "").split(",")[0].trim();
  const ip = xf || req.socket.remoteAddress || "unknown";
  const ua = String(req.headers["user-agent"] || "ua");
  return sha1(`${ip}::${ua}`);
}

// ---------- Slot logic (track/day) ----------
function detectTrackIdFromText(q = "") {
  const s = normQ(q);
  if (s.includes("mile end") || s.includes("mileend") || s.includes("london")) return "mile_end";
  if (s.includes("gillingham") || s.includes("kent")) return "gillingham";
  return null;
}

function extractDayFromText(q = "") {
  const s = normQ(q);
  const days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"];
  const hit = days.find((d) => s.includes(d));
  return hit ? (hit.charAt(0).toUpperCase() + hit.slice(1)) : null;
}

// Only ask track/day/time when it actually matters (eligibility/booking/payment).
function questionNeedsEligibility(q = "") {
  const s = normQ(q);

  // These topics DO need track/day because rules differ (Mile End ticket days, booking requests, deposits).
  const eligibilityTopics = [
    "use my tickets", "use tickets", "valid", "eligib", "can i use",
    "book", "booking", "reserve", "availability", "slots",
    "deposit", "pay", "payment", "48 hours", "whatsapp",
    "mile end", "london track",
  ];

  // These do NOT need track/day/time (avoid rigid behaviour)
  const generalTopics = [
    "what's included", "whats included", "equipment", "helmet", "balaclava",
    "how many laps", "laps per session", "session length", "duration",
    "f1 simulator", "simulator", "refund", "expiry", "gift", "refer", "referral",
    "tickets left", "redeemed", "pin", "dashboard",
    "where", "location", "located", "postcode", "address", "tracks", "venues", "locations"
  ];

  if (hasAny(s, generalTopics)) return false;
  if (hasAny(s, eligibilityTopics)) return true;
  return false;
}

// ---------- Fast routes (return HTML instantly) ----------
function buildTracksHtml(kb) {
  const tracks = kb?.site?.tracks || [];
  if (!Array.isArray(tracks) || !tracks.length) return null;

  const names = tracks.map((t) => `${t.name} (${t.region})`).join(" and ");
  return `We currently run tracks in <strong>${names}</strong>.`;
}

function fastRouteReply(query, kb, sess) {
  const q = normQ(query);
  if (!kb || !sess) return null;

  // Update slots from this message
  const tid = detectTrackIdFromText(query);
  if (tid) sess.slots.trackId = tid;

  const day = extractDayFromText(query);
  if (day) sess.slots.day = day;

  // 1) Track locations / "where is the track" → ALWAYS list both tracks (never default to Gillingham)
  if (hasAny(q, ["where", "location", "located", "postcode", "address", "tracks", "venues", "locations"])) {
    const tracksHtml = buildTracksHtml(kb);
    if (tracksHtml) return `${tracksHtml}<br>Which one are you looking at?`;
  }

  // 2) Customer Dashboard / Ticket Checker
  if (
    hasAny(q, [
      "dashboard", "pin", "gift", "gifting", "referral", "credits",
      "tickets left", "redeemed", "tracking code", "my tickets",
      "check tickets", "ticket balance", "ticket checker", "custdash"
    ])
  ) {
    return `Manage tickets, gifting and referrals in the 
<a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/cdashlogin.php">Customer Dashboard</a>.<br>
For a quick check only, use the 
<a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/custdash.php">Ticket Checker</a>.`;
  }

  // 3) Mile End explicit
  if (hasAny(q, ["mile end", "mileend", "london track"])) {
    const be = kb.site?.urls?.book_experience || "https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php";
    return `Mile End is our <strong>London</strong> outdoor electric track.<br>
• <strong>8 laps</strong> per session on a <strong>450 m</strong> floodlit track<br>
• Minimum height: <strong>155 cm</strong><br>
• Tickets valid: <strong>Mon / Tue / Wed / Sun</strong> (pre-book required)<br>
<a href="${be}">Book Experience</a>`;
  }

  return null;
}

// ---------- KB → docs for embeddings ----------
function flattenDocsFromKB(kb) {
  const d = [];

  // General
  d.push({
    id: "opening",
    text: `${kb.opening?.days || ""} Hours: ${kb.opening?.hours || ""}`.trim(),
    url: kb.site?.urls?.home,
  });

  // Multi-track
  const tracks = kb.site?.tracks && Array.isArray(kb.site.tracks) ? kb.site.tracks : [];
  if (tracks.length) {
    // High-signal overview chunk for "where are your tracks?"
    d.push({
      id: "tracks_overview",
      text:
        `Karting Central tracks: ` +
        tracks
          .map((t) => `${t.name} (${t.region}) — ${t.indoor ? "indoor" : "outdoor"} ${t.type || "karting"}`)
          .join("; ") +
        `. If the user doesn’t specify a track/day, ask which track (and day if Mile End).`,
      url: kb.site?.urls?.home,
    });

    // Per-track chunks
    for (const t of tracks) {
      const karts = t.karts || {};
      const req = t.requirements || {};
      const trk = t.track || {};
      const sess = t.sessions || {};
      const days = Array.isArray(t.ticket_days_allowed) ? t.ticket_days_allowed.join(", ") : null;
      const feats = Array.isArray(t.features) ? t.features.join("; ") : null;

      d.push({
        id: `track_${t.id}`,
        text:
          `${t.name} (${t.region}) — ${t.indoor ? "indoor" : "outdoor"} track. ` +
          (trk.length_m ? `Track length ${trk.length_m} m. ` : "") +
          (trk.floodlit ? `Floodlit for evening racing. ` : "") +
          (trk.open_until ? `Open until ${trk.open_until}. ` : "") +
          (karts.type ? `Karts: ${karts.type}. ` : "") +
          (karts.top_speed_mph ? `Top speed up to ${karts.top_speed_mph} mph. ` : "") +
          (karts.max_on_track ? `Max ${karts.max_on_track} karts on track. ` : "") +
          (sess.laps_per_session ? `Each session is ${sess.laps_per_session} laps. ` : "") +
          (sess.three_sessions_total_laps ? `3 sessions = ${sess.three_sessions_total_laps} laps total. ` : "") +
          (req.min_height_cm ? `Minimum height ${req.min_height_cm} cm. ` : "") +
          (t.must_prebook ? `Must pre-book. ` : "") +
          (days ? `Karting Central tickets valid on: ${days}. ` : "") +
          (feats ? `Features: ${feats}.` : ""),
        url: kb.site?.urls?.home,
      });

      if (t.id === "mile_end") {
        d.push({
          id: "mile_end_ticket_rules",
          text:
            `Mile End (London) ticket rules: minimum height 155 cm. ` +
            `Tickets can be used Monday, Tuesday, Wednesday, and Sunday only; must pre-book.`,
          url: kb.site?.urls?.book_experience || kb.site?.urls?.home,
        });
        d.push({
          id: "mile_end_sessions",
          text:
            `Mile End sessions: 1 session is always 8 laps on a 450 m floodlit outdoor track. ` +
            `3 sessions = 24 laps total; hour booking window with roughly 30 minutes of track time.`,
          url: kb.site?.urls?.book_experience || kb.site?.urls?.home,
        });
      }

      if (t.id === "gillingham" && t.extras?.f1_simulator) {
        d.push({
          id: "gillingham_f1_sim",
          text:
            `Gillingham has an F1 simulator. With Karting Central tickets it is 50% off at £5 per simulator session; simulator sessions are 10 minutes.`,
          url: kb.site?.urls?.home,
        });
      }
    }
  }

  // Equipment
  if (kb.equipment?.included?.length) {
    d.push({
      id: "equipment",
      text: `Included: ${kb.equipment.included.join(", ")}.`,
      url: kb.site?.urls?.safety,
    });
  }

  // Deals (keep)
  d.push({
    id: "deals",
    text: `Session 2 & 3 discounted. Session 3 tiers: 1–4 £12; 5–8 £11; 9–12 £10.`,
    url: kb.site?.urls?.book_tickets,
  });

  d.push({
    id: "promos",
    text: `Promotions do not apply on Saturdays.`,
    url: kb.site?.urls?.terms,
  });

  // Hint chunks from fast_answers (optional)
  for (const f of kb.fast_answers || []) {
    d.push({
      id: `hint_${f.intent}`,
      text: `Hint: ${f.answer}`,
      url: kb.site?.urls?.home,
    });
  }

  return d;
}

async function rebuildEmbeddings() {
  if (!KB) return;
  DOCS = flattenDocsFromKB(KB);
  const resp = await openai.embeddings.create({
    model: EMB_MODEL,
    input: DOCS.map((d) => d.text),
  });
  VECTORS = resp.data.map((e, i) => ({
    id: DOCS[i].id,
    embedding: e.embedding,
    meta: DOCS[i],
  }));
  console.log(`Embedded ${VECTORS.length} KB chunks.`);
}

async function retrieve(query, k = 5) {
  if (!VECTORS.length) return [];
  const q = await openai.embeddings.create({ model: EMB_MODEL, input: query });
  const qEmb = q.data[0].embedding;
  return VECTORS
    .map((v) => ({ v, score: cosine(qEmb, v.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// ---------- KB fetchers ----------
async function fetchKB(force = false) {
  if (!KB_URL) throw new Error("KB_URL not set");
  const headers = { "Cache-Control": "no-cache", Pragma: "no-cache" };
  if (!force) {
    if (etag) headers["If-None-Match"] = etag;
    if (lastModified) headers["If-Modified-Since"] = lastModified;
  }
  const res = await fetch(KB_URL, { headers });
  if (res.status === 304) return false;
  if (!res.ok) throw new Error(`KB fetch failed: ${res.status}`);
  const json = await res.json();
  KB = json;
  etag = res.headers.get("etag") || etag;
  lastModified = res.headers.get("last-modified") || lastModified;
  console.log(`KB loaded (${etag || "no-etag"})`);
  await rebuildEmbeddings();
  return true;
}

async function fetchPrompt(force = false) {
  if (!PROMPT_URL) return false;
  const headers = { "Cache-Control": "no-cache", Pragma: "no-cache" };
  if (!force) {
    if (promptEtag) headers["If-None-Match"] = promptEtag;
    if (promptLastMod) headers["If-Modified-Since"] = promptLastMod;
  }
  const res = await fetch(PROMPT_URL, { headers });
  if (res.status === 304) return false;
  if (!res.ok) throw new Error(`Prompt fetch failed: ${res.status}`);
  PROMPT_TEXT = await res.text();
  promptEtag = res.headers.get("etag") || promptEtag;
  promptLastMod = res.headers.get("last-modified") || promptLastMod;
  console.log(`Prompt loaded from ${PROMPT_URL}`);
  return true;
}

function getSystemPrompt() {
  // .md is the source of truth
  if (PROMPT_TEXT && PROMPT_TEXT.trim()) return PROMPT_TEXT;

  // fallback
  return `
You are the Karting Central website assistant.
- Always use UK English and GBP (£).
- Be concise and helpful.
- Use provided context; never invent prices/hours/policies.
- Output HTML only with <a> links (no bare URLs).
`;
}

// ---------- Promo lead capture ----------
const LEADS_ROOT = path.join(process.cwd(), "leads");
function ensureDir(p) {
  if (!existsSync(p)) mkdirSync(p, { recursive: true });
}
function ymdParts(d = new Date()) {
  const y = String(d.getFullYear());
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return { y, m, day };
}

app.post("/api/promo-lead", async (req, res) => {
  try {
    const { name, email, phone, ts, source } = req.body || {};
    const clean = (s) => String(s || "").trim();

    const nameOk = clean(name).length > 0;
    const emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(clean(email));
    const phoneOk = /^[0-9\s+\-()]{7,}$/.test(clean(phone));
    if (!nameOk || !emailOk || !phoneOk) {
      return res.status(400).json({ ok: false, error: "Invalid name/email/phone." });
    }

    const when = ts && !isNaN(Date.parse(ts)) ? new Date(ts) : new Date();
    const { y, m, day } = ymdParts(when);

    const dirMonth = path.join(LEADS_ROOT, `${y}-${m}`);
    ensureDir(dirMonth);

    const jsonlPath = path.join(dirMonth, `${y}-${m}-${day}.jsonl`);
    const csvPath = path.join(dirMonth, `${y}-${m}-${day}.csv`);

    const record = {
      name: clean(name),
      email: clean(email),
      phone: clean(phone),
      ts: when.toISOString(),
      source: clean(source) || "unknown",
      ip: req.headers["x-forwarded-for"] || req.socket.remoteAddress || "",
      ua: req.headers["user-agent"] || "",
    };

    await fs.appendFile(jsonlPath, JSON.stringify(record) + "\n", "utf8");

    const csvLine =
      [
        record.ts,
        `"${record.name.replace(/"/g, '""')}"`,
        record.email,
        record.phone,
        record.source,
        `"${String(record.ua).replace(/"/g, '""')}"`,
        `"${String(record.ip).replace(/"/g, '""')}"`,
      ].join(",") + "\n";

    if (!existsSync(csvPath)) {
      await fs.appendFile(csvPath, "ts,name,email,phone,source,ua,ip\n", "utf8");
    }
    await fs.appendFile(csvPath, csvLine, "utf8");

    res.json({ ok: true });
  } catch (e) {
    console.error("[promo-lead] error:", e);
    res.status(500).json({ ok: false, error: "Failed to save lead." });
  }
});

// ---------- Chat route (conversation + retrieval) ----------
app.post("/api/faq-response", async (req, res) => {
  const t0 = Date.now();
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });
    if (!KB) return res.status(503).json({ error: "KB not loaded yet" });

    // sessionId optional; if missing we derive one and return it
    const sessionId = String(req.body?.sessionId || "") || inferSessionId(req);
    const sess = upsertSess(sessionId);

    // FAST ROUTES (also updates slots)
    const fast = fastRouteReply(query, KB, sess);
    if (fast) {
      const botHtml = enforceGBP(fast);
      pushTurn(sess, "user", String(query));
      pushTurn(sess, "assistant", stripHtmlToText(botHtml));
      return res.json({ response: botHtml, sources: [], sessionId });
    }

    // Only ask track/day when it actually matters:
    // - ask track for eligibility/booking requests
    // - ask day ONLY for Mile End when needed
    const needsElig = questionNeedsEligibility(query);
    if (needsElig) {
      const trackKnown = !!sess?.slots?.trackId;
      const dayKnown = !!sess?.slots?.day;

      if (!trackKnown) {
        const botHtml =
          `Which track are you looking at — <strong>Gillingham</strong> or <strong>Mile End (London)</strong>?<br>` +
          `<a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php">Book Experience</a>`;
        pushTurn(sess, "user", String(query));
        pushTurn(sess, "assistant", stripHtmlToText(botHtml));
        return res.json({ response: botHtml, sources: [], sessionId });
      }

      if (sess.slots.trackId === "mile_end" && !dayKnown) {
        const botHtml =
          `What day are you looking to race at <strong>Mile End</strong>?<br>` +
          `Tickets are valid <strong>Mon / Tue / Wed / Sun</strong> (pre-book required).<br>` +
          `<a href="https://pos.kartingcentral.co.uk/home/download/pos2/pos2/book_experience.php">Book Experience</a>`;
        pushTurn(sess, "user", String(query));
        pushTurn(sess, "assistant", stripHtmlToText(botHtml));
        return res.json({ response: botHtml, sources: [], sessionId });
      }
    }

    // Conversation-aware retrieval query
    const convoText = (sess?.turns || [])
      .slice(-8)
      .map((t) => `${t.role === "user" ? "User" : "Assistant"}: ${t.content}`)
      .join("\n");

    const retrieved = await retrieve(`${convoText}\nUser: ${query}`, 5);

    const contextBlock = retrieved
      .map(
        (r, i) =>
          `#${i + 1} [${r.v.id}] ${r.v.meta.text}${r.v.meta.url ? ` (URL: ${r.v.meta.url})` : ""}`
      )
      .join("\n\n");

    const historyMsgs = (sess?.turns || []).map((t) => ({
      role: t.role,
      content: t.content,
    }));

    const messages = [
      { role: "system", content: getSystemPrompt() }, // .md is the prompt
      ...historyMsgs,
      {
        role: "user",
        content: `User question: ${query}

Reference notes (optional factual snippets):
${contextBlock || "(none)"}

Rules:
- Continue the conversation using prior messages (do not treat each message as new).
- Do NOT repeatedly ask for track/day/time unless eligibility/booking requires it.
- If user asks where/locations/tracks, list BOTH: Gillingham (Kent) and Mile End (London).
- Output valid HTML only. Use <a> links. Do not output Markdown. Do not invent policies/prices/hours.`,
      },
    ];

    const completion = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages,
      temperature: 0.4,
      presence_penalty: 0.2,
      frequency_penalty: 0.2,
    });

    const text =
      enforceGBP(completion.choices?.[0]?.message?.content?.trim()) ||
      "Sorry, I couldn’t generate a reply.";

    pushTurn(sess, "user", String(query));
    pushTurn(sess, "assistant", stripHtmlToText(text));

    const sources = retrieved.map((r) => ({ id: r.v.id, url: r.v.meta.url }));
    res.json({ response: text, sources, sessionId });

    console.log("[chat] done in", Date.now() - t0, "ms");
  } catch (e) {
    console.error("[chat] error:", e);
    res.status(500).json({ error: "Server error" });
  }
});

// ---------- Status & reload ----------
app.get("/healthz", (_req, res) => res.json({ ok: true, ts: Date.now() }));
app.get("/kb-status", (_req, res) => res.json({ loaded: !!KB, chunks: VECTORS.length }));
app.get("/prompt-status", (_req, res) => res.json({ hasPrompt: !!(PROMPT_TEXT && PROMPT_TEXT.trim()) }));

app.post("/kb-reload", async (_req, res) => {
  try {
    const changed = await fetchKB(true);
    res.json({ ok: true, changed, loaded: !!KB, chunks: VECTORS.length });
  } catch (e) {
    console.error("KB reload error:", e);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post("/prompt-reload", async (_req, res) => {
  try {
    const changed = await fetchPrompt(true);
    res.json({ ok: true, changed, hasPrompt: !!PROMPT_TEXT });
  } catch (e) {
    console.error("Prompt reload error:", e);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get("/debug-config", (_req, res) => {
  res.json({
    ok: true,
    KB_URL: process.env.KB_URL || null,
    PROMPT_URL: process.env.PROMPT_URL || null,
    promptHash: sha1(PROMPT_TEXT || ""),
    promptPreview: (PROMPT_TEXT || "").slice(0, 220),
    kbLoaded: !!KB,
    kbChunks: VECTORS.length,
    kbEtag: etag,
    kbLastModified: lastModified,
    promptEtag,
    promptLastMod,
    sessions: SESS.size,
  });
});

// ---------- Warmup ----------
async function warmOpenAI() {
  try {
    await openai.embeddings.create({ model: EMB_MODEL, input: "warmup" });
  } catch (e) {
    console.warn("[warmup] embeddings:", e.message);
  }
  try {
    await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [{ role: "system", content: "warmup" }, { role: "user", content: "ping" }],
      max_tokens: 8,
      temperature: 0,
    });
  } catch (e) {
    console.warn("[warmup] chat:", e.message);
  }
}

async function warmup() {
  try {
    await fetchPrompt(true);
  } catch (e) {
    console.warn("[warmup] prompt:", e.message);
  }

  try {
    await fetchKB(true);
  } catch (e) {
    console.warn("[warmup] kb:", e.message);
  }

  await warmOpenAI();
}

// ---------- Boot ----------
const port = process.env.PORT || 10000;

app.listen(port, async () => {
  console.log(`Listening on ${port}`);

  await warmup();

  setInterval(() => {
    // clean old sessions
    const now = Date.now();
    for (const [k, v] of SESS.entries()) {
      if (now - (v?.updatedAt || 0) > SESS_TTL_MS) SESS.delete(k);
    }
  }, 5 * 60 * 1000);

  setInterval(() => fetchKB(false).catch((e) => console.warn("KB refresh failed:", e.message)), REFRESH_MS);
  setInterval(() => fetchPrompt(false).catch((e) => console.warn("Prompt refresh failed:", e.message)), REFRESH_MS);

  // Self-ping keeps the node process warm, but Render free tier can still sleep without external ping.
  setInterval(() => {
    fetch(`http://127.0.0.1:${port}/healthz`).catch(() => {});
  }, 60 * 1000);
});
