// server.js — GPT + remote KB with external instructions
import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
app.use(express.json());
app.use(
  cors({
    origin: [
      "https://pos.kartingcentral.co.uk",
      "https://www.kartingcentral.co.uk",
      "http://localhost:3000",
    ],
    methods: ["POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// ---------- OpenAI ----------
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const CHAT_MODEL = "gpt-4o-mini";
const EMB_MODEL = "text-embedding-3-small";

// ---------- Remote KB ----------
const KB_URL = process.env.KB_URL;
const PROMPT_URL = process.env.PROMPT_URL; // new: external Markdown file
const REFRESH_MS = 5 * 60 * 1000;          // 5 minutes

let KB = null;
let DOCS = [];
let VECTORS = [];
let etag = null, lastModified = null;

let PROMPT_TEXT = null;
let promptEtag = null, promptLastMod = null;

// ---------- Utils ----------
const enforceGBP = (t) => (t || "").replace(/\$/g, "£");
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / ((Math.sqrt(na) * Math.sqrt(nb)) || 1);
}



// ---------- Flatten KB ----------
function flattenDocsFromKB(kb) {
  const d = [];
  d.push({ id: "opening", text: `${kb.opening.days} Hours: ${kb.opening.hours}`, url: kb.site.urls.home });
  d.push({ id: "track", text: `Indoor: ${kb.track_and_karts.indoor}. Kart: ${kb.track_and_karts.kart_type}. Top speed up to ${kb.track_and_karts.top_speed_mph} mph. Max ${kb.track_and_karts.max_karts_on_track} karts.`, url: kb.site.urls.home });
  d.push({ id: "requirements", text: `Adult min height: ${kb.requirements.adult_min_height_cm} cm (5ft). Junior min height: ${kb.requirements.junior_min_height_cm} cm. Shoes count toward height: ${kb.requirements.shoes_count_towards_height ? "yes" : "no"}.`, url: kb.site.urls.safety });
  d.push({ id: "equipment", text: `Included: ${kb.equipment.included.join(", ")}.`, url: kb.site.urls.safety });
  d.push({ id: "sessions", text: `Per ticket: ${kb.sessions.per_ticket_includes}. Up to ${kb.sessions.laps_per_session_up_to} laps/session; typical 3-session total: ${kb.sessions.typical_three_session_total_laps}.`, url: kb.site.urls.book_tickets });
  d.push({ id: "deals", text: `Session 2 & 3 discounted (pre-book by phone). Session 3 tiers: 1–4 £12; 5–8 £11; 9–12 £10.`, url: kb.site.urls.book_tickets });
  d.push({ id: "promos", text: `Promotions do not apply on Saturdays.`, url: kb.site.urls.terms });
  d.push({ id: "f1", text: `F1 simulator at Gillingham. 50% off with Karting Central tickets.`, url: kb.site.urls.home });
  d.push({ id: "tracking", text: `Tracking codes for managing tickets. Gifting available via Customer Dashboard.`, url: kb.site.urls.customer_dashboard });
  for (const f of kb.fast_answers || []) {
    d.push({ id: `hint_${f.intent}`, text: `Hint: ${f.answer}`, url: kb.site.urls.home });
  }
  return d;
}

async function rebuildEmbeddings() {
  DOCS = flattenDocsFromKB(KB);
  const resp = await openai.embeddings.create({ model: EMB_MODEL, input: DOCS.map(d => d.text) });
  VECTORS = resp.data.map((e, i) => ({ id: DOCS[i].id, embedding: e.embedding, meta: DOCS[i] }));
  console.log(`Embedded ${VECTORS.length} KB chunks.`);
}

async function retrieve(query, k = 5) {
  const q = await openai.embeddings.create({ model: EMB_MODEL, input: query });
  const qEmb = q.data[0].embedding;
  return VECTORS
    .map(v => ({ v, score: cosine(qEmb, v.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// ---------- KB fetch ----------
async function fetchKB(force = false) {
  if (!KB_URL) throw new Error("KB_URL not set");
  const headers = {};
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

// ---------- Prompt fetch ----------
async function fetchPrompt(force = false) {
  if (!PROMPT_URL) return false;
  const headers = {};
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
  if (PROMPT_TEXT && PROMPT_TEXT.trim()) return PROMPT_TEXT;
  return `
You are the Karting Central website assistant.
- Always use UK English and GBP (£).
- Handle greetings and small talk naturally.
- Use provided context for facts/links; if none is relevant, answer generally but never invent specific prices/hours/policies.
- Be concise and friendly. Start with the direct answer, then 1–2 helpful bullets if needed.
- Add a clear CTA with the correct path (Book Experience, Customer Dashboard, Safety, Events) only when relevant.
- If a user mentions tickets or tracking codes, point to the Customer Dashboard link.
`;
}

// ---------- Routes ----------
app.post("/api/faq-response", async (req, res) => {
  const t0 = Date.now();
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });
    if (!KB) return res.status(503).json({ error: "KB not loaded yet" });

    const retrieved = await retrieve(query, 5);
    const contextBlock = retrieved
      .map((r, i) => `#${i + 1} [${r.v.id}] ${r.v.meta.text}${r.v.meta.url ? ` (URL: ${r.v.meta.url})` : ""}`)
      .join("\n\n");

    const messages = [
      { role: "system", content: getSystemPrompt() },
      {
        role: "user",
        content: `User question: ${query}

Here are optional reference notes:
${contextBlock || "(none)"}

Use them if helpful; otherwise answer generally. Never invent prices/hours/policies.`
      }
    ];

    const completion = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages,
      temperature: 0.5,
      presence_penalty: 0.2,
      frequency_penalty: 0.2,
    });

    const raw = completion.choices?.[0]?.message?.content?.trim() || "";
let text = enforceGBP(raw);
text = unwrapCodeFence(text);        // remove ```html fences if present
text = injectSessionExplainer(text); // ensure clickable expander for “one full session”
    const sources = retrieved.map(r => ({ id: r.v.id, url: r.v.meta.url }));
    res.json({ response: text, sources });
    console.log("[chat] done in", Date.now() - t0, "ms");
  } catch (e) {
    console.error("[chat] error:", e);
    res.status(500).json({ error: "Server error" });
  }
});

app.get("/healthz", (_, res) => res.send("ok"));
app.get("/kb-status", (_, res) => res.json({ loaded: !!KB, chunks: VECTORS.length }));
app.get("/prompt-status", (_, res) => res.json({ hasPrompt: !!PROMPT_TEXT }));

// ---------- Boot ----------
const port = process.env.PORT || 10000;
app.listen(port, async () => {
  console.log(`Listening on ${port}`);
  try {
    await fetchKB(true);
    await fetchPrompt(true);
    setInterval(() => fetchKB(false).catch(e => console.warn("KB refresh failed:", e.message)), REFRESH_MS);
    setInterval(() => fetchPrompt(false).catch(e => console.warn("Prompt refresh failed:", e.message)), REFRESH_MS);
  } catch (e) {
    console.error("Initial load failed:", e);
  }
});
