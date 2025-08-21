// server.js — GPT + remote KB (single file)
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
const KB_URL = process.env.KB_URL;         // e.g. https://pos.kartingcentral.co.uk/home/download/pos2/pos2/knowledge.kartingcentral.json
const REFRESH_MS = 5 * 60 * 1000;          // refresh every 5 min
let KB = null;
let DOCS = [];                              // [{ id, text, url }]
let VECTORS = [];                           // [{ id, embedding, meta }]
let etag = null, lastModified = null;

// ---------- Utils ----------
const enforceGBP = (t) => (t || "").replace(/\$/g, "£");
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i], y = b[i]; dot += x * y; na += x * x; nb += y * y; }
  return dot / ((Math.sqrt(na) * Math.sqrt(nb)) || 1);
}

// Flatten KB into chunks
function flattenDocsFromKB(kb) {
  const d = [];
  d.push({ id: "opening_overview", text: `${kb.opening.days} Hours: ${kb.opening.hours}`, url: kb.site.urls.home });
  d.push({ id: "track_and_karts", text: `Indoor: ${kb.track_and_karts.indoor}. Kart: ${kb.track_and_karts.kart_type}. Top speed up to ${kb.track_and_karts.top_speed_mph} mph. Max ${kb.track_and_karts.max_karts_on_track} karts on track.`, url: kb.site.urls.home });
  d.push({ id: "requirements", text: `Adult min height: ${kb.requirements.adult_min_height_cm} cm (5ft). Junior min height: ${kb.requirements.junior_min_height_cm} cm. Shoes count toward height: ${kb.requirements.shoes_count_towards_height ? "yes" : "no"}.`, url: kb.site.urls.safety });
  d.push({ id: "equipment", text: `Included: ${kb.equipment.included.join(", ")}.`, url: kb.site.urls.safety });
  d.push({ id: "sessions", text: `Per ticket: ${kb.sessions.per_ticket_includes}. Up to ${kb.sessions.laps_per_session_up_to} laps per session; typical 3-session total: ${kb.sessions.typical_three_session_total_laps}.`, url: kb.site.urls.book_tickets });
  d.push({ id: "deals", text: `Session 2 & 3 discounted (pre-book by phone). Session 3 tiers: 1–4 £12; 5–8 £11; 9–12 £10.`, url: kb.site.urls.book_tickets });
  d.push({ id: "promos", text: `Promotions do not apply on Saturdays.`, url: kb.site.urls.terms });
  d.push({ id: "f1_sim", text: `F1 simulator at Gillingham. 50% off with Karting Central tickets.`, url: kb.site.urls.home });
  d.push({ id: "tracking", text: `Tracking codes for managing tickets. Gifting available via Customer Dashboard.`, url: kb.site.urls.customer_dashboard });
  // make fast_answers searchable hints (not final answers)
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

// optional, soft keyword intents used only to bias context
function matchIntent(q) {
  if (!KB?.fast_answers) return null;
  const text = q.toLowerCase();
  for (const f of KB.fast_answers) {
    if (f.keywords?.some(k => text.includes(k.toLowerCase()))) return f;
  }
  return null;
}

// Load KB from remote URL with ETag/Last-Modified
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
  if (!json.site || !json.opening) throw new Error("KB missing required sections");
  KB = json;
  etag = res.headers.get("etag") || etag;
  lastModified = res.headers.get("last-modified") || lastModified;
  console.log(`KB loaded from ${KB_URL} (${etag || "no-etag"})`);
  await rebuildEmbeddings();
  return true;
}
setInterval(() => fetchKB(false).catch(e => console.warn("KB refresh failed:", e.message)), REFRESH_MS);

// ---------- System prompt ----------
const SYSTEM_PROMPT = `
You are the Karting Central website assistant.
- Always use UK English and GBP (£).
- You can handle greetings and small talk naturally.
- Use provided context for facts/links; if none is relevant, answer generally but do not invent specific prices/hours/policies.
- Be concise and friendly. Start with the direct answer, then 1–2 helpful bullets if needed.
- Add a clear CTA with the correct path (Book Experience, Customer Dashboard, Safety, Events) only when relevant.
- If a user mentions existing tickets or tracking codes, prefer the Customer Dashboard link from context.
`;

// ---------- Chat route ----------
app.post("/api/faq-response", async (req, res) => {
  const t0 = Date.now();
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });
    if (!KB)   return res.status(503).json({ error: "KB not loaded yet" });

    // Use soft intent as context only
    const intent = matchIntent(query);
    let intentSnippet = null;
    if (intent) {
      intentSnippet = { id: `intent_${intent.intent}`, text: intent.answer, url: KB.site.urls.home };
      console.log("[chat] intent:", intent.intent);
    }

    // Retrieve KB
    const retrieved = await retrieve(query, 5);
    console.log("[chat] retrieved:", retrieved.map(r => `${r.v.id}:${r.score.toFixed(3)}`).join(", "));

    // Build context (intent first if present)
    const contextItems = [...retrieved.map(r => r.v.meta)];
    if (intentSnippet) contextItems.unshift(intentSnippet);

    const contextBlock = contextItems
      .map((m, i) => `#${i + 1} [${m.id || "kb"}] ${m.text}${m.url ? ` (URL: ${m.url})` : ""}`)
      .join("\n\n");

    // Always ask the model (no canned fallback)
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      {
        role: "user",
        content: `User question: ${query}

Here are optional reference notes (use them if relevant; otherwise answer from general knowledge, but never invent specific prices/hours/policies):
${contextBlock || "(none)"}

When helpful, add 1–2 bullets and a clear CTA with the correct path/link.`
      }
    ];

    const completion = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages,
      temperature: 0.5,
      presence_penalty: 0.2,
      frequency_penalty: 0.2,
    });

    console.log("[chat] openai ok in", Date.now() - t0, "ms");
    const text = enforceGBP(completion.choices?.[0]?.message?.content?.trim()) ||
                 "Sorry, I couldn’t generate a reply.";
    const sources = contextItems.map((m, i) => ({ id: m.id || `kb_${i + 1}`, url: m.url || null }));
    return res.json({ response: text, sources });
  } catch (e) {
    console.error("[chat] error:", e);
    return res.status(500).json({ error: "Server error" });
  }
});

// ---------- Health + status ----------
app.get("/healthz", (_, res) => res.status(200).send("ok"));
app.get("/kb-status", (_, res) =>
  res.json({ loaded: !!KB, chunks: VECTORS.length, etag: etag || null, lastModified: lastModified || null })
);

// ---------- Boot ----------
const port = process.env.PORT || 10000;
app.listen(port, async () => {
  console.log(`Listening on ${port}`);
  try {
    await fetchKB(true);  // initial load
  } catch (e) {
    console.error("KB initial load failed:", e);
  }
});
