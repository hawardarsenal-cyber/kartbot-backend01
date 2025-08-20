1)// server.js — remote KB version
import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
app.use(express.json());
app.use(cors({
  origin: [
    "https://pos.kartingcentral.co.uk",
    "https://www.kartingcentral.co.uk",
    "http://localhost:3000"   // keep for local testing if you like
  ],
  methods: ["POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));


const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ===== Remote KB settings =====
const KB_URL = process.env.KB_URL;               // e.g. https://pos.kartingcentral.co.uk/home/download/pos2/pos2/knowledge.kartingcentral.json
const ADMIN_TOKEN = process.env.ADMIN_TOKEN;     // any strong secret to protect /admin/kb-reload
const REFRESH_MS = 5 * 60 * 1000;                // 5 min

let KB = null;                                   // current knowledge object
let DOCS = [];                                   // flattened chunks
let VECTORS = [];                                // [{id, embedding, meta}]
let etag = null, lastModified = null;            // for conditional GET
const EMB_MODEL = "text-embedding-3-small";

// ===== Utils =====
function enforceGBP(text) { return (text || "").replace(/\$/g, "£"); }
function cosine(a,b){ let dot=0,na=0,nb=0; for(let i=0;i<a.length;i++){const x=a[i],y=b[i]; dot+=x*y; na+=x*x; nb+=y*y;} const d=Math.sqrt(na)*Math.sqrt(nb)||1; return dot/d; }

// ===== KB flattening =====
function flattenDocsFromKB(kb) {
  const docs = [];
  docs.push({ id:"opening_overview", text:`${kb.opening.days} Hours: ${kb.opening.hours}`, url: kb.site.urls.home });
  docs.push({ id:"track_and_karts", text:`Indoor: ${kb.track_and_karts.indoor}. Kart type: ${kb.track_and_karts.kart_type}. Top speed up to ${kb.track_and_karts.top_speed_mph} mph. Max ${kb.track_and_karts.max_karts_on_track} karts on track.`, url: kb.site.urls.home });
  docs.push({ id:"requirements", text:`Adult min height: ${kb.requirements.adult_min_height_cm} cm (5ft). Junior min height: ${kb.requirements.junior_min_height_cm} cm. Shoes count toward height: ${kb.requirements.shoes_count_towards_height ? "yes" : "no"}.`, url: kb.site.urls.safety });
  docs.push({ id:"equipment", text:`Included equipment: ${kb.equipment.included.join(", ")}.`, url: kb.site.urls.safety });
  docs.push({ id:"sessions", text:`Per ticket: ${kb.sessions.per_ticket_includes}. Up to ${kb.sessions.laps_per_session_up_to} laps per session; typical 3-session total: ${kb.sessions.typical_three_session_total_laps}.`, url: kb.site.urls.book_tickets });
  docs.push({ id:"multi_session_deals", text:`Session 2 & 3 discounted (pre-book by phone). Session 3 tiers: 1–4 £12; 5–8 £11; 9–12 £10.`, url: kb.site.urls.book_tickets });
  docs.push({ id:"promotions", text:`Promotional discounts do not apply on Saturdays.`, url: kb.site.urls.terms });
  docs.push({ id:"f1_sim", text:`F1 simulator at Gillingham. 50% off with Karting Central tickets.`, url: kb.site.urls.home });
  docs.push({ id:"tracking_and_tickets", text:`Tracking codes for managing tickets. Gifting available via Customer Dashboard.`, url: kb.site.urls.customer_dashboard });

  for (const f of kb.fast_answers) {
    docs.push({ id:`fast_${f.intent}`, text:`${f.intent}: ${f.answer}`, url: kb.site.urls.home });
  }
  return docs;
}

// ===== Embedding =====
async function rebuildEmbeddings() {
  DOCS = flattenDocsFromKB(KB);
  const resp = await openai.embeddings.create({ model: EMB_MODEL, input: DOCS.map(d => d.text) });
  VECTORS = resp.data.map((e, i) => ({ id: DOCS[i].id, embedding: e.embedding, meta: DOCS[i] }));
  console.log(`Embedded ${VECTORS.length} KB chunks.`);
}

async function retrieve(query, k=5) {
  const q = await openai.embeddings.create({ model: EMB_MODEL, input: query });
  const qEmb = q.data[0].embedding;
  return VECTORS
    .map(v => ({ v, score: cosine(qEmb, v.embedding) }))
    .sort((a,b) => b.score - a.score)
    .slice(0,k);
}

// ===== Fast intents =====
function matchIntent(q) {
  if (!KB?.fast_answers) return null;
  const text = q.toLowerCase();
  for (const f of KB.fast_answers) {
    if (f.keywords?.some(k => text.includes(k.toLowerCase()))) return f;
  }
  return null;
}

// ===== Remote KB loader with ETag/Last-Modified =====
async function fetchKB(force=false) {
  if (!KB_URL) throw new Error("KB_URL not set");
  const headers = {};
  if (!force) {
    if (etag) headers["If-None-Match"] = etag;
    if (lastModified) headers["If-Modified-Since"] = lastModified;
  }
  const res = await fetch(KB_URL, { headers });
  if (res.status === 304) { return false; } // unchanged
  if (!res.ok) throw new Error(`KB fetch failed: ${res.status}`);
  const json = await res.json();

  // basic sanity check
  if (!json.site || !json.opening || !json.fast_answers) {
    throw new Error("KB missing required sections (site/opening/fast_answers)");
  }

  KB = json;
  etag = res.headers.get("etag") || etag;
  lastModified = res.headers.get("last-modified") || lastModified;
  console.log(`KB loaded from ${KB_URL} (${etag || "no-etag"})`);
  await rebuildEmbeddings();
  return true;
}

function scheduleAutoRefresh() {
  setInterval(async () => {
    try {
      const changed = await fetchKB(false);
      if (changed) console.log("KB updated & embeddings rebuilt.");
    } catch (e) {
      console.warn("KB refresh failed:", e.message);
    }
  }, REFRESH_MS);
}

// ===== Prompt =====
const SYSTEM_PROMPT = `
You are the Karting Central website assistant.
- Always use UK English and GBP (£).
- Ground answers in the provided context; if context is missing, answer generally but stay relevant to indoor electric karting and our venue.
- Be concise and friendly. Start with the direct answer, then 1–2 bullets if helpful.
- Include a clear CTA with a correct site path when relevant (Book Experience, Customer Dashboard, Safety, Events).
- Do not invent prices, hours, or policies not present in context. If something is missing, say so and point to booking or the Customer Dashboard.
- If a user mentions existing tickets or tracking codes, prefer the Customer Dashboard link from context.
`;


// ===== Routes =====
app.post("/api/faq-response", async (req, res) => {
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });
    if (!KB)   return res.status(503).json({ error: "KB not loaded yet" });

   // 1) Intent -> use as context (do NOT return early)
const intent = matchIntent(query);
let intentSnippet = null;
if (intent) {
  intentSnippet = {
    id: `intent_${intent.intent}`,
    text: intent.answer,
    url: KB.site.urls.home
  };
}

// 2) Retrieval
const retrieved = await retrieve(query, 5);

// Build a context list (even if empty), and include the intent snippet if present
const contextItems = [...retrieved.map(r => r.v.meta)];
if (intentSnippet) contextItems.unshift(intentSnippet); // make it highest priority

const contextBlock = contextItems
  .map((m, i) => `#${i + 1} [${m.id || "kb"}] ${m.text}${m.url ? ` (URL: ${m.url})` : ""}`)
  .join("\n\n");

// 3) Ask the model every time (fluid answer; use context as guidance)
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
  model: "gpt-4o-mini",
  messages,
  temperature: 0.5,        // a touch more fluid
  presence_penalty: 0.2,
  frequency_penalty: 0.2
});



    const text = enforceGBP(completion.choices?.[0]?.message?.content?.trim()) ||
             "Sorry, I couldn’t find that in our info.";

const sources = contextItems.map((m, i) => ({
  id: m.id || `kb_${i + 1}`,
  url: m.url || null
}));

res.json({ response: text, sources });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

// Health/status
app.get("/healthz", (_, res) => res.send("ok"));
app.get("/kb-status", (_, res) => {
  res.json({
    loaded: !!KB,
    etag: etag || null,
    lastModified: lastModified || null,
    chunks: VECTORS.length || 0
  });
});

// Admin: force reload (Authorization: Bearer <ADMIN_TOKEN>)
app.post("/admin/kb-reload", async (req, res) => {
  try {
    const auth = req.headers.authorization || "";
    const token = auth.startsWith("Bearer ") ? auth.slice(7) : null;
    if (!ADMIN_TOKEN || token !== ADMIN_TOKEN) {
      return res.status(401).json({ error: "Unauthorised" });
    }
    const changed = await fetchKB(true);
    res.json({ ok: true, changed });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ===== Boot =====
const port = process.env.PORT || 10000;
app.listen(port, async () => {
  console.log(`Listening on ${port}`);
  try {
    await fetchKB(true);      // initial load (force)
    scheduleAutoRefresh();    // periodic refresh with ETag/Last-Modified
  } catch (e) {
    console.error("KB initial load failed:", e);
  }
});
