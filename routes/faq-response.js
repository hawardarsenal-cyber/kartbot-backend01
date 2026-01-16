// routes/faq-response.js
import fs from "fs";
import path from "path";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ===== Load knowledge =====
const KNOWLEDGE_PATH = path.join(process.cwd(), "knowledge.kartingcentral.json");
let KB = JSON.parse(fs.readFileSync(KNOWLEDGE_PATH, "utf-8"));

// ===== Build searchable docs from KB =====
const EMB_MODEL = "text-embedding-3-small";
let DOCS = [];    // [{ id, text, url }]
let VECTORS = []; // [{ id, embedding, meta }]

function flattenDocs() {
  const docs = [];

  // structured chunks
  docs.push({
    id: "opening_overview",
    text: `${KB.opening.days} Hours: ${KB.opening.hours}`,
    url: KB.site.urls.home
  });

  docs.push({
    id: "track_and_karts",
    text: `Indoor: ${KB.track_and_karts.indoor}. Kart type: ${KB.track_and_karts.kart_type}. Top speed up to ${KB.track_and_karts.top_speed_mph} mph. Max ${KB.track_and_karts.max_karts_on_track} karts on track.`,
    url: KB.site.urls.home
  });

  docs.push({
    id: "requirements",
    text: `Adult min height: ${KB.requirements.adult_min_height_cm} cm (5ft). Junior min height: ${KB.requirements.junior_min_height_cm} cm. Shoes count toward height: ${KB.requirements.shoes_count_towards_height ? "yes" : "no"}.`,
    url: KB.site.urls.safety
  });

  docs.push({
    id: "equipment",
    text: `Included equipment: ${KB.equipment.included.join(", ")}.`,
    url: KB.site.urls.safety
  });

  docs.push({
    id: "sessions",
    text: `Per ticket: ${KB.sessions.per_ticket_includes}. Up to ${KB.sessions.laps_per_session_up_to} laps per session; typical 3-session total: ${KB.sessions.typical_three_session_total_laps}.`,
    url: KB.site.urls.book_tickets
  });

  docs.push({
    id: "multi_session_deals",
    text: `Session 2 & 3 discounted (pre-book by phone). Session 3 tiers: 1–4 £12; 5–8 £11; 9–12 £10.`,
    url: KB.site.urls.book_tickets
  });

  docs.push({
    id: "promotions",
    text: `Promotions do not apply on Saturdays.`,
    url: KB.site.urls.terms
  });

  docs.push({
    id: "f1_sim",
    text: `F1 simulator at Gillingham. 50% off with tickets.`,
    url: KB.site.urls.home
  });

  docs.push({
    id: "tracking_and_tickets",
    text: `Tracking codes for managing tickets. Gifting available via Customer Dashboard.`,
    url: KB.site.urls.customer_dashboard
  });

  // include your fast answers so they’re retrievable too
  for (const f of KB.fast_answers) {
    docs.push({
      id: `fast_${f.intent}`,
      text: `${f.intent}: ${f.answer}`,
      url: KB.site.urls.home
    });
  }

  return docs;
}

// cosine similarity
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

// embed all docs (called once at boot)
async function embedAll() {
  DOCS = flattenDocs();
  const resp = await openai.embeddings.create({
    model: EMB_MODEL,
    input: DOCS.map(d => d.text)
  });
  VECTORS = resp.data.map((e, i) => ({
    id: DOCS[i].id,
    embedding: e.embedding,
    meta: DOCS[i]
  }));
  console.log(`Embedded ${VECTORS.length} knowledge chunks.`);
}

// retrieve top-k
async function retrieve(query, k = 5) {
  const q = await openai.embeddings.create({ model: EMB_MODEL, input: query });
  const qEmb = q.data[0].embedding;
  return VECTORS
    .map(v => ({ v, score: cosine(qEmb, v.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// quick keyword intent match (fast path)
function matchIntent(q) {
  const text = q.toLowerCase();
  for (const f of KB.fast_answers) {
    if (f.keywords.some(k => text.includes(k.toLowerCase()))) {
      return f;
    }
  }
  return null;
}

// belt & braces: ensure £
function enforceGBP(text) {
  return text.replace(/\$/g, "£");
}

const SYSTEM_PROMPT = `
You are the Karting Central website assistant.
- Always use UK English and GBP (£) in responses.
- Ground answers in the provided context. If context is missing, answer generally but keep it relevant to indoor electric karting and our venue.
- Be concise and friendly. Start with the direct answer, then 1–2 bullets if helpful, and add a clear CTA with a correct site path (Book Experience, Customer Dashboard, Safety, Events) when relevant.
- Do not invent prices, hours, or policies not present in context; if missing, say you'll check and point users to booking or the Customer Dashboard.
- If the user has existing tickets, prefer pointing to the Customer Dashboard link from context.
`;

export async function faqResponseHandler(req, res) {
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });

    // 1) fast intent
    const hit = matchIntent(query);
    if (hit) {
      return res.json({
        response: enforceGBP(hit.answer),
        sources: [{ id: hit.intent, url: KB.site.urls.home, score: 1 }]
      });
    }

    // 2) retrieve from KB
    const retrieved = await retrieve(query, 5);

const injected = [];
if (looksLikeTrackOverviewQuery(query)) {
  const t = buildTracksOverview(KB);
  if (t) injected.push(`#0 [tracks_overview_injected] ${t}`);
}

const contextBlock = [
  ...injected,
  ...retrieved.map(
    (r, i) =>
      `#${i + 1} [${r.v.id}] ${r.v.meta.text}${
        r.v.meta.url ? ` (URL: ${r.v.meta.url})` : ""
      }`
  ),
].join("\n\n");


    // 3) ask model
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      {
        role: "user",
        content: `Question: ${query}\n\nContext:\n${context}\n\nInstructions:\n- Prefer quoting context.\n- If unclear/missing, ask a brief follow-up and suggest the closest next step (Book, Dashboard, Events).`
      }
    ];

    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini

