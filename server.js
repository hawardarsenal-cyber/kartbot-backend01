// server.js
import "dotenv/config";            // optional but handy for local .env
import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import OpenAI from "openai";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());
app.use(
  cors({
    origin: [
      "https://www.kartingcentral.co.uk", // TODO: replace with your real site
      "http://localhost:3000"
    ],
    methods: ["POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"]
  })
);

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------- Load Knowledge Base ----------
const KB_PATH = path.join(__dirname, "routes", "knowledge.kartingcentral.json"); // file lives in /routes/
let KB = JSON.parse(fs.readFileSync(KB_PATH, "utf-8"));

const EMB_MODEL = "text-embedding-3-small";

// Flatten KB into small text chunks
function flattenDocs() {
  const docs = [];

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
    text: `Promotional discounts do not apply on Saturdays.`,
    url: KB.site.urls.terms
  });

  docs.push({
    id: "f1_sim",
    text: `F1 simulator at Gillingham. 50% off with Karting Central tickets.`,
    url: KB.site.urls.home
  });

  docs.push({
    id: "tracking_and_tickets",
    text: `Tracking codes for managing tickets. Gifting available via Customer Dashboard.`,
    url: KB.site.urls.customer_dashboard
  });

  // Fast answers become retrievable chunks too
  for (const f of KB.fast_answers) {
    docs.push({
      id: `fast_${f.intent}`,
      text: `${f.intent}: ${f.answer}`,
      url: KB.site.urls.home
    });
  }

  return docs;
}

// In-memory vectors
let DOCS = [];
let VECTORS = []; // [{ id, embedding, meta }]

// Cosine similarity
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

async function retrieve(query, k = 5) {
  const q = await openai.embeddings.create({ model: EMB_MODEL, input: query });
  const qEmb = q.data[0].embedding;
  return VECTORS
    .map(v => ({ v, score: cosine(qEmb, v.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// Fast keyword intents
function matchIntent(q) {
  const text = q.toLowerCase();
  for (const f of KB.fast_answers) {
    if (f.keywords.some(k => text.includes(k.toLowerCase()))) {
      return f;
    }
  }
  return null;
}

// Belt & braces: force GBP symbol
function enforceGBP(text) {
  return (text || "").replace(/\$/g, "£");
}

// Prompt
const SYSTEM_PROMPT = `
You are the Karting Central website assistant.
- Always use UK English and GBP (£).
- Ground answers in the provided context; if context is missing, answer generally but stay relevant to indoor electric karting and our venue.
- Be concise and friendly. Start with the direct answer, then 1–2 bullets if helpful.
- Include a clear CTA with a correct site path when relevant (Book Experience, Customer Dashboard, Safety, Events).
- Do not invent prices, hours, or policies not present in context. If something is missing, say so and point to booking or the Customer Dashboard.
- If a user mentions existing tickets or tracking codes, prefer the Customer Dashboard link from context.
`;

// Chat route
app.post("/api/faq-response", async (req, res) => {
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });

    // 1) Fast intent
    const intent = matchIntent(query);
    if (intent) {
      return res.json({
        response: enforceGBP(intent.answer),
        sources: [{ id: intent.intent, url: KB.site.urls.home, score: 1 }]
      });
    }

    // 2) Retrieve KB with threshold
    const retrieved = await retrieve(query, 5);
    const top = retrieved[0];
    const RELEVANCE_MIN = 0.75; // adjust if needed

    if (!top || top.score < RELEVANCE_MIN) {
      const fallback = enforceGBP(
        `I may need a bit more detail to help with that.
- I can help with booking, opening hours, height limits, track details, or ticket tracking/gifting.
- If you already have tickets, use the Customer Dashboard to view or gift them.

What would you like to do next?
• Book now: ${KB.site.urls.book_experience}
• Customer Dashboard: ${KB.site.urls.customer_dashboard}`
      );
      return res.json({ response: fallback, sources: [] });
    }

    // 3) Ask the model with grounded context
    const contextBlock = retrieved
      .map((r, i) => `#${i + 1} [${r.v.id}] ${r.v.meta.text} (URL: ${r.v.meta.url})`)
      .join("\n\n");

    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      {
        role: "user",
        content:
`Question: ${query}

Context:
${contextBlock}

Instructions:
- Prefer quoting context.
- If something is unclear or missing, ask a brief follow-up and suggest the closest next step (Book, Dashboard, Events).`
      }
    ];

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.3
    });

    const text =
      enforceGBP(completion.choices?.[0]?.message?.content?.trim()) ||
      "Sorry, I couldn’t find that in our info.";

    const sources = retrieved.map(r => ({
      id: r.v.id,
      url: r.v.meta.url,
      score: Number(r.score.toFixed(3))
    }));

    res.json({ response: text, sources });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

// Health check
app.get("/healthz", (_, res) => res.send("ok"));

// Boot
const port = process.env.PORT || 10000;
app.listen(port, async () => {
  console.log(`Listening on ${port}`);
  try {
    await embedAll();
  } catch (e) {
    console.error("Failed to load knowledge:", e);
  }
});
