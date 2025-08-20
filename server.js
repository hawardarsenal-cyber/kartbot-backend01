import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import OpenAI from "openai";

const app = express();
app.use(express.json());
app.use(cors({ origin: ["https://<YOUR_SITE_DOMAIN>", "http://localhost:3000"], methods: ["POST", "OPTIONS"] }));

// ---------- Load FAQs + build embeddings ----------
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const dataPath = path.join(process.cwd(), "faqs.json");

// In-memory store
let FAQS = [];
let VECTORS = []; // [{ id, embedding, meta }]
const EMB_MODEL = "text-embedding-3-small"; // cheap & good

async function loadFaqs() {
  FAQS = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
  // Make one big text per FAQ for better recall
  const inputs = FAQS.map(f => `${f.question}\n${f.answer}\nTags: ${f.tags?.join(", ") || ""}`);
  const resp = await openai.embeddings.create({ model: EMB_MODEL, input: inputs });
  VECTORS = resp.data.map((d, i) => ({
    id: FAQS[i].id,
    embedding: d.embedding,
    meta: FAQS[i]
  }));
  console.log(`Embedded ${VECTORS.length} FAQs`);
}

// Cosine similarity
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Retrieve top-k FAQs
async function retrieve(query, k = 4) {
  const qEmb = (await openai.embeddings.create({ model: EMB_MODEL, input: query })).data[0].embedding;
  const scored = VECTORS.map(v => ({ v, score: cosine(qEmb, v.embedding) }))
                        .sort((a, b) => b.score - a.score)
                        .slice(0, k);
  return scored;
}

// ---------- System prompt + style ----------
const SYSTEM_PROMPT = `
You are Karting Central's website assistant. Answer **only** using the provided context when available.
If the user asks for something off-topic or not in context, say you're focused on karting at Karting Central and briefly offer what you can help with.

Style:
- Friendly, concise, UK English.
- Start with the direct answer, then 1–2 bullet points if helpful.
- Include a clear action (Book, Contact, Learn more) with a site path when relevant.
- Never invent prices, hours, or policies not in context.
- If unsure, ask a short clarifying question.
`;

function buildMessages(userQuery, contextSnippets) {
  const contextBlock = contextSnippets.length
    ? "Context:\n" + contextSnippets.map((c, i) =>
        `#${i+1} [${c.v.meta.id}] ${c.v.meta.question}\n${c.v.meta.answer}\nURL: ${c.v.meta.url}`
      ).join("\n\n")
    : "Context: (none found)";

  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content:
`User question: ${userQuery}

${contextBlock}

Instructions:
- Prefer quoting answers from context.
- If multiple snippets conflict, say which seems most recent/clear (if you maintain metadata).
- If no relevant context, say what you *can* help with and ask a brief follow-up if appropriate.` }
  ];
}

// ---------- Chat route ----------
app.post("/api/faq-response", async (req, res) => {
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });

    const hits = await retrieve(query, 4);
    const messages = buildMessages(query, hits);

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.3,
    });

    const text = completion.choices?.[0]?.message?.content?.trim() || "Sorry, I couldn't find that.";
    // Optional: attach provenance of which snippets were used
    const sources = hits.map(h => ({ id: h.v.meta.id, url: h.v.meta.url, score: Number(h.score.toFixed(3)) }));

    res.json({ response: text, sources });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

app.get("/healthz", (_, res) => res.send("ok"));

const port = process.env.PORT || 10000;
app.listen(port, async () => {
  console.log(`Listening on ${port}`);
  try { await loadFaqs(); } catch (e) { console.error("Failed to load FAQs:", e); }
});
