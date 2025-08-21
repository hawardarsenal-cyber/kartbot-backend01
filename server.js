import express from "express";
import cors from "cors";

const app = express();

app.use(cors({
  origin: [
    "https://yourdomain.com",     // prod site
    "http://localhost:3000"       // local dev
  ],
  methods: ["POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
}));
app.use(express.json());

// health check (optional but nice for Render)
app.get("/healthz", (_, res) => res.status(200).send("ok"));

// Your chat route: must accept { query } and return { response }
app.post("/api/faq-response", async (req, res) => {
  try {
    const { query } = req.body ?? {};
    if (!query) return res.status(400).json({ error: "Missing 'query'." });

    // Call OpenAI (or proxy) here and build a reply string
    const reply = await getChatReply(query); // implement this
    res.json({ response: reply });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

const port = process.env.PORT || 10000;
app.listen(port, () => console.log(`Listening on ${port}`));
