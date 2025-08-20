require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { OpenAI } = require('openai');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const intentKeywords = [
  {
    intent: "check_tickets",
    keywords: ["tickets", "how many", "balance", "check my tickets"]
  },
  {
    intent: "purchase_tickets",
    keywords: ["buy tickets", "get tickets", "purchase", "book"]
  },
  {
    intent: "opening_hours",
    keywords: ["open", "close", "when", "time", "hours"]
  }
];

const intentResponses = {
  check_tickets: "🎟️ You can check your tickets here: [Customer Dashboard](https://pos.kartingcentral.co.uk/home/download/pos2/pos2/custdash.php)",
  purchase_tickets: "💸 You can purchase new tickets here: [Book Tickets](https://www.kartingcentral.co.uk/gokarting--ticket-bookings)",
  opening_hours: "🕒 We're open from the afternoon until 10PM daily (except Mon & Tues)."
};

function getIntent(query) {
  query = query.toLowerCase();
  for (const entry of intentKeywords) {
    for (const keyword of entry.keywords) {
      if (query.includes(keyword.toLowerCase())) {
        return entry.intent;
      }
    }
  }
  return null;
}

app.post('/api/faq-response', async (req, res) => {
  const { query } = req.body;

  const intent = getIntent(query);
  if (intent && intentResponses[intent]) {
    return res.json({ response: intentResponses[intent] });
  }

  try {
    const chatCompletion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: query }],
    });

    res.json({ response: chatCompletion.choices[0].message.content });
  } catch (error) {
    console.error('OpenAI Error:', error);
    res.status(500).json({ error: '⚠️ Server error. Please try again shortly.' });
  }
});

app.listen(port, () => {
  console.log(`✅ Chatbot API running at http://localhost:${port}`);
});