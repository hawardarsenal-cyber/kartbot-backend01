const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { OpenAI } = require('openai');

dotenv.config();
const app = express();

app.use(cors()); // ✅ Enables all CORS by default
app.use(express.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.post('/api/faq-response', async (req, res) => {
  const { query } = req.body;

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o', // ✅ Make sure you're using "gpt-4o"
      messages: [
        { role: 'system', content: 'You are a helpful karting assistant.' },
        { role: 'user', content: query }
      ],
      temperature: 0.6,
    });

    res.json({ response: response.choices[0].message.content });
  } catch (error) {
    console.error('Error from OpenAI:', error.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`✅ Chatbot API running on port ${PORT}`));
