const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { OpenAI } = require('openai');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// ✅ Enable CORS
app.use(cors({
  origin: '*', // Or specify your frontend domain
}));

app.use(express.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ... your endpoint handlers

app.listen(port, () => {
  console.log(`✅ API running on port ${port}`);
});
