// server.js
require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { pool, initDB } = require('./db');
const { transcribe, summarise, embed, chunkText, groq } = require('./ai');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// ── Simple token auth middleware ──
const TOKENS = {};
(process.env.LOCAL_TOKENS || '').split(',').forEach(pair => {
  const [name, token] = pair.trim().split(':');
  if (name && token) TOKENS[token] = name;
});

function auth(req, res, next) {
  const token = req.headers['x-token'] || req.query.token;
  const user  = TOKENS[token];

  // ADD THIS LINE:
  console.log('Auth check — token received:', JSON.stringify(token), '| known tokens:', JSON.stringify(Object.keys(TOKENS)));

  if (!user) return res.status(401).json({ error: 'Invalid token' });
  req.user = user;
  next();
}

// ── POST /api/meetings — upload audio ──
app.post('/api/meetings', auth, upload.single('audio'), async (req, res) => {
  const title = req.body.title || 'Untitled Meeting';

  const { rows } = await pool.query(
    `INSERT INTO meetings (user_name, title, audio_path, status)
     VALUES ($1, $2, $3, 'processing')
     RETURNING id`,
    [req.user, title, req.file.path]
  );

  const meetingId = rows[0].id;

  res.json({
    id: meetingId,
    status: 'processing'
  });

  setImmediate(async () => {
    try {
      // 1. Transcribe audio to text
      const transcript = await transcribe(req.file.path);

      // 2. Generate structured summary
      const summary = await summarise(transcript);

      // 3. Save transcript and summary
      await pool.query(
        `UPDATE meetings
         SET transcript = $1,
             summary = $2,
             status = 'done'
         WHERE id = $3`,
        [transcript, JSON.stringify(summary), meetingId]
      );

      // 4. Chunk transcript and generate embeddings
      const chunks = chunkText(transcript);

      for (let i = 0; i < chunks.length; i++) {
        const vector = await embed(chunks[i]);

        await pool.query(
          `INSERT INTO chunks
           (meeting_id, user_name, content, embedding, chunk_index)
           VALUES ($1, $2, $3, $4::vector, $5)`,
          [
            meetingId,
            req.user,
            chunks[i],
            JSON.stringify(vector),
            i
          ]
        );
      }

      console.log(`Meeting ${meetingId} processed successfully`);
    } catch (error) {
      console.error('Processing failed:', error);

      await pool.query(
        `UPDATE meetings
         SET status = 'failed'
         WHERE id = $1`,
        [meetingId]
      );
    }
  });
});

// ── GET /api/meetings — list all meetings for current user ──
app.get('/api/meetings', auth, async (req, res) => {
  const { rows } = await pool.query(
    `SELECT
        id,
        title,
        status,
        created_at,
        summary->>'overview' AS overview
     FROM meetings
     WHERE user_name = $1
     ORDER BY created_at DESC`,
    [req.user]
  );

  res.json(rows);
});

// ── GET /api/meetings/:id — full meeting detail ──
app.get('/api/meetings/:id', auth, async (req, res) => {
  const { rows } = await pool.query(
    `SELECT *
     FROM meetings
     WHERE id = $1
       AND user_name = $2`,
    [req.params.id, req.user]
  );

  if (!rows[0]) {
    return res.status(404).json({ error: 'Not found' });
  }

  res.json(rows[0]);
});

// ── GET /api/meetings/:id/status — poll processing status ──
app.get('/api/meetings/:id/status', auth, async (req, res) => {
  const { rows } = await pool.query(
    `SELECT id, status
     FROM meetings
     WHERE id = $1
       AND user_name = $2`,
    [req.params.id, req.user]
  );

  if (!rows[0]) {
    return res.status(404).json({ error: 'Not found' });
  }

  res.json(rows[0]);
});

// ── POST /api/chat — RAG question answering using Groq ──
app.post('/api/chat', auth, async (req, res) => {
  try {
    const { question, meetingId } = req.body;

    // 1. Embed the user's question
    const questionVector = await embed(question);

    // 2. Search relevant chunks
    let query = `
      SELECT
        c.content,
        m.title,
        m.created_at,
        1 - (c.embedding <=> $1::vector) AS similarity
      FROM chunks c
      JOIN meetings m ON m.id = c.meeting_id
      WHERE c.user_name = $2
    `;

    const params = [
      JSON.stringify(questionVector),
      req.user
    ];

    if (meetingId) {
      query += ` AND c.meeting_id = $3 `;
      params.push(meetingId);
    }

    query += `
      ORDER BY c.embedding <=> $1::vector
      LIMIT 6
    `;

    const { rows } = await pool.query(query, params);

    // 3. Build context from retrieved chunks
    const context = rows
      .map((row, index) => {
        const date = new Date(row.created_at).toDateString();
        return `[${index + 1}] From "${row.title}" (${date}):\n${row.content}`;
      })
      .join('\n\n');

    // 4. Ask Groq model
    const response = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      temperature: 0,
      messages: [
        {
          role: 'system',
          content:
            'You are a meeting assistant. Answer using only the provided transcript excerpts. If the answer is not present, say so.'
        },
        {
          role: 'user',
          content: `Excerpts:
${context}

Question: ${question}

Include references to the excerpt numbers and meeting titles in your answer.`
        }
      ]
    });

    const answer = response.choices[0].message.content;

    res.json({
      answer,
      sources: rows
    });
  } catch (error) {
    console.error('Chat failed:', error);
    res.status(500).json({
      error: 'Failed to generate answer'
    });
  }
});

// ── Start Server ──
initDB().then(() => {
  const port = process.env.PORT || 3000;

  app.listen(port, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${port}`);
    console.log('Users:', Object.values(TOKENS).join(', '));
  });
});