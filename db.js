require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

async function initDB() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      name TEXT UNIQUE NOT NULL,
      token TEXT UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS meetings (
      id SERIAL PRIMARY KEY,
      user_name TEXT NOT NULL,
      title TEXT NOT NULL,
      audio_path TEXT,
      transcript TEXT,
      summary JSONB,
      status TEXT DEFAULT 'processing',
      created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id SERIAL PRIMARY KEY,
      meeting_id INT REFERENCES meetings(id) ON DELETE CASCADE,
      user_name TEXT NOT NULL,
      content TEXT NOT NULL,
      embedding vector(384),
      chunk_index INT
    );

    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
      ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
  `);
  console.log('DB ready');
}

module.exports = { pool, initDB };