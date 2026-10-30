"""
database.py — PostgreSQL + pgvector version
============================================
This replaces the JSON file in main.py when you're ready for production.

Setup:
  1. Install PostgreSQL 14+
  2. Install pgvector extension:
       sudo apt install postgresql-16-pgvector    (Ubuntu)
       brew install pgvector                       (Mac)
  3. In psql:
       CREATE EXTENSION vector;
  4. Set your DATABASE_URL in a .env file:
       DATABASE_URL=postgresql://user:password@localhost:5432/dhanrakshak

Usage: import the functions below in main.py instead of the JSON functions.
"""

import os
import numpy as np
import asyncpg
from typing import Optional, List

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/dhanrakshak")

# ── Connection pool (created at startup) ─────────────────────────────
pool = None

async def init_db():
    """Call this from FastAPI startup event."""
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)

    async with pool.acquire() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          SERIAL PRIMARY KEY,
                user_id     TEXT UNIQUE NOT NULL,
                embedding   vector(128) NOT NULL,
                enrolled_at TIMESTAMPTZ DEFAULT NOW(),
                is_active   BOOLEAN DEFAULT TRUE
            );
        """)

        # Create HNSW index for fast approximate nearest-neighbor search
        # This is what makes it fast for millions of users
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS users_embedding_idx
            ON users USING hnsw (embedding vector_cosine_ops);
        """)

    print("[DB] PostgreSQL + pgvector initialised")

# ── Enrol ─────────────────────────────────────────────────────────────
async def db_enrol(user_id: str, embedding: List[float]) -> bool:
    """Insert or update user embedding."""
    emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO users (user_id, embedding)
            VALUES ($1, $2::vector)
            ON CONFLICT (user_id) DO UPDATE SET embedding = $2::vector, enrolled_at = NOW();
        """, user_id, emb_str)
    return True

# ── Fetch embedding ───────────────────────────────────────────────────
async def db_get_embedding(user_id: str) -> Optional[List[float]]:
    """Returns 128-float list or None if not found."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT embedding FROM users WHERE user_id = $1 AND is_active = TRUE",
            user_id
        )
    if row is None:
        return None
    # pgvector returns a string like "[0.12,-0.45,...]"
    raw = row["embedding"]
    return [float(x) for x in str(raw).strip("[]").split(",")]

# ── Find closest user (1:N search) ───────────────────────────────────
async def db_find_closest(embedding: List[float], top_k: int = 1):
    """
    Find the top_k closest users to the given embedding.
    Uses the HNSW index — fast even with millions of users.
    Returns list of (user_id, similarity_score).
    """
    emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT user_id,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM users
            WHERE is_active = TRUE
            ORDER BY embedding <=> $1::vector
            LIMIT $2;
        """, emb_str, top_k)
    return [(row["user_id"], float(row["similarity"])) for row in rows]

# ── Delete user ───────────────────────────────────────────────────────
async def db_delete_user(user_id: str) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET is_active = FALSE WHERE user_id = $1", user_id
        )
    return result != "UPDATE 0"

"""
HOW TO SWAP IN THIS FILE:
─────────────────────────
In main.py, at the top, add:
    from contextlib import asynccontextmanager
    from database import init_db, db_enrol, db_get_embedding, db_delete_user

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db()
        yield

    app = FastAPI(lifespan=lifespan, ...)

Then replace the JSON read/write calls with the async db_* functions above.
e.g. in /api/enrol:
    await db_enrol(req.userId, req.embedding)
    return {"success": True, ...}

in /api/get-embedding:
    emb = await db_get_embedding(userId)
    if emb is None:
        raise HTTPException(404, detail="User not found")
    return {"userId": userId, "embedding": emb}
"""
