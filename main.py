"""
DhanRakshak Face Auth Backend
==============================
Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import numpy as np
import json
import os
import time

# ─── Optional: swap this import for a real DB later ───────────────────
# For now we use a simple JSON file so you can run this with zero setup.
# See database.py for the PostgreSQL + pgvector version.
# ──────────────────────────────────────────────────────────────────────
DB_FILE = "face_db.json"   # local JSON database

# ─── Load DB on startup ───────────────────────────────────────────────
def load_db() -> dict:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_db(db: dict):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

face_db = load_db()   # { "user_001": { "embedding": [...128 floats...], "name": "Ravi" } }

# ─── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="DhanRakshak Face Auth API",
    description="Stores and verifies 128-D face embeddings for banking authentication",
    version="3.0.0"
)

# CORS — allow your HTML page to call this API
# In production: replace "*" with your actual domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # e.g. ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
   
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Request / Response models ────────────────────────────────────────
class EnrolRequest(BaseModel):
    userId: str                 # e.g. "user_001"
    embedding: List[float]      # 128 floats from face-api.js

class VerifyRequest(BaseModel):
    userId: str
    embedding: List[float]      # live embedding from browser

# ─── Utility: cosine similarity ──────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns value between -1 and 1. Higher = more similar."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT 1: Health check
# ═══════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "DhanRakshak Face Auth",
        "enrolled_users": len(face_db),
        "timestamp": time.time()
    }

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT 2: ENROL A NEW USER
# POST /api/enrol
# Body: { "userId": "user_001", "embedding": [0.12, -0.45, ...] }
# ═══════════════════════════════════════════════════════════════════════
@app.post("/api/enrol")
def enrol_user(req: EnrolRequest):
    # Validate embedding length
    if len(req.embedding) != 128:
        raise HTTPException(status_code=400, detail=f"Embedding must be 128-D, got {len(req.embedding)}")

    # Validate userId
    if not req.userId.strip():
        raise HTTPException(status_code=400, detail="userId cannot be empty")

    # Save to DB
    face_db[req.userId] = {
        "embedding": req.embedding,    # 128 floats
        "enrolled_at": time.time(),
        "userId": req.userId
    }
    save_db(face_db)

    print(f"[ENROL] User '{req.userId}' enrolled successfully")
    return {
        "success": True,
        "message": f"User '{req.userId}' enrolled successfully",
        "userId": req.userId,
        "embeddingDimensions": len(req.embedding)
    }

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT 3: GET STORED EMBEDDING (called at login start)
# GET /api/get-embedding/{userId}
# Returns the stored embedding so the browser can compare locally
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/get-embedding/{userId}")
def get_embedding(userId: str):
    if userId not in face_db:
        raise HTTPException(
            status_code=404,
            detail=f"User '{userId}' not found. Please enrol first."
        )
    record = face_db[userId]
    return {
        "userId": userId,
        "embedding": record["embedding"],   # send back to browser
        "enrolledAt": record.get("enrolled_at", 0)
    }

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT 4: SERVER-SIDE VERIFY (optional — browser already compares,
# but you can also do a server-side double-check here)
# POST /api/verify-face
# Body: { "userId": "user_001", "embedding": [...] }
# ═══════════════════════════════════════════════════════════════════════
@app.post("/api/verify-face")
def verify_face(req: VerifyRequest):
    # Look up user
    if req.userId not in face_db:
        raise HTTPException(status_code=404, detail=f"User '{req.userId}' not found")

    if len(req.embedding) != 128:
        raise HTTPException(status_code=400, detail="Embedding must be 128-D")

    # Compare
    live_emb   = np.array(req.embedding, dtype=np.float32)
    stored_emb = np.array(face_db[req.userId]["embedding"], dtype=np.float32)
    similarity = cosine_similarity(live_emb, stored_emb)

    THRESHOLD = 0.55   # ← tune this: 0.5 = lenient, 0.65 = strict

    matched = similarity >= THRESHOLD
    print(f"[VERIFY] User '{req.userId}' | similarity={similarity:.4f} | matched={matched}")

    if matched:
        return {
            "match": True,
            "similarity": round(similarity, 4),
            "userId": req.userId
        }
    else:
        raise HTTPException(
            status_code=401,
            detail={
                "match": False,
                "similarity": round(similarity, 4),
                "message": "Face does not match stored record"
            }
        )

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT 5: LIST ENROLLED USERS (admin/debug)
# GET /api/users
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/users")
def list_users():
    return {
        "count": len(face_db),
        "users": [
            {"userId": uid, "enrolledAt": data.get("enrolled_at", 0)}
            for uid, data in face_db.items()
        ]
    }

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT 6: DELETE USER (admin)
# DELETE /api/users/{userId}
# ═══════════════════════════════════════════════════════════════════════
@app.delete("/api/users/{userId}")
def delete_user(userId: str):
    if userId not in face_db:
        raise HTTPException(status_code=404, detail="User not found")
    del face_db[userId]
    save_db(face_db)
    return {"success": True, "message": f"User '{userId}' deleted"}
