import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.main import app  # Assumes your FastAPI app is named 'app' in app/main.py

# Vercel will look for 'app' in this file

# --- Debug/Health routes for Vercel ---
from fastapi import Request
from fastapi.responses import JSONResponse

@app.get("/api/health")
async def health():
	return {"status": "ok"}

# Optional: fallback root route for debugging
@app.get("/api/")
async def api_root():
	return {"message": "API root is working. If you see this, FastAPI is running on Vercel."}
