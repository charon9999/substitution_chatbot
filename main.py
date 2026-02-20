import asyncio
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from config import MAX_REQUESTS_PER_USER
from rag_pipeline import find_substitutes
from vector_store import index_products
from database import get_categories

# In-memory request counter per IP — resets daily
# Stores {ip: (date, count)}
_request_counts: dict[str, tuple[date, int]] = {}

app = FastAPI(
    title="Product Substitution API",
    description="RAG-based product substitution finder using Qdrant + Gemini",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SubstitutionRequest(BaseModel):
    name: str
    description: str = ""
    supercategory: str
    category: str
    quantity: float
    quantity_unit: str = ""
    unit_price: float = 0
    total_price: float = 0


@app.post("/substitute")
async def get_substitutes(request: SubstitutionRequest, req: Request):
    """Find top 5 product substitutes with adjusted pricing and savings. Streams SSE events."""
    client_ip = req.client.host if req.client else "unknown"
    today = date.today()

    entry = _request_counts.get(client_ip)
    count = (entry[1] if entry and entry[0] == today else 0)

    if count >= MAX_REQUESTS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached ({MAX_REQUESTS_PER_USER} requests). Resets at midnight.",
        )

    count += 1
    _request_counts[client_ip] = (today, count)
    source_item = request.model_dump()
    requests_remaining = MAX_REQUESTS_PER_USER - count

    async def event_stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_pipeline():
            try:
                for event in find_substitutes(source_item):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(None, run_pipeline)

        while True:
            event = await queue.get()
            if event is None:
                break
            if event.get("type") == "result":
                event["data"]["requests_remaining"] = requests_remaining
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/categories")
def list_categories():
    """List all available supercategory + category pairs for selection."""
    return get_categories()


@app.post("/index")
async def reindex_products():
    """Re-index all products from MySQL into Qdrant (runs in background thread)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, index_products)
    return {"message": "Indexing complete."}


@app.get("/health")
def health():
    return {"status": "ok"}


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files last so explicit routes take priority
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
