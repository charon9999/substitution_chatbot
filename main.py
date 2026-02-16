from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import MAX_REQUESTS_PER_USER
from rag_pipeline import find_substitutes
from vector_store import index_products
from database import get_categories

# In-memory request counter per IP
_request_counts: dict[str, int] = defaultdict(int)

app = FastAPI(
    title="Product Substitution API",
    description="RAG-based product substitution finder using ChromaDB + Gemini",
    version="2.0.0",
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
def get_substitutes(request: SubstitutionRequest, req: Request):
    """Find top 5 product substitutes with adjusted pricing and savings."""
    client_ip = req.client.host if req.client else "unknown"

    if _request_counts[client_ip] >= MAX_REQUESTS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({MAX_REQUESTS_PER_USER} requests). Please contact support for more access.",
        )

    _request_counts[client_ip] += 1
    source_item = request.model_dump()

    result = find_substitutes(source_item)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    result["requests_remaining"] = MAX_REQUESTS_PER_USER - _request_counts[client_ip]
    return result


@app.get("/categories")
def list_categories():
    """List all available supercategory + category pairs for selection."""
    return get_categories()


@app.post("/index")
def reindex_products():
    """Re-index all products from MySQL into ChromaDB."""
    index_products()
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
