# Product Substitution Finder

A RAG-based product substitution engine. Given a competitor's product, it finds the best alternatives from the internal catalog — ranked by maximum savings — using semantic vector search and Gemini AI.

---

## Architecture

```
User (Browser)
    │
    ▼
FastAPI  (port 8000)
    │
    ├─► Qdrant Vector DB  ◄─── fastembed (BAAI/bge-small-en-v1.5)
    │       • Semantic search
    │       • Filtered by supercategory + category
    │       • Returns top 50 candidates (slim documents)
    │
    ├─► Gemini 2.5 Flash
    │       • Ranks 50 candidates
    │       • Calculates qty_needed with unit conversions
    │       • Returns top 5 substitutes
    │
    └─► MySQL
            • Enriches final 5 results (bullets, specs, pricing)
```

**Two-level in-memory cache** sits in front of Qdrant and Gemini. Repeat queries for the same product skip both and return instantly.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API server | FastAPI + Uvicorn |
| Vector DB | Qdrant (Docker) |
| Embedding model | `BAAI/bge-small-en-v1.5` via fastembed |
| AI ranking | Google Gemini 2.5 Flash |
| Product database | MySQL |
| Frontend | Vanilla JS (served from `/static`) |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
.
├── main.py              # FastAPI app, endpoints, rate limiting
├── rag_pipeline.py      # Full pipeline: cache → Qdrant → Gemini → DB → response
├── vector_store.py      # Qdrant indexing and semantic search
├── gemini_client.py     # Gemini ranking call (schema-enforced JSON)
├── database.py          # MySQL queries (products, bullets, specs, categories)
├── config.py            # All configuration via environment variables
├── static/
│   └── index.html       # Single-page frontend
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Configuration

All settings are controlled via environment variables (`.env` file or `docker-compose.yml`).

| Variable | Default | Description |
|----------|---------|-------------|
| `MYSQL_HOST` | `host.docker.internal` | MySQL host (use `host.docker.internal` in Docker on Windows/Mac) |
| `MYSQL_PORT` | `3307` | MySQL port |
| `MYSQL_USER` | `root` | MySQL user |
| `MYSQL_PASSWORD` | _(empty)_ | MySQL password |
| `MYSQL_DB` | `staples_data` | MySQL database name |
| `GEMINI_API_KEY` | _(required)_ | Google Gemini API key |
| `GEMINI_RANKING_MODEL` | `gemini-2.5-flash` | Gemini model for ranking step |
| `QDRANT_HOST` | `qdrant` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `TOP_K_VECTOR` | `50` | Candidates fetched from Qdrant per query |
| `TOP_K_FINAL` | `5` | Final substitutes returned to user |
| `MAX_REQUESTS_PER_USER` | `5` | Per-IP rate limit (resets on restart) |
| `CACHE_TTL` | `3600` | Cache lifetime in seconds (`0` = disabled) |

---

## API Endpoints

### `POST /substitute`
Find the best substitutes for a competitor product.

**Request body:**
```json
{
  "name": "Copy Paper Letter Size",
  "description": "8.5 x 11 white 20lb copy paper",
  "supercategory": "Office Supplies",
  "category": "Copy Paper",
  "quantity": 500,
  "quantity_unit": "Sheets",
  "unit_price": 12.99,
  "total_price": 0
}
```
- `name`, `supercategory`, `category`, `quantity` are required.
- Provide either `unit_price` or `total_price` (or both) to get savings calculations. If neither is provided, results show a quote price only.

**Response:**
```json
{
  "source_item": { ... },
  "candidates_evaluated": 50,
  "requests_remaining": 4,
  "substitutes": [
    {
      "rank": 1,
      "sku": "ABC123",
      "product_name": "...",
      "brand_name": "...",
      "candidate_uom": "500 Sheets",
      "unit_type": "DIVISIBLE",
      "qty_needed": 1,
      "our_unit_price": 9.49,
      "our_total_spend": 9.49,
      "their_unit_price": 12.99,
      "their_total_spend": 12.99,
      "savings": 3.50,
      "savings_percentage": 26.94,
      "reason": "...",
      "comparison_notes": "...",
      "product_details": { ... },
      "bullets": ["Feature 1", "Feature 2"],
      "specs": { "Weight": "20lb", "Size": "8.5x11" }
    }
  ]
}
```

**Rate limit:** Returns HTTP 429 when the per-IP limit is exceeded.

---

### `GET /categories`
Returns all supercategory + category pairs for the frontend dropdowns.

```json
[
  { "supercategory": "Office Supplies", "category": "Copy Paper" },
  ...
]
```

---

### `POST /index`
Re-indexes all active products from MySQL into Qdrant. Run this after product data changes.

```bash
curl -X POST http://localhost:8000/index
```

---

### `GET /health`
Simple health check.

```json
{ "status": "ok" }
```

---

### `GET /`
Serves the frontend (`static/index.html`).

---

## How the Ranking Works

Gemini classifies each product attribute as:

- **DIVISIBLE** — can be scaled (sheets, rolls, feet, ml, oz). `qty_needed = ceil(user_qty / candidate_qty)`
- **ABSOLUTE** — cannot be scaled (tabs, compartments, drawers, ports). Candidate must have identical spec or it is excluded.

Ranking priority:
1. Functional match (same purpose and specs)
2. **Maximum savings** — lowest total spend (`qty_needed × price`)
3. Package size is irrelevant
4. Brand is irrelevant

---

## Initial Setup

### Prerequisites
- Docker Desktop running
- `.env` file with credentials (see Configuration table above)

### First-time deployment

```bash
# 1. Start containers
docker-compose up --build -d

# 2. Index all products into Qdrant (one-time, ~2-5 min for 120K products)
curl -X POST http://localhost:8000/index

# 3. Open the frontend
# http://localhost:8000
```

Watch indexing progress:
```bash
docker-compose logs -f api
```

---

## How to Rebuild / Re-deploy on Server

### Rebuild after code changes

```bash
# Pull latest code changes first, then:
docker-compose up --build -d
```

This rebuilds the `api` image with the new code. Qdrant data is preserved in the `qdrant_data` Docker volume — **no re-indexing needed** unless the indexing logic itself changed.

### Re-index products (after MySQL data changes)

```bash
curl -X POST http://localhost:8000/index

# On Windows PowerShell:
curl.exe -X POST http://localhost:8000/index

# Or native PowerShell:
Invoke-WebRequest -Method POST -Uri http://localhost:8000/index
```

This drops and recreates the Qdrant collection. Safe to run at any time — the old index stays active until the new one is fully built.

### Full teardown and rebuild

```bash
# Stop and remove containers + volumes (DELETES Qdrant index)
docker-compose down -v

# Rebuild everything from scratch
docker-compose up --build -d

# Re-index
curl -X POST http://localhost:8000/index
```

### View logs

```bash
docker-compose logs -f api      # Live API logs
docker-compose logs -f qdrant   # Live Qdrant logs
docker-compose logs --tail=100 api   # Last 100 lines
```

### Restart without rebuilding

```bash
docker-compose restart api
```

---

## Performance Notes

The following optimisations are in place for large catalogs (120K+ products):

| Optimisation | Detail |
|---|---|
| Qdrant vector DB | Replaces ChromaDB; purpose-built server, faster at scale |
| `TOP_K_VECTOR = 50` | Was 200 — reduces Gemini prompt tokens by ~75% |
| Slim documents in prompt | Only name, brand, UOM, price, top-10 specs sent to Gemini (no description/bullets) |
| Gemini 2.5 Flash | Replaces Pro for ranking — ~5-10x faster, similar accuracy |
| Two-level cache | Vector search + Gemini results cached 1 hr per product+category combination |
| Batch DB queries | Bullets and specs for all 5 final results fetched in 2 queries total |
| Async endpoints | `/substitute` and `/index` run in thread pools, keeping the event loop free |

---

## MySQL Schema (expected)

```sql
products       (sku, name, short_name, brand_name, description,
                web_price, customer_price, uom, uom_qty,
                manufacturer_name, review_rating, review_count, active)

categories     (sku, supercategory, category, class)

product_bullets      (sku, bullet_text, display_order)

product_specifications (sku, spec_name_id, spec_value)

specification_names  (id, name)
```

Only products with `active = 1` are indexed.
