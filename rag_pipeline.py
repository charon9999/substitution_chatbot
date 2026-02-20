import hashlib
import time

from config import CACHE_TTL
from database import (
    get_products_by_skus,
    get_product_bullets_batch,
    get_product_specs_batch,
)
from vector_store import search_similar_products
from gemini_client import rank_substitutes

# ---------------------------------------------------------------------------
# Two-level in-memory cache (vector search results + Gemini ranking results)
# Each entry: (timestamp, payload)
# ---------------------------------------------------------------------------
_search_cache: dict[str, tuple[float, list[dict]]] = {}
_gemini_cache: dict[str, tuple[float, dict]] = {}


def _search_key(query_text: str, supercategory: str, category: str) -> str:
    return hashlib.md5(f"{query_text}|{supercategory}|{category}".encode()).hexdigest()


def _gemini_key(source_item: dict) -> str:
    parts = "|".join([
        source_item["name"],
        source_item.get("description", ""),
        source_item["supercategory"],
        source_item["category"],
        str(source_item["quantity"]),
        source_item.get("quantity_unit", ""),
    ])
    return hashlib.md5(parts.encode()).hexdigest()


def _cache_get(store: dict, key: str):
    if CACHE_TTL <= 0:
        return None
    entry = store.get(key)
    if entry and (time.time() - entry[0]) < CACHE_TTL:
        return entry[1]
    store.pop(key, None)
    return None


def _cache_set(store: dict, key: str, value) -> None:
    if CACHE_TTL > 0:
        store[key] = (time.time(), value)


def find_substitutes(source_item: dict):
    """
    Full RAG pipeline for user-provided item. Yields SSE event dicts.

    Event types:
        {"type": "status", "message": "..."}
        {"type": "result", "data": {...}}
        {"type": "error",  "message": "..."}

    source_item keys:
        name, description, supercategory, category, quantity, quantity_unit, unit_price, total_price
    """
    supercategory = source_item.get("supercategory", "")
    category = source_item.get("category", "")

    if not supercategory or not category:
        yield {"type": "error", "message": "Both supercategory and category are required."}
        return

    # 1. Build query text from user-provided item details
    query_parts = [source_item["name"]]
    if source_item.get("description"):
        query_parts.append(source_item["description"])
    query_text = "\n".join(query_parts)

    yield {"type": "status", "message": "Searching catalog..."}

    # 2. Vector search — cached by (query_text, supercategory, category)
    s_key = _search_key(query_text, supercategory, category)
    candidates = _cache_get(_search_cache, s_key)
    if candidates is None:
        candidates = search_similar_products(
            query_text=query_text,
            supercategory=supercategory,
            category=category,
            exclude_sku="",
        )
        _cache_set(_search_cache, s_key, candidates)

    if not candidates:
        yield {"type": "result", "data": {
            "source_item": source_item,
            "message": f"No candidate products found in '{supercategory} > {category}'.",
            "substitutes": [],
        }}
        return

    yield {"type": "status", "message": f"Found {len(candidates)} candidates — AI is ranking..."}

    # 3. Gemini ranking — cached by (name, desc, supercategory, category, qty, unit)
    g_key = _gemini_key(source_item)
    gemini_result = _cache_get(_gemini_cache, g_key)
    if gemini_result is None:
        gemini_result = rank_substitutes(
            source_item=source_item,
            candidates=candidates,
        )
        _cache_set(_gemini_cache, g_key, gemini_result)

    yield {"type": "status", "message": "Ranking complete, fetching product details..."}

    # 4. Enrich with DB data — single batch per field type (3 queries total)
    substitute_skus = [s["sku"] for s in gemini_result.get("substitutes", [])]
    db_products = {p["sku"]: p for p in get_products_by_skus(substitute_skus)}
    bullets_map = get_product_bullets_batch(substitute_skus)
    specs_map = get_product_specs_batch(substitute_skus)

    # 5. Compute savings
    their_unit_price = float(source_item.get("unit_price") or 0)
    their_total_spend = float(source_item.get("total_price") or 0)
    user_quantity = float(source_item["quantity"])

    if their_total_spend <= 0 and their_unit_price > 0:
        their_total_spend = their_unit_price * user_quantity

    has_pricing = their_total_spend > 0 or their_unit_price > 0
    substitutes = []

    for sub in gemini_result.get("substitutes", []):
        sku = sub["sku"]
        if sku not in db_products:
            continue

        product = db_products[sku]
        our_unit_price = float(product.get("customer_price") or product.get("web_price") or 0)
        qty_needed = sub["qty_needed"]
        our_total_spend = qty_needed * our_unit_price

        if has_pricing:
            savings = their_total_spend - our_total_spend
            savings_pct = round((savings / their_total_spend) * 100, 2) if their_total_spend else 0
        else:
            savings = None
            savings_pct = None

        substitutes.append({
            # From Gemini (only what it must determine)
            "rank": sub["rank"],
            "reason": sub["reason"],
            "unit_type": sub["unit_type"],
            "qty_needed": qty_needed,
            "comparison_notes": sub["comparison_notes"],
            # From DB
            "sku": sku,
            "product_name": product["name"],
            "brand_name": product.get("brand_name"),
            "candidate_uom": f"{product.get('uom_qty', 1)} {product.get('uom', 'Each')}",
            "our_unit_price": our_unit_price,
            # Computed
            "our_total_spend": round(our_total_spend, 2),
            "their_unit_price": their_unit_price if has_pricing else None,
            "their_total_spend": round(their_total_spend, 2) if has_pricing else None,
            "savings": round(savings, 2) if savings is not None else None,
            "savings_percentage": savings_pct,
            # Enrichment from batch queries
            "product_details": _serialize_product(product),
            "bullets": bullets_map.get(sku, []),
            "specs": specs_map.get(sku, {}),
        })

    yield {"type": "result", "data": {
        "source_item": source_item,
        "candidates_evaluated": len(candidates),
        "substitutes": substitutes,
    }}


def _serialize_product(product: dict) -> dict:
    """Make product dict JSON-serializable (handle Decimal types)."""
    result = {}
    for k, v in product.items():
        if hasattr(v, "as_integer_ratio"):  # Decimal/float
            result[k] = float(v)
        else:
            result[k] = v
    return result
