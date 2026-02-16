from database import get_products_by_skus, get_product_bullets, get_product_specs
from vector_store import search_similar_products
from gemini_client import rank_substitutes


def find_substitutes(source_item: dict) -> dict:
    """
    Full RAG pipeline for user-provided item.

    source_item keys:
        name, description, supercategory, category, quantity, quantity_unit, unit_price, total_price
    """
    supercategory = source_item.get("supercategory", "")
    category = source_item.get("category", "")

    if not supercategory or not category:
        return {"error": "Both supercategory and category are required."}

    # 1. Build query text from user-provided item details
    query_parts = [source_item["name"]]
    if source_item.get("description"):
        query_parts.append(source_item["description"])
    query_text = "\n".join(query_parts)

    # 2. Search ChromaDB filtered by supercategory + category
    candidates = search_similar_products(
        query_text=query_text,
        supercategory=supercategory,
        category=category,
        exclude_sku="",
    )

    if not candidates:
        return {
            "source_item": source_item,
            "message": f"No candidate products found in '{supercategory} > {category}'.",
            "substitutes": [],
        }

    # 3. Send candidates to Gemini for ranking (returns minimal fields)
    gemini_result = rank_substitutes(
        source_item=source_item,
        candidates=candidates,
    )

    # 4. Enrich with DB data and compute savings
    substitute_skus = [s["sku"] for s in gemini_result.get("substitutes", [])]
    db_products = {p["sku"]: p for p in get_products_by_skus(substitute_skus)}

    their_unit_price = float(source_item.get("unit_price") or 0)
    their_total_spend = float(source_item.get("total_price") or 0)
    user_quantity = float(source_item["quantity"])

    # Derive total spend from whichever price the user provided
    if their_total_spend > 0:
        # User gave total — use it directly
        pass
    elif their_unit_price > 0:
        # User gave unit price only — compute total
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
            # Extra DB enrichment
            "product_details": _serialize_product(product),
            "bullets": get_product_bullets(sku),
            "specs": get_product_specs(sku),
        })

    return {
        "source_item": source_item,
        "candidates_evaluated": len(candidates),
        "substitutes": substitutes,
    }


def _serialize_product(product: dict) -> dict:
    """Make product dict JSON-serializable (handle Decimal types)."""
    result = {}
    for k, v in product.items():
        if hasattr(v, "as_integer_ratio"):  # Decimal/float
            result[k] = float(v)
        else:
            result[k] = v
    return result
