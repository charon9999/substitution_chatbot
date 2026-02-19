import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, TOP_K_VECTOR
from database import get_product_count, get_products_batch_for_indexing


def _sku_to_uuid(sku: str) -> str:
    """Convert a SKU string to a deterministic UUID (Qdrant requires int or UUID IDs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, sku))

_BATCH_SIZE_DB = 500
_BATCH_SIZE_QDRANT = 100


def _get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _build_document(product: dict) -> str:
    """Rich document used for embedding generation — good semantic coverage."""
    parts = [
        f"Product: {product['name']}",
        f"Brand: {product['brand_name'] or 'N/A'}",
        f"UOM: {product['uom_qty'] or 1} {product['uom'] or 'Each'}",
        f"Customer Price: ${product['customer_price'] or product['web_price'] or 0}",
    ]
    if product.get("description"):
        parts.append(f"Description: {product['description']}")
    if product.get("bullets"):
        parts.append("Features: " + "; ".join(product["bullets"]))
    if product.get("specs"):
        spec_str = "; ".join(f"{k}: {v}" for k, v in product["specs"].items())
        parts.append(f"Specifications: {spec_str}")
    return "\n".join(parts)


def _build_slim_document(product: dict) -> str:
    """Compact document sent to Gemini — cuts prompt tokens by ~80%."""
    parts = [
        f"Product: {product['name']}",
        f"Brand: {product['brand_name'] or 'N/A'}",
        f"UOM: {product['uom_qty'] or 1} {product['uom'] or 'Each'}",
        f"Customer Price: ${product['customer_price'] or product['web_price'] or 0}",
    ]
    if product.get("specs"):
        specs = list(product["specs"].items())[:10]  # cap at 10 key specs
        spec_str = "; ".join(f"{k}: {v}" for k, v in specs)
        parts.append(f"Specifications: {spec_str}")
    return "\n".join(parts)


def index_products():
    """Index all products from MySQL into Qdrant. Streams in batches."""
    client = _get_client()

    # Drop existing collection so re-index is always clean
    try:
        client.delete_collection(QDRANT_COLLECTION)
    except Exception:
        pass

    total = get_product_count()
    if total == 0:
        print("No products found to index.")
        return

    print(f"Indexing {total} products into Qdrant...", flush=True)
    indexed = 0

    for offset in range(0, total, _BATCH_SIZE_DB):
        products = get_products_batch_for_indexing(offset, _BATCH_SIZE_DB)
        if not products:
            break

        for i in range(0, len(products), _BATCH_SIZE_QDRANT):
            batch = products[i: i + _BATCH_SIZE_QDRANT]

            ids = [_sku_to_uuid(p["sku"]) for p in batch]
            documents = [_build_document(p) for p in batch]
            metadatas = [
                {
                    "sku": p["sku"],
                    "supercategory": p.get("supercategory") or "",
                    "category": p.get("category") or "",
                    "class": p.get("class") or "",
                    "brand_name": p.get("brand_name") or "",
                    "web_price": float(p.get("web_price") or 0),
                    "uom": p.get("uom") or "",
                    "uom_qty": int(p.get("uom_qty") or 1),
                    "name": p.get("name") or "",
                    "slim_doc": _build_slim_document(p),  # stored for Gemini prompt
                }
                for p in batch
            ]

            client.add(
                collection_name=QDRANT_COLLECTION,
                documents=documents,
                metadata=metadatas,
                ids=ids,
            )
            indexed += len(batch)
            pct = round(indexed / total * 100)
            print(f"Indexed {indexed}/{total} products ({pct}%)", flush=True)

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"Indexing complete. Total: {info.points_count} products in Qdrant.", flush=True)


def search_similar_products(
    query_text: str,
    supercategory: str,
    category: str,
    exclude_sku: str,
    n_results: int = TOP_K_VECTOR,
) -> list[dict]:
    """Search for similar products filtered by supercategory and category."""
    client = _get_client()

    query_filter = Filter(
        must=[
            FieldCondition(key="supercategory", match=MatchValue(value=supercategory)),
            FieldCondition(key="category", match=MatchValue(value=category)),
        ]
    )

    results = client.query(
        collection_name=QDRANT_COLLECTION,
        query_text=query_text,
        query_filter=query_filter,
        limit=n_results + 1,  # +1 in case source product appears
    )

    candidates = []
    for result in results:
        sku = result.metadata.get("sku", "")
        if exclude_sku and sku == exclude_sku:
            continue
        candidates.append(
            {
                "sku": sku,
                "score": result.score,
                "metadata": result.metadata,
                # slim_doc is what Gemini sees — keeps prompt token count low
                "document": result.metadata.get("slim_doc", result.document or ""),
            }
        )

    return candidates[:n_results]


if __name__ == "__main__":
    print("Starting product indexing...")
    index_products()
