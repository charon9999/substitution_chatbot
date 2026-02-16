import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, TOP_K_VECTOR
from database import get_all_products_for_indexing


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _build_document(product: dict) -> str:
    """Build a rich text document from product data for embedding."""
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


def index_products():
    """Index all products from MySQL into ChromaDB. Clears existing collection first."""
    client = _get_client()

    # Delete and recreate
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    products = get_all_products_for_indexing()
    if not products:
        print("No products found to index.")
        return

    # ChromaDB has a batch limit, process in chunks
    batch_size = 100
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        ids = [p["sku"] for p in batch]
        documents = [_build_document(p) for p in batch]
        metadatas = [
            {
                "supercategory": p.get("supercategory") or "",
                "category": p.get("category") or "",
                "class": p.get("class") or "",
                "brand_name": p.get("brand_name") or "",
                "web_price": float(p.get("web_price") or 0),
                "uom": p.get("uom") or "",
                "uom_qty": int(p.get("uom_qty") or 1),
                "name": p.get("name") or "",
            }
            for p in batch
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Indexed {min(i + batch_size, len(products))}/{len(products)} products")

    print(f"Indexing complete. Total: {collection.count()} products in ChromaDB.")


def search_similar_products(
    query_text: str,
    supercategory: str,
    category: str,
    exclude_sku: str,
    n_results: int = TOP_K_VECTOR,
) -> list[dict]:
    """Search for similar products filtered by supercategory and category."""
    client = _get_client()
    collection = client.get_collection(CHROMA_COLLECTION)

    # Build filter: must match supercategory AND category, exclude the source SKU
    where_filter = {
        "$and": [
            {"supercategory": {"$eq": supercategory}},
            {"category": {"$eq": category}},
        ]
    }

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results + 1,  # +1 in case source product is in results
        where=where_filter,
    )

    candidates = []
    if results and results["ids"] and results["ids"][0]:
        for idx, sku in enumerate(results["ids"][0]):
            if sku == exclude_sku:
                continue
            candidates.append(
                {
                    "sku": sku,
                    "distance": results["distances"][0][idx] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][idx] if results.get("metadatas") else {},
                    "document": results["documents"][0][idx] if results.get("documents") else "",
                }
            )

    return candidates[:n_results]


if __name__ == "__main__":
    print("Starting product indexing...")
    index_products()
