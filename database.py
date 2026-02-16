import pymysql
from contextlib import contextmanager
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB


@contextmanager
def get_connection():
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        yield conn
    finally:
        conn.close()


def get_product_by_sku(sku: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.*, c.supercategory, c.category, c.class
                FROM products p
                LEFT JOIN categories c ON p.sku = c.sku
                WHERE p.sku = %s
                LIMIT 1
                """,
                (sku,),
            )
            return cur.fetchone()


def get_product_bullets(sku: str) -> list[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bullet_text FROM product_bullets WHERE sku = %s ORDER BY display_order",
                (sku,),
            )
            return [row["bullet_text"] for row in cur.fetchall()]


def get_product_specs(sku: str) -> dict[str, str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sn.name, ps.spec_value
                FROM product_specifications ps
                JOIN specification_names sn ON ps.spec_name_id = sn.id
                WHERE ps.sku = %s
                """,
                (sku,),
            )
            return {row["name"]: row["spec_value"] for row in cur.fetchall()}


def get_all_products_for_indexing() -> list[dict]:
    """Fetch all active products with their categories, specs, and bullets for ChromaDB indexing."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.sku, p.name, p.short_name, p.brand_name, p.description,
                       p.web_price, p.customer_price, p.uom, p.uom_qty,
                       p.manufacturer_name, p.review_rating, p.review_count,
                       c.supercategory, c.category, c.class
                FROM products p
                LEFT JOIN categories c ON p.sku = c.sku
                WHERE p.active = 1
                """
            )
            products = cur.fetchall()

    # Batch-fetch bullets and specs
    skus = [p["sku"] for p in products]
    if not skus:
        return []

    with get_connection() as conn:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(skus))

            cur.execute(
                f"SELECT sku, bullet_text FROM product_bullets WHERE sku IN ({placeholders}) ORDER BY sku, display_order",
                skus,
            )
            bullets_map: dict[str, list[str]] = {}
            for row in cur.fetchall():
                bullets_map.setdefault(row["sku"], []).append(row["bullet_text"])

            cur.execute(
                f"""
                SELECT ps.sku, sn.name, ps.spec_value
                FROM product_specifications ps
                JOIN specification_names sn ON ps.spec_name_id = sn.id
                WHERE ps.sku IN ({placeholders})
                """,
                skus,
            )
            specs_map: dict[str, dict[str, str]] = {}
            for row in cur.fetchall():
                specs_map.setdefault(row["sku"], {})[row["name"]] = row["spec_value"]

    for p in products:
        p["bullets"] = bullets_map.get(p["sku"], [])
        p["specs"] = specs_map.get(p["sku"], {})

    return products


def get_categories() -> list[dict]:
    """Fetch all distinct supercategory + category pairs."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT supercategory, category FROM categories ORDER BY supercategory, category"
            )
            return cur.fetchall()


def get_products_by_skus(skus: list[str]) -> list[dict]:
    """Fetch full product details for a list of SKUs."""
    if not skus:
        return []
    with get_connection() as conn:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(skus))
            cur.execute(
                f"""
                SELECT p.*, c.supercategory, c.category, c.class
                FROM products p
                LEFT JOIN categories c ON p.sku = c.sku
                WHERE p.sku IN ({placeholders})
                """,
                skus,
            )
            products = {row["sku"]: row for row in cur.fetchall()}

    # Preserve input order
    result = []
    for sku in skus:
        if sku in products:
            result.append(products[sku])
    return result
