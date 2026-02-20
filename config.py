import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3307))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "staples_data")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-pro"          # Final reasoning / fallback
GEMINI_RANKING_MODEL = os.getenv("GEMINI_RANKING_MODEL", "gemini-2.5-flash")  # Fast ranking step

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "products"

TOP_K_VECTOR = 25   # Candidates fetched from Qdrant and sent to Gemini
TOP_K_FINAL = 5

MAX_REQUESTS_PER_USER = int(os.getenv("MAX_REQUESTS_PER_USER", 10))

CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # seconds; 0 = disable
