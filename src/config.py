import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- OpenAI and Qdrant Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Model and Collection Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1"
COLLECTION_NAME = "firepal_lfb_v1"
VECTOR_SIZE = 1536

# --- SQL Database config ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./polidex.db")
