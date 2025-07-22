import os
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from src.database import SessionLocal, engine, Base
from src.models import WebContentChunk
from src.config import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    VECTOR_SIZE,
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def clear_and_create_db_and_tables():
    """Drops and recreates the web_content_chunks table."""
    try:
        print("Clearing and recreating database tables...")
        Base.metadata.drop_all(bind=engine, tables=[WebContentChunk.__table__])
        Base.metadata.create_all(bind=engine)
        print("Database and tables are now clean and ready.")
    except Exception as e:
        print(f"Error during table recreation: {e}")
        raise


def recreate_qdrant_collection():
    """Recreates the Qdrant collection to ensure it's empty and configured correctly."""
    try:
        print(f"Recreating Qdrant collection '{COLLECTION_NAME}'...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
        print("Qdrant collection is ready.")
        return True
    except Exception as e:
        print(f"Error recreating Qdrant collection: {e}")
        return False


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of text strings."""
    if not texts:
        return []
    try:
        response = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []


def process_and_ingest_data():
    """Loads data, creates embeddings, and populates both SQL and Qdrant databases."""
    data_dir = Path(__file__).parent.parent / "data" / "safety"
    if not data_dir.is_dir():
        print(
            f"Error: Data directory not found at {data_dir}. Please create it and add your JSON files."
        )
        return

    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}.")
        return

    print(f"Found {len(json_files)} JSON files to process.")

    all_chunks_to_process = []

    # Helper function to process a single document object (a dictionary)
    def process_document_object(doc_data):
        source_url = doc_data.get("url", "")
        source_title = doc_data.get("title", "").strip()

        for section in doc_data.get("content_sections", []):
            heading = section.get("heading")

            for p_text in section.get("paragraphs", []):
                if p_text and p_text.strip():
                    all_chunks_to_process.append(
                        {
                            "text_content": p_text,
                            "source_url": source_url,
                            "source_title": source_title,
                            "heading": heading,
                            "chunk_type": "paragraph",
                        }
                    )

            for lst in section.get("lists", []):
                for li_text in lst.get("items", []):
                    if li_text and li_text.strip():
                        all_chunks_to_process.append(
                            {
                                "text_content": li_text,
                                "source_url": source_url,
                                "source_title": source_title,
                                "heading": heading,
                                "chunk_type": "list_item",
                            }
                        )

    for file_path in json_files:
        print(f"--- Processing file: {file_path.name} ---")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            process_document_object(data)
        elif isinstance(data, list):
            print(
                f"File {file_path.name} contains a list. Processing {len(data)} items."
            )
            for doc_item in data:
                if isinstance(doc_item, dict):
                    process_document_object(doc_item)
                else:
                    print(
                        f"Warning: Found a non-dictionary item in the list within {file_path.name}. Skipping."
                    )
        else:
            print(
                f"Warning: Root of JSON file {file_path.name} is not a dictionary or list. Skipping."
            )

    if not all_chunks_to_process:
        print("No text content found to ingest across all files.")
        return

    print(f"Extracted a total of {len(all_chunks_to_process)} text chunks to ingest.")

    db: Session = SessionLocal()
    try:
        batch_size = 100
        # Use a consistent variable name for clarity
        list_of_chunks = all_chunks_to_process
        for i in range(0, len(list_of_chunks), batch_size):
            batch = list_of_chunks[i : i + batch_size]
            print(
                f"Processing batch {i//batch_size + 1}/{(len(list_of_chunks) + batch_size - 1)//batch_size}..."
            )

            texts_to_embed = [chunk["text_content"] for chunk in batch]
            embeddings = generate_embeddings(texts_to_embed)

            if not embeddings or len(embeddings) != len(batch):
                print("Skipping batch due to embedding generation failure.")
                continue

            sql_objects_to_add = []
            qdrant_points_to_add = []

            for j, chunk_data in enumerate(batch):
                chunk_id = str(uuid.uuid4())

                sql_object = WebContentChunk(
                    chunk_id=chunk_id,
                    text_content=chunk_data["text_content"],
                    source_url=chunk_data["source_url"],
                    source_title=chunk_data["source_title"],
                    heading=chunk_data["heading"],
                    chunk_type=chunk_data["chunk_type"],
                )
                sql_objects_to_add.append(sql_object)

                qdrant_payload = {
                    "sql_db_chunk_id": chunk_id,
                    "source_url": chunk_data["source_url"],
                    "source_title": chunk_data["source_title"],
                    "heading": chunk_data["heading"],
                    "text_snippet": chunk_data["text_content"][:200],
                }

                qdrant_points_to_add.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embeddings[j],
                        payload={
                            k: v for k, v in qdrant_payload.items() if v is not None
                        },
                    )
                )

            db.add_all(sql_objects_to_add)
            db.commit()
            print(
                f"Successfully committed {len(sql_objects_to_add)} records to SQL database."
            )

            if qdrant_points_to_add:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=qdrant_points_to_add,
                    wait=True,
                )
                print(
                    f"Successfully upserted {len(qdrant_points_to_add)} points to Qdrant."
                )

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
        db.rollback()
    finally:
        db.close()

    print("Data ingestion complete.")


if __name__ == "__main__":
    print("Starting Firepal data ingestion process...")
    # 1. Clear and create the SQL database tables
    clear_and_create_db_and_tables()

    # 2. Clear and create the Qdrant collection
    if recreate_qdrant_collection():
        # 3. If setup is successful, ingest the data
        process_and_ingest_data()
    else:
        print("Aborting ingestion due to Qdrant collection setup failure.")
