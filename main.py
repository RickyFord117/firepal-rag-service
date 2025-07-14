import os
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from qdrant_client import QdrantClient
from openai import OpenAI
from sqlalchemy.orm import Session
from src import models, database
from src.config import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    VECTOR_SIZE,
    LLM_MODEL,
)

database.Base.metadata.create_all(bind=database.engine)


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    ai_response: Dict[str, Any]


openai_client_instance: Optional[OpenAI] = None
qdrant_client_instance: Optional[QdrantClient] = None


def get_openai_client() -> OpenAI:
    """Dependency to get a singleton OpenAI client instance."""
    global openai_client_instance
    if openai_client_instance is None:
        logging.info("--- Initializing OpenAI client ---")
        openai_client_instance = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client_instance


def get_qdrant_client() -> QdrantClient:
    """Dependency to get a singleton Qdrant client instance."""
    global qdrant_client_instance
    if qdrant_client_instance is None:
        logging.info("--- Initializing Qdrant client ---")
        qdrant_client_instance = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return qdrant_client_instance


app = FastAPI()

# Add your CORS middleware settings.
# This will allow your deployed frontend to communicate with this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://polidex-ecc7b.web.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_embedding(text: str, openai_client: OpenAI) -> List[float]:
    """Generates an embedding for a single text string."""
    try:
        response = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to generate text embedding."
        )


def search_web_content(query: str, qdrant_client: QdrantClient, openai_client: OpenAI):
    """
    Searches the Qdrant vector database for web content chunks relevant to a query.
    """
    logging.info(f"--- TOOL: Executing search_web_content(query='{query}') ---")
    query_vector = get_embedding(query, openai_client)
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
    )
    return [result.payload for result in search_results]


def fetch_web_content(ids: List[str]):
    """
    Fetches the full text content for a list of chunk IDs from the SQL database.
    """
    logging.info(f"--- TOOL: Executing fetch_web_content(ids={ids}) ---")
    db: Session = next(get_db())
    # --- FIX: Query the correct WebContentChunk model ---
    fetched_chunks = (
        db.query(models.WebContentChunk)
        .filter(models.WebContentChunk.chunk_id.in_(ids))
        .all()
    )
    return {chunk.chunk_id: chunk.text_content for chunk in fetched_chunks}


@app.get("/")
def read_root():
    """A simple root endpoint to confirm the service is running."""
    return {"Hello": "From Cloud Run"}


@app.get("/api/v1/health")
def health_check():
    """The health check endpoint for your frontend to call."""
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    openai_client: OpenAI = Depends(get_openai_client),
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
):
    """
    Orchestrates the multi-step Function Calling conversation with the LLM for the Firepal chatbot.
    """
    logging.info(f"Received chat request with query: '{request.query}'")

    tools = [
        {
            "type": "function",
            "name": "search_web_content",
            "description": "Searches the London Fire Brigade safety website content to find information relevant to a user's query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query to search for.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "type": "function",
            "name": "fetch_web_content",
            "description": "Retrieves the full text for specific content IDs found via the search tool.",
            "parameters": {
                "type": "object",
                "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["ids"],
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": "You are Firepal, a helpful AI assistant from the London Fire Brigade. Your goal is to answer public fire-safety questions clearly and concisely. First, use the `search_web_content` tool to find relevant information. Then, use the `fetch_web_content` tool to get the full text. Finally, synthesize the fetched content into a helpful answer. Always cite the source URL for the information you provide. If you cannot find a relevant answer in the provided tools, you must state that you cannot answer and recommend the user contact the BDO or visit the LFB website for more information.",
        },
        {"role": "user", "content": request.query},
    ]

    for _ in range(5):
        response = openai_client.responses.create(
            model=LLM_MODEL,
            input=messages,
            tools=tools,
        )
        tool_calls = response.output

        if not response.output_text and tool_calls:
            messages.append(tool_calls[0].model_dump())
            available_functions = {
                "search_web_content": search_web_content,
                "fetch_web_content": fetch_web_content,
            }
            for tool_call in tool_calls:
                logging.info(tool_call)
                function_name = tool_call.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.arguments)

                # Execute the function with the provided arguments
                if function_name == "search_web_content":
                    function_response = function_to_call(
                        qdrant_client=qdrant_client,
                        openai_client=openai_client,
                        **function_args,
                    )
                elif function_name == "fetch_web_content":
                    function_response = function_to_call(**function_args)
                else:
                    continue
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": str(function_response),
                    }
                )
        else:
            logging.info("LLM returned final response.")
            final_response = {
                "id": str(uuid.uuid4()),
                "type": "ai",
                "text": response.output_text,
                "sources": [],  # We will implement source extraction from the response text next
                "timestamp": str(datetime.utcnow()),
            }
            return ChatResponse(ai_response=final_response)

    raise HTTPException(
        status_code=500, detail="AI assistant failed to generate a final response."
    )
