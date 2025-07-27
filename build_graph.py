import os
import json
import logging
from dotenv import load_dotenv
from pathlib import Path
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from neo4j.debug import watch

from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def process_record(record, file_path, graph, transformer):
    """
    Processes a single JSON record (dictionary) to extract content from the
    'title' and 'content' keys, transform it, and load it into Neo4j.
    """
    # Extract title and content based on the provided JSON structure
    title = record.get("title", "").strip()
    content_list = record.get(
        "content",
    )

    # Ensure content_list is actually a list before joining
    if isinstance(content_list, list):
        # Join the list of paragraphs into a single string
        full_text = "\n".join(content_list)
    else:
        full_text = ""

    # Combine title and content for a comprehensive document
    combined_content = f"{title}\n\n{full_text}".strip()

    if combined_content:
        try:
            # Create a LangChain Document from the combined text
            doc = Document(page_content=combined_content)
            # Use the transformer to convert the document into graph-ready format
            graph_documents = transformer.convert_to_graph_documents([doc])
            # Add the transformed documents to the graph
            graph.add_graph_documents(graph_documents)
            logging.info(
                f"  Successfully processed and added a record from {file_path}."
            )
        except Exception as e:
            logging.error(f"  Failed to process a record from {file_path}: {e}")
    else:
        logging.warning(
            f"  No 'title' or 'content' found in a record within {file_path}. Skipping record."
        )


def build_knowledge_graph():
    """
    Reads JSON files from the data/safety directory, extracts text content,
    and uses an LLM to build a knowledge graph in Neo4j.
    """
    # 1. Define the data directory
    # This correctly points to the 'data/safety' folder from the project root.
    data_dir = Path(__file__).parent / "data" / "safety"
    if not data_dir.exists():
        logging.error(f"Data directory not found at: {data_dir}")
        return

    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        logging.error(f"No JSON files found in: {data_dir}")
        return

    logging.info(f"Found {len(json_files)} JSON files to process.")

    # 2. Initialize connections
    logging.info("Performing a direct driver pre-connection test...")
    try:
        # with watch("neo4j"): - for detailed logs
        with GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        ) as driver:
            driver.verify_connectivity()
            logging.info(
                "Direct driver connection successful. Proceeding with LangChain."
            )
    except ServiceUnavailable as e:
        logging.error(f"Direct connection test failed: {e}")
        logging.error(f"Could not connect to the Aura database at '{NEO4J_URI}'.")
        logging.error(
            "This often happens if the database is asleep or a firewall is blocking the connection."
        )
        logging.error("Please perform the following checks:")
        logging.error(
            "1. Go to your Neo4j Aura console and ensure the database status is 'Running'."
        )
        logging.error("2. If it's sleeping, wake it up and try again in a minute.")
        logging.error(
            "3. Check for any local or network firewalls/VPNs that might be blocking outbound connections."
        )
        return
    except AuthError as e:
        logging.error(
            f"Direct connection test failed with an authentication error: {e}"
        )
        logging.error(
            "Please double-check your NEO4J_USERNAME and NEO4J_PASSWORD in the .env file."
        )
        return
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during the direct connection test: {e}"
        )
        return

    try:
        logging.info("Connecting to Neo4j database...")
        graph = Neo4jGraph(
            url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
        )
        logging.info(graph)
        # Clear the existing graph to start fresh (optional)
        graph.query("MATCH (n) DETACH DELETE n")
        logging.info("Cleared existing graph data.")

        logging.info
        llm = ChatOpenAI(temperature=0, model=LLM_MODEL, api_key=OPENAI_API_KEY)

        # This transformer is the core of the process. It uses the LLM to
        # convert text into graph nodes and relationships.
        llm_transformer = LLMGraphTransformer(llm=llm)
        logging.info(
            "Successfully initialized Neo4j, OpenAI, and LangChain transformer."
        )

    except Exception as e:
        logging.error(f"Failed to initialize connections: {e}")
        return

    # 3. Process each JSON file
    for file_path in json_files:
        try:
            logging.info(f"Processing file: {file_path.name}...")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both single JSON objects and lists of objects
            if isinstance(data, list):
                logging.info(
                    f"  Detected a list of objects. Iterating through {len(data)} records."
                )
                for record in data:
                    if isinstance(record, dict):
                        process_record(record, file_path, graph, llm_transformer)
                    else:
                        logging.warning(
                            f"  Item in {file_path} is not a valid object. Skipping."
                        )
            elif isinstance(data, dict):
                logging.info("  Detected a single object.")
                process_record(data, file_path, graph, llm_transformer)
            else:
                logging.warning(
                    f"  {file_path} contains an unexpected data type. Skipping."
                )

        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while processing {file_path}: {e}"
            )

    logging.info("Refreshing graph schema...")
    graph.refresh_schema()
    logging.info("Knowledge graph build process complete.")


if __name__ == "__main__":
    # This makes the script runnable from the command line with `python build_graph.py`
    build_knowledge_graph()
