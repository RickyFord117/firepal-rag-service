# Firepal RAG Service

This repository contains the backend API for the Firepal chatbot. It is a Python application built with the FastAPI framework and designed to be deployed as a containerized service on Google Cloud Run.

The service provides a conversational interface for answering fire safety questions. It uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, source-based answers.

## Tech Stack

- **Web Framework:** FastAPI
- **Hosting:** Google Cloud Run
- **Vector Database:** Qdrant
- **Language Models:** OpenAI
- **Relational Database:** SQLite (for temporary/cloud storage) & SQLAlchemy

## Configuration

This service requires several environment variables to be set for it to function correctly, including API keys for external services. These are managed via a .env file for local development or set directly in the Cloud Run service configuration.

To get the required API keys and configuration values, please reach out to the repository owner.

## Deployment

This service is deployed as a container to Google Cloud Run. Ensure you have the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated (`gcloud auth login`).

To deploy the service, run the following command from the root of the project directory:

```bash
gcloud run deploy fastapi-service --source . --region=us-central1 --allow-unauthenticated
```
