# RAG2 Application

## Overview

The `rag2` folder contains an application that implements Retrieval-Augmented Generation (RAG) using LangChain. This app combines language models with external data sources to provide accurate and context-aware responses.

## Features

- **Document Ingestion:** Load and index documents for retrieval.
- **Question Answering:** Answer user queries using both language models and retrieved documents.
- **Customizable Pipelines:** Easily modify retrieval and generation components.
- **Extensible:** Integrate with various vector stores and LLM providers.

## SQL Agent Integration

- **SQL Agent:** Integrates a LangChain SQL Agent for natural language querying of SQL databases.
- **Database Support:** Compatible with popular databases such as SQLite, PostgreSQL, and MySQL.
- **Seamless Querying:** Users can ask questions in plain English, and the agent translates them into SQL queries.
- **Secure Access:** Connection details and credentials are managed via environment variables for security.
- **Extensible Logic:** Easily customize the agent to support additional SQL dialects or advanced query logic.

## Getting Started

1. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure Environment**
    - Set API keys and environment variables as needed in a `.env` file.

3. **Run the Application**
    ```bash
    python chatapp.py
    ```
    - Alternatively use the run script

## Folder Structure

- `chatapp.py` - Main application entry point.
- `ingest.py` - Scripts for document ingestion.
- `config/` - Configuration files.
- `data/` - Source documents and datasets.

## Usage

- Ingest documents:
  ```bash
  python ingest.py --source data/
  ```
- Start the app and interact via the provided interface or API.

## License

See [LICENSE](./LICENSE) for details.

