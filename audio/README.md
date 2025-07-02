# RAG2 Application

## Overview

The `audio` folder contains an application that implements Retrieval-Augmented Generation (RAG) using LangChain. This app combines language models with external data sources to save interactions to have context for future conversations. This application will receive audio input and then retrieves the information and then provides the output in audio. 

TTS processing is still being done. The current output is very long and rudimentary. LLM response is fed into text to speech conversion. Needs more work. 

## Features


- **Question Answering:** Answer user queries using both language models 
- **Customizable Pipelines:** Easily modify retrieval and generation components.
- **Extensible:** Integrate with various vector stores and LLM providers.


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

