# **Metro Hospitals HR Policy Assistant**

This repository contains a Streamlit-based **Retrieval-Augmented Generation (RAG)** chatbot designed for the **Metro Group of Hospitals HR Policy** document.
The assistant allows HR staff and employees to converse with an LLM that references the existing **ChromaDB knowledge base** containing HR policies.
Optionally, the chatbot can also call **Tavily Web Search** for up-to-date external information.

---

## **Repository Layout**

| Path                   | Description                                                                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`app.py`**           | Main Streamlit application. Provides navigation, chat UI, response modes, RAG/web-search toggles, and orchestrates calls into the LLM, embedding model, vector store, and search utilities. |
| **`process_pdf.py`**   | One-time preprocessing script that ingests the Metro Hospitals HR Policy PDF into the persistent Chroma vector store via chunking + embeddings.                                             |
| **`requirements.txt`** | Python dependency list required for running the app (Streamlit, LangChain, ChromaDB, Tavily, Groq SDK, etc.).                                                                               |
| **`.env`**             | Stores environment variables such as API keys (Groq, Jina, Tavily). This file is ignored by Git.                                                                                            |
| **`.gitignore`**       | Ensures logs, environment files, caches, and DB artifacts are excluded from version control.                                                                                                |
| **`.devcontainer/`**   | Optional VS Code Dev Container configuration for consistent development environments.                                                                                                       |
| **`chroma_db/`**       | The persistent Chroma vector store. Generated after running `process_pdf.py`. Essential for RAG functionality.                                                                              |
| **`logs/`**            | Directory where the application writes runtime logs (via `utils.logger`). Auto-created at runtime.                                                                                          |
| **`temp_docs/`**       | Temporary holding area for processed documents during ingestion (used by utilities).                                                                                                        |

---

## **Configuration**

### `config/config.py`

Centralized configuration module used to:

* Load API keys (Groq, Jina, Tavily)
* Provide safe getters for secrets
* Abstract away direct reliance on Streamlit’s `st.secrets`

---

## **Models**

| File                       | Description                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **`models/embeddings.py`** | Wrapper for initializing the free **Jina Embeddings** model. Handles API key validation and logging.                              |
| **`models/llm.py`**        | Wrapper for initializing the **Groq LLM**. Handles model selection, API key setup, and returning a unified `.invoke()` interface. |

---

## 🛠 **Utilities**

| File                              | Description                                                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **`utils/document_loader.py`**    | Loads PDF/TXT/CSV/DOC/DOCX documents into raw text. Handles directory-based ingestion.                                       |
| **`utils/document_processor.py`** | Manages chunking (word/sentence-based), metadata handling, and preparing text chunks for embedding and vector storage.       |
| **`utils/text_splitter.py`**      | Customizable text splitter built on LangChain's recursive splitter with chunk overlap and token/word control.                |
| **`utils/vector_store.py`**       | Full ChromaDB integration: create/load stores, add embeddings, migration, dimension checks, and similarity search retrieval. |
| **`utils/web_search.py`**         | Tavily search integration. Contains logic for when to use external search and helpers for formatting results.                |
| **`utils/logger.py`**             | Unified logger for the entire app. Logs activity & errors to `logs/chatbot.log`.                                             |

---

## **How Components Work Together**

### **Document Ingestion (One-Time Setup)**

Performed via `process_pdf.py`:

* Loads the Metro Hospitals HR Policy PDF
* Splits text into overlapping chunks
* Embeds chunks using Jina embeddings
* Stores them into ChromaDB
* Generates persistent vector store (`chroma_db/`)

### **Runtime Chat Logic (`app.py`)**

* Loads Chroma vector store
* Initializes Jina embeddings + Groq LLM
* Retrieves top-k relevant chunks
* Optionally runs Tavily web search
* Injects combined context into system prompt
* Returns an accurate HR policy answer

### **Config & Logging**

* `config/` manages secrets
* `utils/logger.py` writes logs for debugging and audit trails

---

## **Getting Started**

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your `.env`

```
GROQ_API_KEY=your_key
JINA_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 3. Build the knowledge base (only once)

```bash
python process_pdf.py
```

This generates the `./chroma_db` directory.

### 4. Start the chatbot

```bash
streamlit run app.py
```

### 5. Visit in browser

```
http://localhost:8501
```

---

## **Summary**

This project is a complete **RAG-powered HR Policy Assistant** tailored for **Metro Group of Hospitals**, using:

* **Streamlit** for the UI
* **Groq LLaMA models** for the LLM
* **Jina embeddings** for vector generation
* **ChromaDB** for policy retrieval
* **Tavily** for optional web search

The assistant delivers accurate, contextual HR policy answers to employees and HR teams.
