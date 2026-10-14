# Knowledge Graph-Enhanced RAG (GraphRAG)

## 📌 1. Project Overview
This project implements a Knowledge Graph-Enhanced Retrieval-Augmented Generation (GraphRAG) system designed to move beyond simple semantic chunk search. By combining unstructured raw text vector search with structured Neo4j graph traversal and local LLM generation (via Ollama Llama 3.2), the system accurately answers complex, multi-hop queries. It also includes an evaluation pipeline using the RAGAS framework to quantitatively compare Vanilla Vector Search against GraphRAG.

---
## ⚙️ 2. Setup And Installation Guide
1. Ensure you have Python installed, along with local **Neo4j** and **Ollama** servers running.
2. Install the required dependencies using the `requirements.txt` file:
```bash
   pip install -r requirements.txt

```

3. Run the data ingestion pipeline to extract triples and save them to Neo4j:
```bash
python ingestion.py

```


4. Execute the hybrid query engine to perform vector retrieval and graph traversal:
```bash
python hybrid_engine.py

```


5. Run the RAGAS evaluation framework to compare performance:
```bash
python evaluate_ragas.py

```


---
## 🏗️ 3. Project Architecture

The project architecture consists of three core integrated modules:

* **Data Ingestion Pipeline (`ingestion.py`)**: Parses the text corpus, connects to Ollama (`llama3.2`) in JSON mode to extract Knowledge Graph triples (Subject, Predicate, Object), and automatically persists them into the Neo4j database using dynamic `MERGE` queries.
* **Hybrid Retrieval Engine (`hybrid_engine.py`)**: Simultaneously executes raw text vector search using HuggingFace embeddings (`all-MiniLM-L6-v2`) with cosine similarity scoring, alongside multi-hop Neo4j graph traversal, combining both contexts to generate precise answers via Llama 3.2.
* **Evaluation Framework (`evaluate_ragas.py`)**: Utilizes the RAGAS library and local LLM/embeddings to evaluate both Vanilla RAG and GraphRAG pipelines based on key metrics (faithfulness, answer relevancy, and context precision) along with execution latency tracking.

---
## 💻 4. Tech Stack & Components

* **Programming Language**: Python

* **Framework**: LangChain (langchain_community)

* **Graph Database**: Neo4j

* **LLM & Runtime**: Llama 3.2 (via local Ollama instance)

* **Embedding Model**: HuggingFace Sentence-Transformers (all-MiniLM-L6-v2)

* **Evaluation Framework**: RAGAS

* **Mathematical & Data Processing Library**: NumPy

---
## 📁 5. Folder Structure

```text
GraphRAG-Project/
│
├── .vscode/
│   └── settings.json
├── .gitignore
├── ingestion.py
├── hybrid_engine.py
├── evaluate_ragas.py
└── requirements.txt

```
---
## 🌐 6. Database Schema

The project uses **Neo4j** as a graph database structured as follows:

* **Nodes**: `:Entity` nodes representing extracted entities with a unique `name` property (e.g., "Microsoft", "GitHub", "San Francisco").
* **Relationships**: Dynamic, directional relationship types extracted from text and normalized to uppercase with underscores (e.g., `-[:FOUNDED_BY]->`, `-[:ACQUIRED]->`, `-[:HEADQUARTERED_IN]->`).


---
## ✨ 7. Features & Enhancements

* **Hybrid Context Retrieval**: Successfully merges unstructured cosine similarity vector search with structured graph path traversal for robust context building.
* **Offline Local Execution**: Powered entirely offline using local Ollama (`llama3.2`) and HuggingFace sentence-transformers (`all-MiniLM-L6-v2`).
* **Robust Connectivity & Retries**: Built-in retry mechanisms for stable database connections and API handling.
* **Automated RAGAS Evaluation**: Automated measurement of **Faithfulness**, **Answer Relevancy**, and **Context Precision** metrics with detailed Pandas comparison outputs and latency reports.

---
## 📈 8. Future Improvements

* Scaling data ingestion to handle batch processing of multiple larger documents and unstructured files.
* Integrating advanced reranking techniques (such as Reciprocal Rank Fusion) for optimized hybrid chunk ordering.
* Expanding graph traversal depth to support more complex multi-hop reasoning paths.

---


