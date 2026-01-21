
# RAG with Reranking 🔍📚

A Python project that demonstrates **Retrieval-Augmented Generation (RAG)** with an extra **reranking step in the ReAct phase**, and records responses at the chunk level. The core idea is to improve the relevance of retrieved context before final generation by using reranking. 

---

## 🧠 Overview

**RAG (Retrieval-Augmented Generation)** combines information retrieval with large language model generation — first retrieving relevant documents based on similarity to a query, then generating an answer rooted in that context. Reranking is an optional but powerful stage that **reorders the retrieved chunks by relevance** before they’re passed to the LLM, helping the model produce more accurate outputs. 

This project demonstrates:
- A RAG pipeline enhanced with **reranking in the ReAct loop**
- Tracking and storing responses at the chunk level for analysis
- A simple Python interface (script + optional notebook)

---

## 🚀 Key Features

✔️ Retrieve documents or text chunks for a query  
✔ Rerank retrieved results based on relevance scoring  
✔ ReAct integration (Retrieve → Act → Generate)  
✔ Logs or records model responses per chunk  
✔ Useful for experiments and benchmarking

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Retrieval | Vector search or similarity search |
| Reranking | Custom scoring or model-based reranker |
| Notebook | Jupyter Notebook (analysis & demos) |
| Dependencies | See `requirements.txt` |

---

## 📥 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/sisira214/RAG_with_reranking.git
   cd RAG_with_reranking


2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS/Linux
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 How It Works

1. **Retrieval:** A retriever fetches the top K most relevant text chunks for a given query (e.g., using embeddings or similarity metrics).
2. **Reranking:** Instead of passing these top K directly to generation, a reranker sort/reranks them according to finer relevance criteria (semantic scores or model-based ranking). ([DEV Community][1])
3. **ReAct Phase:** The pipeline uses the reranked chunks in a ReAct (Reason + Act) strategy — generating better grounded answers.
4. **Record Outputs:** Each chunk’s contribution and model response are logged or recorded, enabling analysis of what helped the model answer best.

---

## 🧪 Usage

### Run script

```bash
python SearchAgent.py
```

This runs the RAG pipeline with reranking and prints or logs results.

### Interactive Notebook

Open `searchAgent.ipynb` in Jupyter to see step-by-step extraction, reranking, and response recording, with visual outputs.

---

## 📁 Recommended Structure

```
RAG_with_reranking/
├── .gitignore
├── LICENSE
├── README.md
├── SearchAgent.py         # Core retrieval + reranking logic
├── SearchApp.py          # Example runner or interface
├── searchAgent.ipynb     # Notebook for demos / step-through
└── requirements.txt      # Python dependencies
```

---

## 🧠 Why Reranking Matters

In RAG systems, initial retrieval based on embeddings or simple similarity can return *close but not most relevant* chunks. Reranking rescues this by sorting candidates with deeper semantic relevance, leading to better generation quality — a common improvement strategy in advanced RAG pipelines. ([DEV Community][1])

---

## 🤝 Contributing

Contributions are welcome! You can help by:

* Adding more sophisticated reranker models (e.g., cross-encoders)
* Integrating external datasets for retrieval
* Adding metrics for evaluation (like precision, recall)

---

## 📜 License

This project is licensed under the **MIT License** — see `LICENSE` for details. 

```

