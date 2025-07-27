# 🧠 RAG Chatbot – Weekend Demo Project

Welcome to my demo weekend project!  
This is a lightweight implementation of a **Retrieval-Augmented Generation (RAG)** system using **LangChain**, **FAISS**, and **Groq's LLaMA-4 model**.  
The purpose of this project is to **demonstrate how RAG works**: fetching relevant document snippets and generating human-like responses using LLMs.

---

## 📘 What is RAG?

**RAG (Retrieval-Augmented Generation)** combines:
- 🔍 **Information retrieval** – finds relevant documents based on your query
- 🧠 **Language generation** – uses a powerful LLM to generate an answer based on those documents

This allows the chatbot to answer questions not just from its internal memory, but from **external, up-to-date knowledge**.

---

## 📁 Project Structure
```bash
.
├── data/
│   ├── KBI/             # Input folder for raw documents
│   └── FinalKB/         # Processed documents moved here after ingestion
├── faiss_index/         # Folder where FAISS index files are saved
├── utils/
│   ├── embedding_store.py  # Handles document loading, embedding, and indexing
│   └── rag_chain.py        # RAG chain orchestration (retrieval + LLM)
├── .env                 # Store your GROQ API key here
├── main.py              # Entry point: CLI to chat or ingest documents
└── requirements.txt

```

---

> **📝 Note:** This project was built using **Python 3.13.2**.  

---

## 🚀 How to Use

### 1. Create & activate  Virtual Environment 
```bash
python -m venv venv 

venv\scripts\activate
```



### 2. Install dependencies
```bash
pip install -r requirements.txt
```


### 3. Set your Groq API key
Create a .env file in the root directory and add:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add Documents
Place your .txt files into the data/KBI/ folder.


### 5. Run the App
```bash
python app.py
```


---

## 💬 Available Commands in Console

- ❓ **Ask questions directly** – Chat with the bot using natural language.
- 📄 **Type `ingest_doc`** – Ingest and embed new `.txt` files from the `data/KBI/` folder.
- ❌ **Type `esc`** – Exit the chatbot application.

