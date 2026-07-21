# GigaCorp Customer Support RAG Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

&gt; An intelligent customer support assistant powered by **Retrieval-Augmented Generation (RAG)** with **conversational memory** and **source citations**. Built with LangChain, Groq LLM, and local HuggingFace embeddings.

![App Screenshot](assignment-1/docs/Screenshot (5270).png)

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🔍 **RAG Retrieval** | Answers questions using a local vector store (FAISS) built from GigaCorp's FAQ |
| 💬 **Conversational Memory** | Remembers context across multi-turn conversations (e.g., "Do you ship to India?" → "How much does it cost?") |
| 📚 **Source Citations** | Every answer includes references to the exact policy document and section |
| 🚀 **Fast & Free** | Uses Groq's `openai/gpt-oss-120b` (500 tok/s) + local embeddings — zero API costs |
| 🔒 **Privacy-First** | All embeddings computed locally; no customer data sent to external APIs |
| 🎨 **Clean Web UI** | Professional Streamlit interface with chat history and agent stats |

---

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI                         │
│  (app.py) - Chat interface, message history, source display     │
└──────────────────────────────┬──────────────────────────────────┘
│
┌──────────────────────────────▼──────────────────────────────────┐
│                    GigaCorpSupportAgent                         │
│  (src/agents/support_agent.py) - Main orchestrator              │
└──────────────┬───────────────────────────────┬──────────────────┘
│                               │
┌──────────▼──────────┐      ┌───────────▼────────────┐
│   RAG Chain         │      │  Conversation Memory   │
│  (src/chains/)      │      │  (src/memory/)         │
│  • Retrieval        │      │  • Buffer Window (k=5) │
│  • Prompt Building  │      │  • Session Management  │
│  • Groq LLM Call    │      │  • Context Injection   │
└──────────┬──────────┘      └────────────────────────┘
│
┌──────────▼──────────────────────────────────────────────┐
│              FAISS Vector Store                         │
│  (src/retrieval/vector_store.py)                        │
│  • Document Loading (Markdown FAQ)                      │
│  • Semantic Chunking (MarkdownHeaderTextSplitter)       │
│  • HuggingFace Embeddings (all-MiniLM-L3-v2)            │
│  • Similarity Search with Relevance Scores              │
└─────────────────────────────────────────────────────────┘


### Data Flow

1. **User asks question** → Streamlit UI captures input
2. **Retrieve context** → FAISS searches for top-k relevant FAQ chunks
3. **Inject memory** → Previous conversation turns added to prompt
4. **Build prompt** → System instructions + context + history + question
5. **Generate answer** → Groq LLM (`openai/gpt-oss-120b`) produces response
6. **Store memory** → Q&A exchange saved for future context
7. **Display** → Answer + source citations shown in UI

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|------------|------|
| **LLM** | Groq `openai/gpt-oss-120b` | Free tier, 500 tok/s, 131K context |
| **Embeddings** | `sentence-transformers/paraphrase-MiniLM-L3-v2` | Local, free, 384-dim, perfect for small KBs |
| **Vector Store** | FAISS (CPU) | Fast similarity search, persists to disk |
| **Framework** | LangChain + LangChain-Core | Modular RAG pipeline, easy to extend |
| **Memory** | ConversationBufferWindowMemory | Sliding window prevents token overflow |
| **UI** | Streamlit | Rapid prototyping, one-file deployment |
| **Config** | Pydantic-Settings | Type-safe environment variables |
| **Deployment** | Streamlit Community Cloud | Free hosting with GitHub integration |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- A [Groq](https://console.groq.com) account (free, no credit card required)

### 1. Clone the Repository
```
git clone https://github.com/DeepeshSherawat04/AI-Agent.git
cd AI-Agent/assignment-1

```

### 2. Set Up Virtual Environment
```
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies
pip install -r requirements.txt
Note: First run will download the embedding model (~90MB). This is automatic and one-time only.

### 4. Configure Environment Variables
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your free key at: https://console.groq.com/keys

Your .env should look like:

GROQ_API_KEY=gsk_your_actual_key_here
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  

### 5. Run the Application

```bash
python -m streamlit run app.py
The app will open at http://localhost:8501

This works on **Windows, macOS, and Linux** regardless of PATH issues.

---



🧪 Testing the Application
Test Credentials
No login required. The app is publicly accessible once deployed.
Test Scenarios : 

Scenario 1: Basic RAG Retrieval
You: Do you ship to India?
Agent: Yes, we ship to India. Shipping options include:

- **Standard delivery:** $12.99 (7‑10 business days) – free for orders over $150.
- **Express delivery:** $29.99 (3‑5 business days).

Let me know if you need more details!
Source: Shipping Policy v2.1, Section 3.1


Scenario 2: Conversational Memory (Follow-up)
You: Do you ship to India?
Agent: Yes, we ship to India...

You: How much does it cost to ship there?
Agent: - **Standard shipping to India:** Free on orders over $150.
- **Orders below $150:** The exact standard shipping fee isn’t listed in our knowledge base.
If you need the precise cost for smaller orders or details about express shipping, please reach out to support@gigacorp.com for assistance.
(Note: "there" correctly refers to India from previous turn)


Scenario 3: Source Citations
You: What is your return policy?
Agent: Our return policy:

- **30‑day hassle‑free returns** – you have 30 days from delivery to return an item.
- Items must be **unused**, in their **original packaging**, and include the **receipt or proof of purchase**.
- **Refunds** are processed within **5‑7 business days** after we receive the returned item.

If you need help initiating a return, you can do so through your GigaCorp account under “Order History” or contact our support team. International customers are responsible for return‑shipping costs.
Source: Return Policy v1.8, Section 2.0


Scenario 4: Out-of-Scope Query
You: Who is the CEO of GigaCorp?
Agent: I don't have information about that in our knowledge base. Please contact support@gigacorp.com for assistance.


Scenario 5: Greeting Handling
You: hi
Agent: Hello! How can I assist you today?
