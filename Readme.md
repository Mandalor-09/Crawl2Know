# 🧠 AI Community Moderator Bot

An intelligent, AI-powered community moderation agent built as a Telegram bot that can answer user queries based on brand knowledge from websites and uploaded documents.

This project was created as part of the **AI Community Moderator MVP Challenge** to demonstrate a retrieval-augmented generation (RAG) system with hybrid indexing, reranking, and brand-aligned responses.

---

## 🚀 Features

### ✅ Knowledge Integration
- 🕸️ **Website Crawling:** Crawls brand websites and extracts structured content for knowledge base creation.
- 📄 **Document Processing:** Accepts and processes uploaded PDFs, DOCX, TXT, or Markdown files.
- 🧱 **Advanced RAG Pipeline:** Cleans, chunks, and indexes data using a BM25 (TF-IDF) + FAISS hybrid retrieval strategy.
- 🤖 **Contextual Answers:** Answers user questions based on the indexed knowledge, ensuring responses are grounded in provided data.

### ✅ Conversational Interface
- 💬 **Telegram Deployment:** Interacts with users in real-time via a Telegram bot.
- 🔁 **Conversational Memory:** Maintains conversation history for context-aware query augmentation, enabling follow-up questions.
- 🧠 **Query Understanding:** Implements *step-back prompting* to handle vague or abstract user queries more effectively.

### ✅ Brand Alignment
- 🎯 **Customizable Prompts:** Utilizes a custom prompt template to reflect the desired brand tone and voice.
- 📎 **Source-Grounded Responses:** Answers are generated using a context-aware LLM (Groq LLaMA3) strictly based on the provided information.
- 📝 **Trusted Information:** Relies on the indexed brand data as the single source of truth for responses.

---

## 🧱 Tech Stack

| Category          | Tool / Library                                      | Purpose                                           |
|-------------------|-----------------------------------------------------|---------------------------------------------------|
| **LLM Engine**    | Groq API (LLaMA3-70B & LLaMA3-8B) via LangChain     | Answer Generation, Query Transformation           |
| **Embeddings**    | `BAAI/bge-small-en` (via HuggingFace & LangChain)   | Semantic text representation for dense retrieval  |
| **Reranker**      | `cross-encoder/ms-marco-TinyBERT-L-2-v2` (Sentence Transformers) | Refines search results for relevance          |
| **Retrieval**     | Hybrid: BM25 (TF-IDF via Scikit-learn) + FAISS (via LangChain) | Efficient sparse and dense document retrieval   |
| **Doc Parsing**   | PyMuPDF (`fitz`), `python-docx`                     | Extracting text from PDF and DOCX files         |
| **Web Crawling**  | `crawl4ai`                                          | Deep website crawling, HTML to Markdown           |
| **Bot Framework** | `python-telegram-bot`                               | Telegram bot interaction and API handling       |
| **Orchestration** | LangChain                                           | Building and managing RAG chains & prompts      |
| **Environment**   | Python, `python-dotenv`                             | Core logic and environment variable management  |
| **Runtime**       | Single Python script (`main.py`)                    | All core logic consolidated                     |

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-community-moderator-bot.git
cd ai-community-moderator-bot
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
create a .env file
```bash
GROQ_API_KEY="your_groq_api_key_here"
TELEGRAM_TOKEN="your_telegram_bot_token_here"
```

### 5. Run the Bot
```bash
python main.py
```