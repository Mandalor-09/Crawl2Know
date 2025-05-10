# Basic Input ---------------------------------------------------------------------------------
from datetime import datetime
import logging, os, random, asyncio, hashlib
from datetime import datetime
from urllib.parse import urlparse, urlunparse
import re
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# ----------------------------------------------------------------------------------------------

# Crawler Input ---------------------------------------------------------------------------------
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from urllib.parse import urlparse
# ----------------------------------------------------------------------------------------------

# RAG Imports ----------------------------------------------------------------------------------
import os
import fitz  # PyMuPDF
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import numpy as np
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# ----------------------------------------------------------------------------------------------

# Telegram import ------------------------------------------------------------------------------
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
# ----------------------------------------------------------------------------------------------

# Logging Setup --------------------------------------------------------------------------------
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# ----------------------------------------------------------------------------------------------

# Helpers

def normalize_url(url: str) -> str:
    p = urlparse(url)
    scheme = p.scheme.lower()
    netloc = p.netloc.lower()
    path = p.path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, "", "", ""))

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_text_general(text):
    # Remove common UI/navigation words often found on websites
    text = re.sub(r"\b(Login|Register|Sign ?Up|Subscribe|Contact|Home|About|Next|Previous|Back|Logout)\b", "", text, flags=re.IGNORECASE)
    
    # Remove promotional content patterns
    text = re.sub(r"(promo code|buy now|purchase with|subscribe to|get access|securely with)", "", text, flags=re.IGNORECASE)
    
    # Remove copyright and repetitive footer lines
    text = re.sub(r"(©\s?\d{4}\s?.*|All rights reserved|Terms of Use|Privacy Policy)", "", text, flags=re.IGNORECASE)
    
    # Remove ASCII art, emojis, non-printable symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text

def smart_line_filter(line):
    if len(line.strip().split()) < 3:
        return False
    if line.strip().isupper():
        return False
    if re.match(r"^[\W\d\s]+$", line):  # symbols/numbers only
        return False
    if "http" in line:
        return False
    return True


def extract_text_from_files(file_paths):
    all_texts = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"⚠️ Skipping missing file: {path}")
            continue

        ext = os.path.splitext(path)[1].lower()

        if ext == ".txt" or ext == ".md":
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

        elif ext == ".pdf":
            content = ""
            with fitz.open(path) as doc:
                for page in doc:
                    content += page.get_text()

        elif ext == ".docx":
            content = ""
            doc = Document(path)
            for para in doc.paragraphs:
                content += para.text + "\n"

        else:
            print(f"⚠️ Unsupported file format: {path}")
            continue

        all_texts.append({
            "file_path": path,
            "text": content.strip()
        })

    return all_texts
# ----------------------------------------------------------------------------------------------

# Crawler Setup --------------------------------------------------------------------------------
# async def proxy_list():
#     with open('proxy.txt', "r") as proxy_file:
#         proxies = ['https://' + str(line.strip()) for line in proxy_file if line.strip()]
#         return random.choice(proxies)

async def craw_website(url):
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        viewport_width=1280,
        viewport_height=720,
        #proxy="http://proxy.example.com:8080",
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36",
        #text_mode=True,
        #light_mode=True,
    )
    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "ignore_images":True,
            "escape_html": False,
            "body_width": 80,
        }
    )
    crawler_config = CrawlerRunConfig(
        #wait_for="css:.main-content",
        word_count_threshold=0,
        only_text=True,
        remove_forms=True,
        check_robots_txt=True,
        scan_full_page=True,
        exclude_external_links=True,
        js_only=False,
        log_console=True,
        markdown_generator=md_generator,
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2, 
            include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=True,  # Enable streaming for arun_many()
        verbose=True,
    )
    os.makedirs('data',exist_ok=True)
    collected = []
    seen_normalized_urls = set() 
    seen_content_hashes = set()  

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async for res in await crawler.arun(url=url, config=crawler_config):
            logging.info(f"--- Processing: {res.url} ---") 
            if res.success:
                # 1. URL Deduplication
                normalized_url_for_dedup = normalize_url(res.url) # Your existing function
                logging.info(f"  Normalized URL for deduplication: {normalized_url_for_dedup}")
                if normalized_url_for_dedup in seen_normalized_urls:
                    logging.warning(f"  SKIP (URL): Normalized URL '{normalized_url_for_dedup}' (from raw: {res.url}) already seen.")
                    logging.info(f"--- Finished (skipped URL): {res.url} ---")
                    continue
                # Add to seen_urls AFTER content check, or here if you want to skip re-fetching known URLs even if content might differ (less common)
                # For now, let's add it after a successful content processing.

                # 2. Prepare Content
                current_page_raw_markdown = ""
                if res.markdown:
                    if hasattr(res.markdown, 'raw_markdown'):
                        current_page_raw_markdown = res.markdown.raw_markdown
                    else:
                        current_page_raw_markdown = str(res.markdown) # Fallback if not a MarkdownGenerationResult

                if not current_page_raw_markdown:
                    logging.warning(f"  No raw_markdown content for {res.url}. Skipping content hash and collection for this page.")
                    if res.html:
                         logging.warning(f"  HTML content (first 200 chars): {res.html[:200]}")
                    seen_normalized_urls.add(normalized_url_for_dedup) # Still mark URL as seen
                    logging.info(f"--- Finished (no content): {res.url} ---")
                    continue

                # 3. Content Hash Deduplication (using the content we intend to save)
                content_hash = hash_text(current_page_raw_markdown)
                logging.info(f"  Content Hash (from raw_markdown): {content_hash}")
                if content_hash in seen_content_hashes:
                    logging.warning(f"  SKIP (Content): Content hash '{content_hash}' for {res.url} already seen.")
                    # Also mark this URL as seen if we skipped due to content
                    seen_normalized_urls.add(normalized_url_for_dedup)
                    logging.info(f"--- Finished (skipped content): {res.url} ---")
                    continue

                # If we are here, both URL (normalized) and content hash are new.
                seen_normalized_urls.add(normalized_url_for_dedup)
                seen_content_hashes.add(content_hash)

                # Log details about the content being kept
                raw_md_length = len(current_page_raw_markdown)
                fit_md_length = 0
                if res.markdown and hasattr(res.markdown, 'fit_markdown'):
                     fit_md_text = getattr(res.markdown, "fit_markdown", "") # Get fit_markdown text
                     fit_md_length = len(fit_md_text if fit_md_text else "")

                logging.info(f"  KEEPING: {res.url}")
                logging.info(f"    Raw Markdown length: {raw_md_length}")
                logging.info(f"    Fit Markdown length (info only): {fit_md_length}")
                logging.info(f"    Final Text word count (from raw): {len(current_page_raw_markdown.split())}")

                cleaned = clean_text_general(current_page_raw_markdown)
                lines = cleaned.splitlines()
                cleaned = "\n".join(line for line in lines if smart_line_filter(line))
                collected.append((res.url, cleaned)) # Store the original URL and the raw markdown

            else:
                logging.error(f"  FAILED: {res.url}: {res.error_message}")
            logging.info(f"--- Finished processing: {res.url} ---")

    # Write combined output
    if collected:
        os.makedirs(f"data/{urlparse(url).netloc}",exist_ok=True)
        out_file = f"data/{urlparse(url).netloc}/{datetime.now():%S%M%H_%Y%m%d}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            for page_url, md_content in collected:
                f.write(f"## {page_url}\n\n{md_content}\n\n---\n\n")
        logging.info(f"Wrote {len(collected)} pages to {out_file}")
        return out_file
    else:
        logging.info("No content collected to write to file.")
        return None

# RAG ----------------------------------------------------------------------------------------
def chunk_documents(all_texts, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )

    all_chunks = []
    for file_doc in all_texts:
        text = file_doc["text"]
        source = file_doc["file_path"]

        chunks = splitter.create_documents([text], metadatas=[{"file_path": source}])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            all_chunks.append(chunk)

    return all_chunks  # Returns a list of langchain.Document objects

# STEP 1: Build BM25 Index
def build_bm25_index(docs: List[Document]):
    texts = [doc.page_content for doc in docs]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


# STEP 2: Build FAISS Index with bge-small
def build_faiss_index(docs: List[Document]):
    texts = [doc.page_content for doc in docs]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    return FAISS.from_documents(docs, embeddings)


# STEP 3: Hybrid Retrieval
def hybrid_retrieve(query: str, all_docs: List[Document], vectorizer, tfidf_matrix, faiss_index, top_k=5):
    sparse_hits = []
    if vectorizer and tfidf_matrix is not None and hasattr(tfidf_matrix, 'shape') and tfidf_matrix.shape[0] > 0: # Check if matrix is valid
        try:
            sparse_query_vec = vectorizer.transform([query])
            sparse_scores = (sparse_query_vec @ tfidf_matrix.T).toarray().flatten()
            # Ensure enough documents for argsort if less than top_k
            num_sparse_candidates = min(top_k, len(sparse_scores))
            sparse_indices = np.argsort(sparse_scores)[::-1][:num_sparse_candidates]
            sparse_hits = [all_docs[i] for i in sparse_indices if i < len(all_docs)]
        except Exception as e:
            logging.error(f"Error in BM25 retrieval: {e}")

    dense_hits = []
    if faiss_index:
        try:
            dense_hits = faiss_index.similarity_search(query, k=top_k)
        except Exception as e:
            logging.error(f"Error in FAISS retrieval: {e}")

    # Combine and deduplicate by page_content
    combined_docs_dict = {}
    # Prioritize dense hits if content is identical by adding them first
    for doc in dense_hits:
        if doc.page_content not in combined_docs_dict:
            combined_docs_dict[doc.page_content] = doc
    for doc in sparse_hits:
        if doc.page_content not in combined_docs_dict:
            combined_docs_dict[doc.page_content] = doc
    
    logging.info(f"Hybrid retrieval: BM25 found {len(sparse_hits)}, FAISS found {len(dense_hits)}. Combined unique: {len(combined_docs_dict)}")
    return list(combined_docs_dict.values()) # Reranker will take top N from this combined list

# STEP 4: Rerank with CrossEncoder
def rerank_docs(query: str, docs: List[Document], model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2", top_n=5):
    reranker = CrossEncoder(model_name)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_n]]


# STEP 5: Full Pipeline
def retrieve_top_docs(query: str, all_chunks: List[Document], top_k=5, rerank_top_n=3):
    # Build indexes
    vectorizer, tfidf_matrix = build_bm25_index(all_chunks)
    faiss_index = build_faiss_index(all_chunks)

    # Hybrid retrieval
    candidate_docs = hybrid_retrieve(query, all_chunks, vectorizer, tfidf_matrix, faiss_index, top_k)

    # Reranking
    final_docs = rerank_docs(query, candidate_docs, top_n=rerank_top_n)

    return final_docs

def generate_answer_from_docs(query: str, docs: List[Document]):
    context = "\n\n".join([f"{doc.page_content}" for doc in docs])

    prompt_template = PromptTemplate.from_template("""
You are a helpful assistant with access to context from documents.

Answer the following question based on the provided context.

If the answer is not explicitly available, respond with "I don't know."

---

Context:
{context}

---

Question: {question}

Answer:""")

    chain = LLMChain(
        llm=ChatGroq(
            temperature=0.3,
            model_name="llama3-70b-8192"  # or llama3-8b if you want a smaller one
        ),
        prompt=prompt_template
    )

    result = chain.run({
        "context": context,
        "question": query
    })

    return result

def format_history_for_llm(history: List[dict], max_turns=3) -> str:
    formatted_history = []
    # Take last N turns, ensure it alternates user/assistant if possible
    # For simplicity, just taking last N messages
    for turn in history[-max_turns*2:]: # Approx last max_turns of user+bot
        role = "User" if turn["role"] == "user" else "Bot"
        formatted_history.append(f"{role}: {turn['content']}")
    return "\n".join(formatted_history)

def augment_query_with_history(current_query: str, history: List[dict], llm_model_name: str = "llama3-8b-8192") -> str:
    if not history: # No history, return original query
        return current_query

    formatted_hist = format_history_for_llm(history)
    if not formatted_hist: # If history was empty or just one turn of user.
        return current_query

    template = """Given the following conversation history and the current user query, rewrite the user query to be a standalone question that can be understood without the history.
Focus on resolving pronouns (like "he", "she", "it", "his", "her", "its", "they", "their") and ambiguous references by incorporating specific entities or topics mentioned earlier in the conversation.
If the current query is already a standalone question and requires no modification, return it as is.

Conversation History:
{history}

Current User Query: {current_query}

Standalone Query:"""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=ChatGroq(temperature=0.0, model_name=llm_model_name), prompt=prompt) # Temp 0 for more deterministic rewrite

    try:
        response = chain.invoke({"history": formatted_hist, "current_query": current_query})
        standalone_query = response.get("text", current_query).strip()
        if not standalone_query: # LLM returned empty
            return current_query
        logging.info(f"Augmented query: Original='{current_query}', History='{formatted_hist}', Standalone='{standalone_query}'")
        return standalone_query
    except Exception as e:
        logging.error(f"Error in query augmentation: {e}")
        return current_query # Fallback to original query

def query_transform(original_query: str, transform_type: str = "step_back", llm_model_name: str = "llama3-8b-8192") -> str:
    if transform_type == "step_back":
        template = """Given the following user question, rephrase it as a more general, abstract "step-back" question that seeks to understand the broader context or principles.
Original Question: {original_question}
Step-back Question:"""
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=ChatGroq(temperature=0.2, model_name=llm_model_name), prompt=prompt)
        try:
            response = chain.invoke({"original_question": original_query})
            transformed = response.get("text", original_query).strip()
            logging.info(f"Step-back transformed query: '{original_query}' to '{transformed}'")
            if not transformed: # Ensure it doesn't return empty
                return original_query
            return transformed
        except Exception as e:
            logging.error(f"Error in step-back transform: {e}")
            return original_query
    return original_query

rag_components_cache = {}

async def get_rag_components_for_user(user_id: int, file_paths: List[str]):
    """
    Builds or retrieves RAG components from cache for a user.
    File paths are used as a key to represent the state of the documents.
    """
    # Create a stable cache key from the sorted list of file paths
    cache_key_tuple = tuple(sorted(file_paths))
    cache_key = f"{user_id}_{hash_text(str(cache_key_tuple))}"

    if cache_key in rag_components_cache:
        logging.info(f"RAG components for user {user_id} (key: {cache_key}) found in cache.")
        return rag_components_cache[cache_key]

    logging.info(f"Building RAG components for user {user_id} (key: {cache_key})...")
    texts = extract_text_from_files(file_paths)
    if not texts:
        logging.warning(f"No text extracted for user {user_id}. Cannot build RAG components.")
        return None

    all_chunks = chunk_documents(texts)
    if not all_chunks:
        logging.warning(f"No chunks created for user {user_id}. Cannot build RAG components.")
        return None

    vectorizer, tfidf_matrix = build_bm25_index(all_chunks)
    faiss_index = build_faiss_index(all_chunks)

    if (vectorizer is None or tfidf_matrix is None) and faiss_index is None:
        logging.error(f"Failed to build both BM25 and FAISS indexes for user {user_id}.")
        return None

    components = {
        "all_chunks": all_chunks,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "faiss_index": faiss_index,
    }
    rag_components_cache[cache_key] = components
    return components

def full_rag_chat(query: str, all_chunks: List[Document]):
    # Query transformation
    transformed_query = query_transform(query)

    # Hybrid retrieval + reranking
    top_docs = retrieve_top_docs(query, all_chunks, top_k=10, rerank_top_n=5)
    
    # Generate answer using Groq + Llama3
    answer = generate_answer_from_docs(query, top_docs)

    return answer
# ------------------------------------------------------------------------------------------

# Telegram Bot -----------------------------------------------------------------------------
# Global user session store
user_files_store = {} # Stores list of file paths for each user_id
user_conversation_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_files_store[user_id] = [] # Initialize or clear file list for the user
    user_conversation_history[user_id] = []

    # Clear RAG cache for this user if they restart
    keys_to_delete = [key for key in rag_components_cache if key.startswith(f"{user_id}_")]
    for key in keys_to_delete:
        del rag_components_cache[key]
    logging.info(f"User {user_id} started. Cleared their files and RAG cache.")
    await update.message.reply_text(
        "👋 Welcome! Send documents (PDF, DOCX, TXT, MD) or a website URL to crawl.\n"
        "Once sources are added, send your question.\n"
        "Use /reset to clear all your uploaded data."
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_files_store[user_id] = []
    user_conversation_history[user_id] = []

    # Clear RAG cache for this user
    keys_to_delete = [key for key in rag_components_cache if key.startswith(f"{user_id}_")]
    for key in keys_to_delete:
        del rag_components_cache[key]
    logging.info(f"User {user_id} reset. Cleared their files and RAG cache.")
    await update.message.reply_text("🧹 All your uploaded documents and website data have been cleared.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not update.message.document:
        return
    try:
        doc_file = await update.message.document.get_file()
        # Ensure unique filenames to prevent overwrites if multiple users upload 'file.pdf'
        unique_filename_base = f"{doc_file.file_unique_id}_{update.message.document.file_name}"
        # Sanitize filename to be safe for file systems
        safe_filename = re.sub(r'[^\w\.\-]', '_', unique_filename_base)
        
        uploads_dir = os.path.join(os.getcwd(), "user_uploads", str(user_id))
        os.makedirs(uploads_dir, exist_ok=True)
        full_file_path = os.path.join(uploads_dir, safe_filename)

        await doc_file.download_to_drive(full_file_path)

        user_files_store.setdefault(user_id, []).append(full_file_path)
        # Invalidate cache for this user as their document set has changed
        keys_to_delete = [key for key in rag_components_cache if key.startswith(f"{user_id}_")]
        for key in keys_to_delete:
            del rag_components_cache[key]

        await update.message.reply_text(f"📄 Saved: {update.message.document.file_name}. You can upload more, provide a URL, or send your query.")
        logging.info(f"User {user_id} uploaded {full_file_path}")
    except Exception as e:
        logging.error(f"Error handling document for user {user_id}: {e}")
        await update.message.reply_text("⚠️ Sorry, there was an error processing your document.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text_message = update.message.text.strip() # This is the user's current raw query

    # Add current user message to history
    user_conversation_history.setdefault(user_id, []).append({"role": "user", "content": text_message})

    # --- URL Handling Block (ensure bot responses here are also added to history) ---
    if text_message.lower().startswith("http://") or text_message.lower().startswith("https://"):
        await update.message.reply_text(f"🌐 Attempting to crawl: {text_message}\nThis might take a moment...")
        bot_intermediate_reply = f"🌐 Attempting to crawl: {text_message}\nThis might take a moment..."
        # It's good practice to add bot's thinking messages to history if they are significant for context
        # user_conversation_history[user_id].append({"role": "assistant", "content": bot_intermediate_reply})

        try:
            crawled_file_path = await craw_website(text_message)
            if crawled_file_path and os.path.exists(crawled_file_path):
                user_files_store.setdefault(user_id, []).append(crawled_file_path)
                keys_to_delete = [key for key in rag_components_cache if key.startswith(f"{user_id}_")]
                for key in keys_to_delete:
                    del rag_components_cache[key]
                
                bot_success_reply = f"✅ Website content saved from {urlparse(text_message).netloc}. You can now ask questions about it or add more sources."
                user_conversation_history[user_id].append({"role": "assistant", "content": bot_success_reply}) # Store this reply
                await update.message.reply_text(bot_success_reply)
                logging.info(f"User {user_id} crawled and added {crawled_file_path}")
            else:
                bot_fail_reply = "⚠️ Could not extract significant content from the website, or an error occurred."
                user_conversation_history[user_id].append({"role": "assistant", "content": bot_fail_reply}) # Store this reply
                await update.message.reply_text(bot_fail_reply)
        except Exception as e:
            logging.error(f"Error crawling website {text_message} for user {user_id}: {e}")
            bot_error_reply = "⚠️ Sorry, there was an error crawling that website."
            user_conversation_history[user_id].append({"role": "assistant", "content": bot_error_reply}) # Store this reply
            await update.message.reply_text(bot_error_reply)
        return # End processing if it was a URL

    # --- Question Handling Block ---
    if user_id not in user_files_store or not user_files_store[user_id]:
        bot_reply_text = "❗Please upload a document or provide a URL first before asking a question."
        user_conversation_history[user_id].append({"role": "assistant", "content": bot_reply_text})
        await update.message.reply_text(bot_reply_text)
        return

    # Bot is about to think, add this to history if you want the augmentation LLM to know the bot acknowledged the query
    # thinking_message = "🤔 Thinking..."
    # user_conversation_history[user_id].append({"role": "assistant", "content": thinking_message})
    await update.message.reply_text("🤔 Thinking...")

    original_query_for_this_turn = text_message # The raw query user typed in this turn
    logging.info(f"User {user_id} asked original query: {original_query_for_this_turn}")

    # 1. Augment the original query with conversation history
    current_history = user_conversation_history.get(user_id, [])
    # History for augmentation should be up to, but not including, the user's current message.
    # Since we added the current user message at the very start of handle_text,
    # we take history[:-1]
    history_for_augmentation = current_history[:-1] if len(current_history) > 1 else []

    # standalone_query IS DEFINED HERE:
    standalone_query = augment_query_with_history(original_query_for_this_turn, history_for_augmentation)
    logging.info(f"User {user_id} augmented (standalone) query for RAG: {standalone_query}")

    try:
        rag_sys_components = await get_rag_components_for_user(user_id, user_files_store[user_id])
        if not rag_sys_components:
            bot_error_reply = "⚠️ Could not prepare your documents for search. Please try re-uploading or using /reset."
            user_conversation_history[user_id].append({"role": "assistant", "content": bot_error_reply})
            await update.message.reply_text(bot_error_reply)
            return

        # 2. Optional: Further transform the standalone_query (e.g., step-back)
        #    For now, we'll use standalone_query directly for retrieval to test its effectiveness.
        #    If you want to apply step-back to the *already augmented* query:
        #    retrieval_query = query_transform(standalone_query, transform_type="step_back")
        #    Otherwise, if standalone_query is good enough:
        retrieval_query = standalone_query
        logging.info(f"User {user_id} final retrieval query: {retrieval_query}")


        candidate_docs = hybrid_retrieve(
            retrieval_query, # Use the query that has conversational context
            rag_sys_components["all_chunks"],
            rag_sys_components["vectorizer"],
            rag_sys_components["tfidf_matrix"],
            rag_sys_components["faiss_index"],
            top_k=10
        )

        if not candidate_docs:
            bot_info_reply = "ℹ️ I couldn't find any specific documents related to your query from the provided sources."
            user_conversation_history[user_id].append({"role": "assistant", "content": bot_info_reply})
            await update.message.reply_text(bot_info_reply)
            return

        # Rerank using the standalone_query as it's more complete
        final_top_docs = rerank_docs(standalone_query, candidate_docs, top_n=3)

        if not final_top_docs:
            bot_info_reply = "ℹ️ After refining, I couldn't pinpoint specific documents for your query from the provided sources."
            user_conversation_history[user_id].append({"role": "assistant", "content": bot_info_reply})
            await update.message.reply_text(bot_info_reply)
            return

        # Generate the answer using the original query the user typed for this specific turn,
        # but with the context found using the standalone (context-aware) query.
        answer = generate_answer_from_docs(original_query_for_this_turn, final_top_docs)
        
        user_conversation_history[user_id].append({"role": "assistant", "content": answer if answer else "No answer generated."})
        await update.message.reply_text(answer)

    except Exception as e:
        logging.error(f"Error processing query for user {user_id}: {original_query_for_this_turn}. Error: {e}", exc_info=True)
        bot_error_reply = "⚠️ An unexpected error occurred while processing your request. Please try again."
        user_conversation_history[user_id].append({"role": "assistant", "content": bot_error_reply})
        await update.message.reply_text(bot_error_reply)

# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 Telegram bot is running...")
    app.run_polling()


# if __name__ == "__main__":
#     output_path = asyncio.run(craw_website("https://quotes.toscrape.com/"))
#     print(f"\nSaved to: {output_path}" if output_path else "No data was saved.")

#     if output_path:
#         # Step 1: Read crawled content
#         all_texts = extract_text_from_files([output_path])

#         # Step 2: Chunk it
#         all_chunks = chunk_documents(all_texts)

#         # Step 3: Ask a sample question
#         query = "Where was Theodor Seuss Geisel was born?And what is the birth date"
#         response = full_rag_chat(query, all_chunks)
        
#         print("\n🤖 Answer:", response)


#if __name__ == "__main__":
    #output_path=asyncio.run(craw_website("https://quotes.toscrape.com/"))
    #output_path=asyncio.run(craw_website("https://www.scrapethissite.com/"))
    #output_path=asyncio.run(craw_website("https://www.mandalor09.site/"))
    #print(f"\nSaved to: {output_path}" if output_path else "No data was saved.")

# ----------------------------------------------------------------------------------------------