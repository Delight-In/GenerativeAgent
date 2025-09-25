import os
import uuid
import re
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from PyPDF2 import PdfReader
import pdfplumber
import docx
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from llama_cpp import Llama

# --------- CONFIG ---------
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
CHROMA_DIR = "./chroma_db1"
COLLECTION_NAME = "local_doc_collection"
SIMILARITY_THRESHOLD = 0.52
CHUNK_SIZE = 800
TOP_K = 3
# --------------------------

# --------- Loaders ---------

def extract_text(file_path):
    ext = file_path.lower()
    if ext.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif ext.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif ext.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type")

def extract_text_from_pdf(file_path):
    text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
    except:
        reader = PdfReader(file_path)
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    text.append(t)
            except:
                continue
    return "\n".join(text)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks

# --------- Embedding ---------

def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embs / norms

# --------- Chroma Setup ---------

def get_chroma_collection(embedder):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        collection = client.create_collection(name=COLLECTION_NAME)
    return collection

def index_document(file_path, embedder, collection):
    raw_text = clean_text(extract_text(file_path))
    chunks = chunk_text(raw_text, max_chars=CHUNK_SIZE)
    embeddings = embed_texts(embedder, chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": os.path.basename(file_path), "chunk": i} for i in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)
    print(f"✅ Indexed {len(chunks)} chunks from {file_path}.")

# --------- Semantic Search ---------

def search_similar_chunks(query, embedder, collection, top_k=TOP_K):
    query_emb = embed_texts(embedder, [query])[0]
    results = collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k, include=["documents", "distances", "metadatas"])

    docs = results['documents'][0]
    distances = results['distances'][0]

    sims = 1 - np.array(distances)  # cosine similarity conversion from distance
    retrieved = [{"text": doc, "score": sim} for doc, sim in zip(docs, sims)]

    in_context = len(retrieved) > 0 and max(sims) >= SIMILARITY_THRESHOLD
    return in_context, retrieved, query_emb

# --------- Prompt Template ---------

def build_prompt(question: str, context_chunks: List[dict], in_context: bool) -> str:
    if not in_context:
        return f"""You are a helpful assistant. The user asked an out-of-context question. Just answer it generally.\n\nQuestion: {question}\nAnswer:"""

    context_text = "\n\n".join([f"{i+1}. {chunk['text']}" for i, chunk in enumerate(context_chunks)])
    return f"""You are a helpful assistant. Use ONLY the below CONTEXT to answer the question.
and frame answer which matched most according to question and greet politely and
also if question is related to topic then try to answer in short if doc is not containing details.

CONTEXT:
{context_text}

Question: {question}
Answer:"""

# --------- Llama Inference ---------

def load_llama_model():
    return Llama(model_path=MODEL_PATH, n_ctx=4096)

def run_llm(llm: Llama, prompt: str):
    output = llm(prompt=prompt, max_tokens=512, stop=["\n\n", "</s>"])
    return output["choices"][0]["text"].strip()
