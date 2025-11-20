from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os

# ---- Project root (works for Scripts/ and Notebooks/) ----
if "__file__" in globals():
    # running as Scripts/gpt5_model.py
    ROOT = Path(__file__).resolve().parents[1]   # .../RAG-Italian-OTC-Medicine
else:
    # running inside Notebooks/gpt5_model.ipynb
    ROOT = Path.cwd().parent                     # parent of Notebooks/ -> project root

class Config:
    # Paths
    PDF_FOLDER = ROOT / "medicinali"
    CACHE_DIR  = ROOT / ".cache"

    # Models
    GENERATION_MODEL = "gpt-5"
    EMBEDDING_MODEL  = "text-embedding-3-small"

    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Retrieval
    TOP_K = 10
    SIMILARITY_THRESHOLD = 0.30

    # Batching
    BATCH_SIZE = 64
    VERBOSE = True

    # Cache artifact filenames
    EMBEDDINGS_PATH = "embeddings.npy"
    INDEX_PATH      = "faiss_index.idx"
    METADATA_PATH   = "metadata.pkl"
    CHUNKS_PATH     = "chunks.pkl"

config = Config()

# ---- Env + client ----
load_dotenv(ROOT / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY non trovato nell'ambiente. Imposta la variabile e riesegui la cella.")

client = OpenAI()
print("client ready, ROOT =", ROOT)
print("PDF_FOLDER:", config.PDF_FOLDER)
print("CACHE_DIR:", config.CACHE_DIR)

import os, pickle, numpy as np
from tqdm.auto import tqdm
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
faiss.omp_set_num_threads(1)  # helps stability on macOS

print("Imports OK (FAISS, NumPy, PyPDF2)")

# ---- Ensure folders / show status ----
os.makedirs(config.CACHE_DIR, exist_ok=True)

if os.path.exists(config.PDF_FOLDER):
    pdf_count = sum(f.lower().endswith('.pdf') for f in os.listdir(config.PDF_FOLDER))
    print(f'Config caricata | PDF trovati: {pdf_count}')
    if pdf_count == 0:
        print('Nessun PDF trovato nella cartella.')
    print(f'Retrieval: TOP_K={config.TOP_K}, THRESHOLD={config.SIMILARITY_THRESHOLD}')
    print(f'Model → Gen: {config.GENERATION_MODEL} | Emb: {config.EMBEDDING_MODEL}')
    print(f'Cache dir: {config.CACHE_DIR}')
else:
    print(f'ERRORE: Cartella non trovata: {config.PDF_FOLDER}')
    print('Suggerimenti:')
    print('1) Verifica che la cartella esista')
    print('2) Controlla il percorso (maiuscole/minuscole contano)')

# Loading OpenAI chat

from openai import OpenAI
import os

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Key not in env file.")

client = OpenAI()
model_name = "gpt-5"  # 

print(f"OpenAI model ready: {model_name}")

def generate_with_openai(prompt: str, model: str = model_name, max_tokens: int = 512, temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ## CELL FOR CLEARING CACHE
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss  # provided by faiss-cpu

# OpenAI SDK + dotenv

from openai import OpenAI
client = OpenAI()

print("Libraries imported (OpenAI, FAISS, PyPDF2, text splitters)")


# ## Cell 5: Text cleaning functions

# In[8]:


def clean_text(text: str) -> str:
    """Cleans text by removing non-printable characters"""
    if not text or not isinstance(text, str):
        return ''
    cleaned = ''.join(c for c in text if c.isprintable() or c == '\n')
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

print('Text cleaning func loaded')


# ## 📄 Cella 6: Extraction PDF

# In[9]:


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract clean text from PDF"""
    try:
        reader = PdfReader(pdf_path)
        content = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text and len(text.strip()) > 20:
                    cleaned = clean_text(text)
                    if cleaned:
                        content.append(cleaned)
            except:
                continue
        return '\n'.join(content).strip()
    except Exception as e:
        print(f'PDF reading error: {e}')
        return ''

print('PDF extraction function loaded')


def answer_with_rag(question: str):
    """
    Return eval metrics
    """
    t0 = time()

    # --- retrieval ---
    t_retrieval_start = time()
    q_emb = embed_question(question)         # your existing function
    docs, scores = retrieve(q_emb)           # your existing function
    context = "\n\n".join(d.page_content for d in docs)
    t_retrieval = time() - t_retrieval_start

    # --- generation ---
    t_gen_start = time()
    resp = client.chat.completions.create(
        model="gpt-5",   
        messages=[
            {"role": "system", "content": "Sei un assistente per medicinali OTC italiani."},
            {"role": "user", "content": f"Domanda: {question}\n\nContesto:\n{context}"}
        ],
        temperature=0.1,
    )
    t_generation = time() - t_gen_start

    answer = resp.choices[0].message.content
    usage = resp.usage   # has prompt_tokens, completion_tokens, total_tokens

    t_total = time() - t0

    return {
        "answer": answer,
        "context": context,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "t_retrieval": t_retrieval,
        "t_generation": t_generation,
        "t_total": t_total,
    }