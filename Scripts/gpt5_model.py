from pathlib import Path


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent.parent
else:
    # If inside a notebook
    ROOT = Path.cwd().resolve()

import os, pickle, numpy as np
from tqdm.auto import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time 

import faiss
faiss.omp_set_num_threads(1)  

print("Imports OK (FAISS, NumPy, PyPDF2)")


# === OpenAI RAG Config (OpenAI-only) ===
import os
import warnings
warnings.filterwarnings('ignore')

# SDK
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(ROOT/'.env')

# ---- API key / client ----
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in your .env file and rerun cell")

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in your .env file and rerun cell")

openai_client = OpenAI()
client = OpenAI()

# ---- Config ----
class Config:
    # Paths (adjust as needed)
    PDF_FOLDER = ROOT / 'medicinali'
    CACHE_DIR  = ROOT / '.cache'  # cache hidden
    # Models
    GENERATION_MODEL = 'gpt-5.1'
    EMBEDDING_MODEL  = 'text-embedding-3-small'  

    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Retrieval
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.15

    # Batching
    BATCH_SIZE = 64
    VERBOSE = True

    # Cache artifact filenames
    EMBEDDINGS_PATH = 'embeddings.npy'
    INDEX_PATH      = 'faiss_index.idx'
    METADATA_PATH   = 'metadata.pkl'
    CHUNKS_PATH     = 'chunks.pkl'  

config = Config()

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


# ## Cell 4: Import Libraries

# In[4]:


import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss  # provided by faiss-cpu


print("Libraries imported (OpenAI, FAISS, PyPDF2, text splitters)")


# In[5]:


import os
import pickle
import numpy as np
import faiss

CACHE_DIR = config.CACHE_DIR

emb_path  = CACHE_DIR / "embeddings.npy"
idx_path  = CACHE_DIR / "faiss_index.idx"
meta_path = CACHE_DIR / "metadata.pkl"

if not (emb_path.exists() and idx_path.exists() and meta_path.exists()):
    raise FileNotFoundError(
        f"Cache files not found in {CACHE_DIR}.\n"
        "Run 'python3 -m Scripts.embedding' in terminal"
    )

# embeddings are optional at runtime, but nice to have
embeddings = np.load(emb_path)
index = faiss.read_index(str(idx_path))
with open(meta_path, "rb") as f:
    chunks = pickle.load(f)

print("Cache loaded")
print("  embeddings:", embeddings.shape)
print("  index ntotal:", index.ntotal)
print("  chunks:", len(chunks))
print("  example chunk keys:", chunks[0].keys())


# ## Cell 10: Retrieval function

# In[6]:


def embed_query(query: str) -> np.ndarray:
    """Return a normalized embedding vector (1, D) for the query."""
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(v)
    return v


def retrieve_relevant_chunks(
    query: str,
    top_k: int = None,
    threshold: float = None,
    verbose: bool = True,
):
    if top_k is None:
        top_k = config.TOP_K
    if threshold is None:
        threshold = config.SIMILARITY_THRESHOLD

    q_vec = embed_query(query)
    distances, indices = index.search(q_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx == -1:
            continue
        if threshold is not None and score < threshold:
            continue

        meta = chunks[idx]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "text": meta["text"],
                "document": meta["document"],
                "chunk_id": meta["chunk_id"],
            }
        )

    if verbose:
        print(f"\nRetrieved {len(results)} chunks:")
        for r in results:
            print(f"- [{r['document']} - chunk {r['chunk_id']}] score={r['score']:.3f}")

    return results


# In[7]:


def retrieve_relevant_chunks(
    query: str,
    top_k: int = None,
    threshold: float = None,  # unused, kept for compatibility
    verbose: bool = True,
):
    """
    1) FAISS retrieval with document diversity.
    2) If the query explicitly mentions brand names (Moment, Tachipirina, etc.),
       force in at least one chunk that contains each brand in its text.
    """
    if top_k is None:
        top_k = config.TOP_K

    # ---- 1) FAISS + diversity ----
    initial_k = max(top_k * 4, 20)
    q_vec = embed_query(query)  # (1, D), normalized
    distances, indices = index.search(q_vec, initial_k)

    max_per_doc = 3
    grouped = {}  # doc -> list[(score, idx)]

    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = chunks[idx]
        doc = meta["document"]
        if doc not in grouped:
            grouped[doc] = []
        if len(grouped[doc]) < max_per_doc:
            grouped[doc].append((float(score), idx))

    flat = []
    for doc, items in grouped.items():
        for score, idx in items:
            flat.append((score, idx))

    flat.sort(key=lambda x: x[0], reverse=True)

    results = []
    scores = []
    for rank, (score, idx) in enumerate(flat[:top_k], start=1):
        meta = chunks[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "text": meta["text"],
                "document": meta["document"],
                "chunk_id": meta["chunk_id"],
            }
        )
        scores.append(float(score))

    # ---- 2) Brand-name safety net ----
    q_lower = query.lower()

    # crude brand detection: capitalized tokens in original query
    brand_candidates = set()
    for token in query.replace("?", " ").replace(",", " ").split():
        cleaned = token.strip("?.!,").lower()
        if cleaned and len(cleaned) > 3:  # ignore "e", "tra", "le", etc.
            brand_candidates.add(cleaned)


    # e.g. "Differenze tra Moment e Tachipirina"
    #  -> {"moment", "tachipirina"}

    for brand in brand_candidates:
        # already have this brand in retrieved text?
        if any(brand in r["text"].lower() for r in results):
            continue

        candidates = []
        for idx, meta in enumerate(chunks):
            if brand in meta["text"].lower():
                # approximate score via dot product with query embedding
                s = float(embeddings[idx] @ q_vec[0])
                candidates.append((s, idx))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)

        # take the best candidate not already present
        for s, idx in candidates[:3]:
            meta = chunks[idx]
            if any(
                meta["document"] == r["document"]
                and meta["chunk_id"] == r["chunk_id"]
                for r in results
            ):
                continue

            results.append(
                {
                    "rank": None,
                    "score": float(s),
                    "text": meta["text"],
                    "document": meta["document"],
                    "chunk_id": meta["chunk_id"],
                }
            )
            scores.append(float(s))
            break  # one per brand is enough

    # ---- 3) Final sort + re-rank ----
    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:top_k]
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    scores = [r["score"] for r in results]

    if verbose:
        print(f"\nRetrieved {len(results)} chunks (TOP_K={top_k}, initial_k={initial_k}):")
        for r in results:
            print(f"- [{r['document']} - chunk {r['chunk_id']}] score={r['score']:.3f}")

    return results, scores


# ## Cell 12: Response system (Extractive QA)

# In[8]:


def format_history(history, max_turns: int = 5) -> str:
    if not history:
        return ""

    history = history[-max_turns:]

    lines = []
    for user_msg, bot_msg in history:
        lines.append(f"Utente: {user_msg}")
        lines.append(f"Assistente: {bot_msg}")
    return "\n".join(lines)


def answer_question(
    query: str,
    top_k: int = None,
    verbose: bool = True,
    chat_history=None
) -> dict:
    if top_k is None:
        top_k = config.TOP_K

    # --- Retrieval ---
    retrieved, scores_list = retrieve_relevant_chunks(
        query,
        top_k=top_k,
        verbose=verbose
    )

    # Decide if context is actually meaningful
    best_score = max(scores_list) if scores_list else 0.0
    MIN_BEST_SCORE = 0.03  # tune if needed

    if not retrieved or best_score < MIN_BEST_SCORE:
        return {
            "query": query,
            "answer": (
                "Non ho trovato contesto sufficientemente rilevante nei documenti "
                "per rispondere con sicurezza."
            ),
            "sources": [],
            "confidence": float(best_score),
        }

    # --- Build context block from retrieved chunks ---
    context_blocks = []
    for r in retrieved:
        header = f"[{r['document']} - chunk {r['chunk_id']}]"
        context_blocks.append(f"{header}\n{r['text']}")
    context = "\n\n".join(context_blocks)

    # --- Session memory ---
    history_text = format_history(chat_history) if chat_history else ""
    if history_text:
        history_section = f"Storia della conversazione (ultimi turni):\n{history_text}\n\n"
    else:
        history_section = ""

    system_prompt = (
        "Sei un assistente che risponde a domande sui farmaci da banco italiani "
        "(OTC) usando solo le informazioni fornite nel contesto. "
        "Se non trovi la risposta nel contesto, dichiara esplicitamente che non puoi rispondere."
    )

    user_content = (
        f"{history_section}"
        f"Domanda attuale dell'utente: {query}\n\n"
        f"Contesto estratto dai documenti:\n{context}\n\n"
        "Rispondi in italiano, in modo conciso e preciso. Se la domanda non può essere "
        "risolta usando solo il contesto fornito, dillo esplicitamente."
    )

    # --- GPT-4o generation ---
    resp = openai_client.chat.completions.create(
        model=config.GENERATION_MODEL,  # e.g. "gpt-4o-mini" or "gpt-4o"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )

    answer = resp.choices[0].message.content

    # --- Confidence based on similarity scores ---
    if scores_list:
        avg_score = float(np.mean(scores_list))
    else:
        avg_score = float(np.mean([r.get("score", 0.0) for r in retrieved]))

    sources = list({r.get("document", "Sconosciuto") for r in retrieved})

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "confidence": avg_score,
        "retrieved_chunks": retrieved,
    }


# ## Cell 14: Interactive Chat 
# 
# **Use:**
# - Ask questions in natural language
# - Type 'exit' or 'quit' to exit 
# - Type 'stats' to view system stats 

# In[9]:


def interactive_chat():
    """Interactive RAG Chat using OpenAI"""
    print('\n' + '='*60)
    print('CHAT INTERATTIVA RAG (OpenAI)')
    print('='*60)
    print('\nComandi:')
    print('  - exit / quit : esci dalla chat')
    print('  - stats       : mostra statistiche del sistema')
    print('='*60 + '\n')

    query_count = 0

    while True:
        try:
            user_input = input('Tu: ').strip()
            if not user_input:
                continue

            # Exit command
            if user_input.lower() in ['exit', 'quit']:
                print('\nArrivederci!')
                break

            # Stats command
            elif user_input.lower() == 'stats':
                print(f'\nSystem stats:')
                print(f'  - Queries made: {query_count}')
                print(f'  - Total chunks: {len(chunks):,}')
                print(f'  - Documents: {len(set(c.get("document","?") for c in chunks))}')
                print(f'  - Index size: {index.ntotal:,} vectors')
                continue

            # Normal question
            query_count += 1
            print("\nRunning...\n")

            result = answer_question(user_input, verbose=False)

            print('Assistente:')
            print('-'*60)
            print(result['answer'])
            print('-'*60)

            if result['sources']:
                src = result['sources'][0]
                print(f"Fonte principale: {src['document']} (similarità: {src['score']:.0%})")

            print()

        except KeyboardInterrupt:
            print('\n\nUscita manuale. A presto!')
            break
        except Exception as e:
            print(f'\nErrore: {e}\n')

print('Chat pronta!')
print('\nPer avviare la chat, esegui: interactive_chat()')

def answer_with_rag(question: str) -> dict:
    """
    Used for quantitative_evaluation.

    Single-shot RAG call:
    - retrieve chunks with FAISS
    - build context
    - generate answer with OpenAI
    - return answer + basic metrics (timings, tokens, similarity scores)
    """
    t0 = time.perf_counter()

    # --- Retrieval timing ---
    t_retrieval_start = time.perf_counter()
    retrieved, scores_list = retrieve_relevant_chunks(
        question,
        top_k=config.TOP_K,
        verbose=False
    )
    t_retrieval = time.perf_counter() - t_retrieval_start

    MIN_BEST_SCORE = 0.05
    best_score = max(scores_list) if scores_list else 0.0

    # If retrieval is too weak, bail out early
    if (not retrieved) or (best_score < MIN_BEST_SCORE):
        t_total = time.perf_counter() - t0
        return {
            "query": question,
            "answer": (
                "Non ho trovato contesto sufficientemente rilevante nei documenti "
                "per rispondere con sicurezza."
            ),
            "sources": [],
            "confidence": float(best_score),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "t_retrieval": t_retrieval,
            "t_generation": 0.0,
            "t_total": t_total,
        }

    # --- Build context (same style as answer_question) ---
    context_blocks = []
    for r in retrieved:
        header = f"[{r['document']} - chunk {r['chunk_id']}]"
        context_blocks.append(f"{header}\n{r['text']}")
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "Sei un assistente che risponde a domande sui farmaci da banco italiani "
        "(OTC) usando solo le informazioni fornite nel contesto. "
        "Se non trovi la risposta nel contesto, dichiara esplicitamente che non puoi rispondere."
    )

    user_content = (
        f"Domanda attuale dell'utente: {question}\n\n"
        f"Contesto estratto dai documenti:\n{context}\n\n"
        "Rispondi in italiano, in modo conciso e preciso. Se la domanda non può essere "
        "risolta usando solo il contesto fornito, dillo esplicitamente."
    )

    # --- Generation timing ---
    t_gen_start = time.perf_counter()
    resp = openai_client.chat.completions.create(
        model=config.GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    t_generation = time.perf_counter() - t_gen_start
    t_total = time.perf_counter() - t0

    answer = resp.choices[0].message.content

    # --- Token usage (if available) ---
    usage = getattr(resp, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (
            prompt_tokens + completion_tokens
        )
    else:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

    # --- Confidence metric based on similarity scores ---
    if scores_list:
        avg_score = float(np.mean(scores_list))
    else:
        avg_score = 0.0

    # Unique list of source documents
    sources = list({r.get("document", "Sconosciuto") for r in retrieved})

    return {
        "query": question,
        "answer": answer,
        "sources": sources,
        "confidence": avg_score,   # you could also store best_score instead
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "t_retrieval": t_retrieval,
        "t_generation": t_generation,
        "t_total": t_total,
    }