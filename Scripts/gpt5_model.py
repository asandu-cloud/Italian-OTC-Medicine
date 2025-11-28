from pathlib import Path

# Robust project root: works both as notebook-converted .py and imported module
if "__file__" in globals():
    # Running as a .py file (e.g. Notebooks/gpt4o_model.py)
    ROOT = Path(__file__).resolve().parent.parent  
else:
    # Running from inside the Notebooks/ folder in Jupyter
    ROOT = Path.cwd().resolve().parent             

load_dotenv(ROOT / ".env")


import os, pickle, numpy as np
from tqdm.auto import tqdm
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

load_dotenv(ROOT/'.env')

# ---- API key / client ----
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY non trovato nell'ambiente. Imposta la variabile e riestheseegui la cella.")

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
    SIMILARITY_THRESHOLD = 0.3

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


import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss  # provided by faiss-cpu


print("Libraries imported (OpenAI, FAISS, PyPDF2, text splitters)")


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

def retrieve_relevant_chunks(query, top_k=None, threshold=None, verbose=True):
    """
    Retrieves relevant chunks using OpenAI embeddings + FAISS.
    Includes safety guards for FAISS crashes on Apple Silicon.
    """
    if top_k is None:
        top_k = config.TOP_K
    if threshold is None:
        threshold = config.SIMILARITY_THRESHOLD

    # Ensure FAISS index and chunks exist
    if 'index' not in globals() or getattr(index, "ntotal", 0) == 0:
        raise RuntimeError("FAISS index not loaded. Run embedding generation or ensure_faiss_ready().")
    if 'chunks' not in globals() or not chunks:
        raise RuntimeError("Chunks not loaded. Run the PDF→chunk cell.")

    # Get embedding from OpenAI
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=query)
    qe = np.array(resp.data[0].embedding, dtype=np.float32)
    qe = np.expand_dims(qe, axis=0)
    faiss.normalize_L2(qe)

    # Double-check dimensionality
    if qe.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dimension mismatch: query={qe.shape[1]}, index={index.d}. "
            "Rebuild the FAISS index with the same embedding model."
        )

    # Create a *copy* of the array to avoid FAISS segfaults on MPS
    qe = np.ascontiguousarray(qe)

    # Try search safely
    try:
        distances, indices = index.search(qe, top_k)
    except Exception as e:
        raise RuntimeError(f"FAISS search failed: {e}")

    results, scores = [], []
    for idx, score in zip(indices[0], distances[0]):
        if score >= threshold:
            results.append(chunks[idx])
            scores.append(float(score))

    if verbose:
        print(f"Found {len(results)}/{top_k} relevant chunks (≥ {threshold})")

    return results, scores

print("Stable retrieval function loaded for OpenAI + FAISS on CPU")


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

    retrieved = retrieve_relevant_chunks(query, top_k=top_k, verbose=verbose)

    # --- NEW: normalize retrieved items to dicts ---
    normalized = []
    for item in retrieved:
        # Case 1: already a dict -> keep as is
        if isinstance(item, dict):
            normalized.append(item)
            continue

        # Case 2: list/tuple like [text, meta_dict, score] or similar
        if isinstance(item, (list, tuple)):
            meta = None
            text = None
            score = None

            for elem in item:
                if isinstance(elem, dict) and "document" in elem:
                    meta = elem
                elif isinstance(elem, str):
                    text = elem
                elif isinstance(elem, (int, float)):
                    score = float(elem)

            if meta is not None:
                normalized.append({
                    "document": meta.get("document", ""),
                    "chunk_id": meta.get("chunk_id", -1),
                    "text": text if text is not None else meta.get("text", ""),
                    "score": score if score is not None else float(meta.get("score", 0.0)),
                })
            continue

        # Anything else we just ignore
        # (shouldn't happen in your pipeline)
        continue

    retrieved = normalized
    # --- END NEW BLOCK ---

    if not retrieved:
        return {
            "query": query,
            "answer": "Non ho trovato contesto rilevante nei documenti.",
            "sources": [],
            "confidence": 0.0,
        }

    # Build context string for the LLM
    context_blocks = []
    for r in retrieved:
        header = f"[{r['document']} - chunk {r['chunk_id']}]"
        context_blocks.append(f"{header}\n{r['text']}")
    context = "\n\n".join(context_blocks)

    # === SESSION MEMORY ===
    history_text = format_history(chat_history) if chat_history else ""

    if history_text:
        history_section = f"Storia della conversazione (ultimi turni):\n{history_text}\n\n"
    else:
        history_section = ""


    # --- rest of your existing code stays the same ---
    system_prompt = (
        "Sei un assistente che risponde a domande su farmaci da banco italiani "
        "(OTC) usando solo le informazioni fornite nel contesto. "
        "Se non trovi la risposta nel contesto, dichiara esplicitamente che non puoi rispondere."
    )

    messages = [
    {
        "role": "system",
        "content": [
            {"type": "input_text", "text": system_prompt},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    f"{history_section}"
                    f"Domanda attuale: {query}\n\n"
                    "Contesto (estratti dai documenti):\n"
                    f"{context}\n\n"
                    "Rispondi in italiano, in modo conciso e preciso. "
                    "Se necessario, cita i nomi dei foglietti illustrativi."
                    "Se la domanda si basa su informazioni non presenti nel contesto, dillo esplicitamente."
                ),
            }
        ],
    },
]



    resp = client.responses.create(
        model=config.GENERATION_MODEL,
        input=messages,
        max_output_tokens=600,
        temperature=0.2,
    )
    answer = resp.output[0].content[0].text if hasattr(resp, "output") else resp.choices[0].message.content

    scores = [r["score"] for r in retrieved]
    avg_score = float(np.mean(scores))

    sources = list({r["document"] for r in retrieved})

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "confidence": avg_score,
        "retrieved_chunks": retrieved,
    }

def answer_with_rag(question: str) -> dict:
    """
    Wrapper used by quantitative_evaluation.py.

    - Uses the same retrieval stack as `answer_question`
    - GPT5.1 for generation
    - Returns timing + token-usage metrics in a format compatible with the evaluator.
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

    if not retrieved:
        t_total = time.perf_counter() - t0
        return {
            "query": question,
            "answer": "Non ho trovato contesto rilevante nei documenti.",
            "sources": [],
            "confidence": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "t_retrieval": t_retrieval,
            "t_generation": 0.0,
            "t_total": t_total,
        }

    # --- Build context exactly like in answer_question() ---
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

    full_prompt = (
        f"Sistema:\n{system_prompt}\n\n"
        f"Domanda attuale dell'utente: {question}\n\n"
        f"Contesto estratto dai documenti:\n{context}\n\n"
        "Rispondi in italiano, in modo conciso e preciso. Se la domanda non può essere "
        "risolta usando solo il contesto fornito, dillo esplicitamente."
    )

    # --- Generation timing ---
    t_gen_start = time.perf_counter()
    gemini_response = gemini_client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=full_prompt
    )
    t_generation = time.perf_counter() - t_gen_start
    t_total = time.perf_counter() - t0

    answer = gemini_response.text

    # --- Token usage (if provided by the Gemini client) ---
    usage = getattr(gemini_response, "usage_metadata", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (
            prompt_tokens + completion_tokens
        )
    else:
        # Fallback when usage metadata is missing
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

    # --- Confidence based on similarity scores ---
    if scores_list:
        avg_score = float(np.mean(scores_list))
    else:
        avg_score = 0.0

    sources = list({r.get("document", "Sconosciuto") for r in retrieved})

    return {
        "query": question,
        "answer": answer,
        "sources": sources,
        "confidence": avg_score,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "t_retrieval": t_retrieval,
        "t_generation": t_generation,
        "t_total": t_total,
    }
