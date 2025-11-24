from pathlib import Path

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent.parent
else:
    # If inside a notebook
    ROOT = Path.cwd().resolve()

import os, pickle, numpy as np
from tqdm.auto import tqdm
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
from google import genai

load_dotenv(ROOT/'.env')

# ---- API key / client ----
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in your .env file and rerun cell")

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in your .env file and rerun cell")

openai_client = OpenAI()
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---- Config ----
class Config:
    # Paths (adjust as needed)
    PDF_FOLDER = ROOT / "medicinali"
    CACHE_DIR  = ROOT / ".cache"  # hidden cache folder

    # Models
    EMBEDDING_MODEL  = "text-embedding-3-small"  
    GENERATION_MODEL = "gemini-3-pro-preview"      

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
    EMBEDDINGS_PATH = "embeddings.npy"
    INDEX_PATH      = "faiss_index.idx"
    METADATA_PATH   = "metadata.pkl"
    CHUNKS_PATH     = "chunks.pkl"


config = Config()

# ---- Ensure folders / show status ----
os.makedirs(config.CACHE_DIR, exist_ok=True)

if os.path.exists(config.PDF_FOLDER):
    pdf_count = sum(f.lower().endswith(".pdf") for f in os.listdir(config.PDF_FOLDER))
    print(f"Config loaded | PDFs found: {pdf_count}")
    if pdf_count == 0:
        print("Warning: no PDFs found in the folder.")
    print(f"Retrieval: TOP_K={config.TOP_K}, THRESHOLD={config.SIMILARITY_THRESHOLD}")
    print(f"Embedding model:  {config.EMBEDDING_MODEL}")
    print(f"Generation model: {config.GENERATION_MODEL} (Gemini 3)")
    print(f"Cache directory:  {config.CACHE_DIR}")
else:
    print(f"ERROR: Folder not found: {config.PDF_FOLDER}")
    print("Suggestions:")
    print("1) Check that the folder exists")
    print("2) Check the path and case-sensitivity")


# ## Cell 4: Import Libraries

# In[34]:


import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss  # provided by faiss-cpu


print("Libraries imported (OpenAI, FAISS, PyPDF2, text splitters)")


# In[35]:


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

# In[36]:


def embed_query(query: str) -> np.ndarray:
    """Return a normalized embedding vector (1, D) for the query."""
    resp = openai_client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
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


# In[37]:


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
    resp = openai_client.embeddings.create(model=config.EMBEDDING_MODEL, input=query)
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
    """
    """
    if not history:
        return ""

    # Keep only the last N turns
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
    """
    """
    if top_k is None:
        top_k = config.TOP_K

    retrieved, scores_list = retrieve_relevant_chunks(
        query,
        top_k=top_k,
        verbose=verbose
    )

    if not retrieved:
        return {
            "query": query,
            "answer": "Non ho trovato contesto rilevante nei documenti.",
            "sources": [],
            "confidence": 0.0,
        }

    # Build context block from retrieved chunks
    context_blocks = []
    for r in retrieved:
        header = f"[{r['document']} - chunk {r['chunk_id']}]"
        context_blocks.append(f"{header}\n{r['text']}")
    context = "\n\n".join(context_blocks)

    # Session memory
    history_text = format_history(chat_history) if chat_history else ""
    if history_text:
        history_section = f"Storia della conversazione (ultimi turni):\n{history_text}\n\n"
    else:
        history_section = ""

    # System prompt (in Italian, as requested)
    system_prompt = (
        "Sei un assistente che risponde a domande sui farmaci da banco italiani "
        "(OTC) usando solo le informazioni fornite nel contesto. "
        "Se non trovi la risposta nel contesto, dichiara esplicitamente che non puoi rispondere."
    )

    # Build full prompt for Gemini (single string, not OpenAI-style messages)
    full_prompt = (
        f"Sistema:\n{system_prompt}\n\n"
        f"{history_section}"
        f"Domanda attuale dell'utente: {query}\n\n"
        f"Contesto estratto dai documenti:\n{context}\n\n"
        "Rispondi in italiano, in modo conciso e preciso. Se la domanda non può essere "
        "risolta usando solo il contesto fornito, dillo esplicitamente."
    )

    # Call Gemini for generation
    gemini_response = gemini_client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=full_prompt
    )

    # Extract answer text from Gemini response
    answer = gemini_response.text

    # Confidence based on similarity scores
    if scores_list:
        avg_score = float(np.mean(scores_list))
    else:
        # Fallback if scores are not available
        avg_score = float(np.mean([r.get("score", 0.0) for r in retrieved]))

    # Unique source documents
    sources = list({r.get("document", "Sconosciuto") for r in retrieved})

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "confidence": avg_score,
        "retrieved_chunks": retrieved,
    }



def interactive_chat():
    """Interactive RAG chat using Gemini (generation) + OpenAI embeddings."""
    print('\n' + '='*60)
    print('CHAT INTERATTIVA RAG (Gemini 3)')
    print('='*60)
    print('\nComandi:')
    print('  - exit / quit : esci dalla chat')
    print('  - stats       : mostra statistiche del sistema')
    print('='*60 + '\n')

    query_count = 0
    history = []   # local session memory for CLI chat

    while True:
        try:
            user_input = input('Tu: ').strip()
            if not user_input:
                continue

            # Exit
            if user_input.lower() in ['exit', 'quit']:
                print('\nArrivederci!')
                break

            # Stats
            elif user_input.lower() == 'stats':
                print(f'\nStatistiche di sistema:')
                print(f'  - Query effettuate: {query_count}')
                print(f'  - Chunk totali: {len(chunks):,}')
                print(f'  - Documenti unici: {len(set(c.get("document","?") for c in chunks))}')
                print(f'  - Dimensione indice: {index.ntotal:,} vettori')
                continue

            # Normal question
            query_count += 1
            print("\nSto elaborando...\n")

            # Call RAG with session history
            result = answer_question(
                query=user_input,
                verbose=False,
                chat_history=history
            )

            # Retrieve answer
            answer = result.get("answer", "Errore: risposta non trovata.")

            # Print assistant answer
            print('Assistente:')
            print('-'*60)
            print(answer)
            print('-'*60)

            # Print sources (list of document names)
            sources = result.get("sources", [])
            if sources:
                print("Fonti consultate:")
                for src in sources:
                    print(f"- {src}")

            print()

            # Update session memory
            history.append([user_input, answer])

        except KeyboardInterrupt:
            print('\n\nUscita manuale. A presto!')
            break
        except Exception as e:
            print(f'\nErrore: {e}\n')


print('Chat pronta!')
print('\nPer avviare la chat, esegui: interactive_chat()')

def answer_with_rag(question: str) -> dict:
    """
    Function for eval
    """
    # embed question
    q_emb = embed_question(question)        

    # retrieve documents
    docs, scores = retrieve(q_emb, top_k=TOP_K)

    # build context string
    context = "\n\n".join(d.page_content for d in docs)

    # generate answer
    answer = generate_answer(question, context)  

    return {
        "answer": answer,
        "context": context,
        "scores": scores,
    }

