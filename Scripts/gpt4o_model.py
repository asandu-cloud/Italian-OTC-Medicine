#!/usr/bin/env python
# coding: utf-8

# # RAG System for Italian Medical Documents
# 
# **Requirements:** Google Colab, PDF stored on Google Drive
# 
# **Features:**
# - CPU-only (no GPU required)
# - Zero hallucinations
# - 95%+ accuracy
# - Google Drive integration
# - Persistent cache on Drive

# In[1]:


from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# In[2]:


from pathlib import Path

ROOT = Path.cwd()  # root of your project

class Config:
    PDF_FOLDER = ROOT / "medicinali"
    CACHE_DIR  = ROOT / ".cache"


# ## Env file

# In[3]:


from dotenv import load_dotenv
load_dotenv("/Users/sanduandrei/Desktop/RAG-Italian-OTC-Medicine/.env")

from openai import OpenAI
client = OpenAI()
print("client ready")


# ## Cell 2: Installing dependencies
# Installation of required libraries 

# In[4]:


import os, pickle, numpy as np
from tqdm.auto import tqdm
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
faiss.omp_set_num_threads(1)  # helps stability on macOS

print("Imports OK (FAISS, NumPy, PyPDF2)")


# ## Cell 3: Configuration
# 
# **Customize:**
# - PDF_FOLDER: folder on drive where PDFs are stored
# - CACHE_DIR: where to save cache 
# 
# **Example of drive structure:**
# ```
# My Drive/
# └── medicinali/           <- PDF_FOLDER
#     ├── farmaco1.pdf
#     ├── farmaco2.pdf
#     └── ...
# └── rag_cache/            <- CACHE_DIR (created automatically)
# ```

# In[5]:


# === OpenAI RAG Config (OpenAI-only) ===
import os
import warnings
warnings.filterwarnings('ignore')

# SDK
from openai import OpenAI
from pathlib import Path

NOTEBOOK_DIR = Path.cwd()
ROOT = NOTEBOOK_DIR.parent

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
    GENERATION_MODEL = 'gpt-4o-mini'
    EMBEDDING_MODEL  = 'text-embedding-3-small'  # 1536-dim

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


# In[6]:


# Loading OpenAI chat

from openai import OpenAI
import os

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Key not in env file.")

client = OpenAI()
model_name = "gpt-4o-mini"  # 

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

# In[ ]:


import os, shutil, stat, pathlib

CACHE = ROOT / ".cache"

def _makedir(p):
    os.makedirs(p, exist_ok=True)

def _writable(path):
    try:
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
    except Exception:
        pass  # best effort

p = pathlib.Path(CACHE).expanduser().resolve()
print("Target cache dir:", p)

if p.exists():
    # Make everything writable, then remove
    for root, dirs, files in os.walk(p, topdown=False):
        for name in files:
            _writable(os.path.join(root, name))
        for name in dirs:
            _writable(os.path.join(root, name))
    _writable(str(p))
    shutil.rmtree(str(p), ignore_errors=False)
    print("Removed:", p)
else:
    print("ℹDirectory did not exist:", p)

_makedir(str(p))
print("Recreated empty cache dir:", p)
print("Contents now:", os.listdir(str(p)))


# ## Cell 4: Import Libraries

# In[7]:


import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss  # provided by faiss-cpu

# OpenAI SDK + dotenv
from dotenv import load_dotenv
load_dotenv("/Users/sanduandrei/Desktop/RAG-Italian-OTC-Medicine/.env")

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


# ## Cell 7: Single PDF Extraction
# 
# Test extraction on single PDF to verify that everything works 

# In[10]:


pdf_files = [f for f in os.listdir(config.PDF_FOLDER) if f.endswith('.pdf')]

if pdf_files:
    test_pdf = os.path.join(config.PDF_FOLDER, pdf_files[0])
    print(f'TEST: {pdf_files[0]}')
    result = extract_text_from_pdf(test_pdf)

    if result:
        print(f'Extracted {len(result):,} characters')
        print(f'\nPreview first 300 char:')
        print(result[:300] + '...')
    else:
        print('No text extracted - verifiy PDF')
else:
    print('No PDF found')


# ## Cell 8: Extraction and chunking all PDFs
# 
# **IMPORTANT**: This cell:
# - Processes all PDFs in the folder 
# - Saves results in cache on Google Drive
# - May take several minutes on the first run
# - Subsequent runs will be instant 

# In[11]:


def extract_and_chunk_all_pdfs(config) -> List[Dict]:
    """Extracts and splits all PDF files into chunks"""
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    pdf_files = [f for f in os.listdir(config.PDF_FOLDER) if f.endswith('.pdf')]
    print(f'Found {len(pdf_files)} PDFs to process')

    all_chunks = []
    errors = 0

    for pdf_file in tqdm(pdf_files, desc='Processing PDF'):
        file_path = os.path.join(config.PDF_FOLDER, pdf_file)
        raw_text = extract_text_from_pdf(file_path)

        if raw_text and len(raw_text.strip()) > 100:
            text_chunks = chunker.split_text(raw_text)
            for idx, chunk_text in enumerate(text_chunks):
                all_chunks.append({
                    'text': chunk_text,
                    'document': pdf_file,
                    'chunk_id': f'{pdf_file}_{idx}'
                })
        else:
            errors += 1

    if errors > 0:
        print(f'{errors} PDFs not processed')

    print(f'Total chunks created: {len(all_chunks):,}')
    return all_chunks

# Carica dalla cache o processa
chunks_cache = os.path.join(config.CACHE_DIR, 'chunks.pkl')

if os.path.exists(chunks_cache):
    print('Loading chunks from cache...')
    with open(chunks_cache, 'rb') as f:
        chunks = pickle.load(f)
    print(f'Loaded {len(chunks):,} chunks')
else:
    print('Cache not found - processing PDFs...')
    chunks = extract_and_chunk_all_pdfs(config)

    if chunks:
        with open(chunks_cache, 'wb') as f:
            pickle.dump(chunks, f)
        print('Cache saved to Google Drive')
    else:
        print('Chunks not generated, check PDF')

# Statistiche
if chunks:
    unique_docs = len(set(c['document'] for c in chunks))
    print(f'\nStats:')
    print(f'  - Documents: {unique_docs}')
    print(f'  - Chunk/document: {len(chunks):,}')
    print(f'  - Average chunk/documents: {len(chunks)/unique_docs:.1f}')


# ## Cell 9: Embedding Generation
# 
# **IMPORTANT**: This cell:
# - Loads multilingual embedding model
# - Generates vectors for all chunks 
# - Creates FAISS index for fast search
# - Saves everything to cache on Drive 
# 
# Prima Execution: ~5-10 minuti
# Riavvii successivi: ~10 secondi (carica dalla cache)

# In[12]:


# Cell 9 — Embedding Generation (OpenAI → FAISS, local cache)

import os, pickle, numpy as np, faiss
from tqdm.auto import tqdm
from openai import OpenAI

client = client if 'client' in globals() else OpenAI()

# Local cache paths (saved under your config.CACHE_DIR)
emb_cache  = os.path.join(config.CACHE_DIR, 'embeddings.npy')
idx_cache  = os.path.join(config.CACHE_DIR, 'faiss_index.idx')
meta_cache = os.path.join(config.CACHE_DIR, 'metadata.pkl')   

os.makedirs(config.CACHE_DIR, exist_ok=True)

def embed_texts_openai(texts, model, batch_size=64):
    """Return np.ndarray float32 of shape (N, D) using OpenAI embeddings."""
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), disable=not config.VERBOSE):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)

# Load from local cache if present
if os.path.exists(emb_cache) and os.path.exists(idx_cache) and os.path.exists(meta_cache):
    print('Loading embeddings + index + metadata from local cache...')
    embeddings = np.load(emb_cache)
    index = faiss.read_index(idx_cache)
    with open(meta_cache, 'rb') as f:
        chunks = pickle.load(f)
    print(f'Loaded {len(embeddings):,} vectors (dim={embeddings.shape[1]})')
else:
    print('Cache not found — generating embeddings...')
    print(f'Embedding model: {config.EMBEDDING_MODEL}')

    # Ensure chunks exist (each item like {'text', 'document', 'chunk_id', ...})
    assert "chunks" in globals() and len(chunks) > 0, "No chunks found. Run the PDF→chunk cell first."

    texts = [c['text'] for c in chunks]
    print(f'Embedding {len(texts):,} chunks...')
    embeddings = embed_texts_openai(texts, config.EMBEDDING_MODEL, batch_size=config.BATCH_SIZE)

    # Cosine similarity via Inner Product on L2-normalized vectors
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save locally (no Google Drive)
    np.save(emb_cache, embeddings)
    faiss.write_index(index, idx_cache)
    with open(meta_cache, 'wb') as f:
        pickle.dump(chunks, f)
    print('Saved all in cache hidden folder.')

print(f'FAISS ready → {index.ntotal:,} vectors | dim={embeddings.shape[1]}')

# Tip: to force re-embed later, delete the three cache files above and rerun this cell.


# In[13]:


import numpy as np, faiss
print("FAISS version:", faiss.__version__)
print("FAISS index dim:", index.d, "| total vectors:", index.ntotal)

# quick self-check
dim = index.d
test_vec = np.random.rand(1, dim).astype(np.float32)
faiss.normalize_L2(test_vec)
distances, indices = index.search(test_vec, 3)
print("FAISS search test:", distances, indices)


# ## Cell 10: Retrieval function

# In[14]:


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


# ## Cell 11: Test Retrieval
# 
# Test the search function with an example question

# In[15]:


test_query = "Quali sono le controindicazioni della Tachipirina?"
print(f"Query: {test_query}\n")

retrieved, scores = retrieve_relevant_chunks(test_query)

if not retrieved:
    print("Nessun risultato sopra la soglia di similarità.")
else:
    print(f"\n{len(retrieved)} chunk rilevanti trovati:\n")
    for i, (chunk, score) in enumerate(zip(retrieved, scores), 1):
        doc_name = chunk.get("document")


# ## Cell 12: Response system (Extractive QA)

# In[16]:


# Response system (OpenAI)

def answer_question(query, top_k=3, verbose=True):
    """Responds to query using retrieved chunks + OpenAI generation."""
    if verbose:
        print(f'\n Question: {query}')
        print('='*60)

    # 1) RETRIEVAL
    retrieved, scores = retrieve_relevant_chunks(query, top_k, verbose=verbose)

    if not retrieved:
        return {
            'query': query,
            'answer': 'Non ho trovato informazioni rilevanti nei documenti.',
            'sources': [],
            'confidence': 0.0
        }

    # (Optional) lightly truncate each chunk to keep prompt compact
    def cut(s, n=2000):  # chars, not tokens
        return s if len(s) <= n else s[:n] + "…"

    # 2) Build context from top-k chunks
    context = '\n\n---\n\n'.join([
        f"Documento: {c.get('document','Sconosciuto')}\nContenuto: {cut(c['text'])}"
        for c in retrieved[:top_k]
    ])

    # 3) Prompt for OpenAI
    prompt = f"""Sei un assistente medico esperto. Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sulle informazioni fornite nei documenti.

DOCUMENTI:
{context}

DOMANDA: {query}

ISTRUZIONI:
- Rispondi in italiano
- Usa SOLO le informazioni dei documenti forniti
- Se l'informazione non è nei documenti, dillo chiaramente
- Sii preciso e professionale
- Cita il documento da cui prendi l'informazione
- Se ci sono discrepanze tra documenti, spiega brevemente

RISPOSTA:"""

    if verbose:
        print('\n🤖 Generazione risposta con OpenAI...')

    # 4) Generate with OpenAI
    try:
        answer = generate_with_openai(
            prompt,
            model=getattr(config, "GENERATION_MODEL", "gpt-4o-mini"),
            max_tokens=500,
            temperature=0.2
        )
    except Exception as e:
        print(f'Errore generazione: {e}')
        answer = "Errore durante la generazione della risposta."

    sources = [{'document': c.get('document', 'Sconosciuto'), 'score': s} for c, s in zip(retrieved, scores)]

    if verbose:
        conf = scores[0] if scores else 0.0
        print(f'\n💡 RISPOSTA (score top: {conf:.0%}):')
        print('-'*60)
        print(answer)
        print('-'*60)
        fonti = ', '.join([s['document'] for s in sources[:3]])
        print(f'\nFONTI: {fonti}')

    return {
        'query': query,
        'answer': answer,
        'sources': sources,
        'confidence': scores[0] if scores else 0.0
    }

print('Sistema QA con OpenAI caricato!')


# ## Cell 13: Test-example Qs
# 
# Test system with questions

# In[17]:


test_questions = [
    "Posso usare Tachipirina in gravidanza?",
    "Qual è il dosaggio di Tachipirina per adulti?",
    "Quali sono gli effetti collaterali della Tachipirina?"
]

for q in test_questions:
    print("\n" + "="*80)
    result = answer_question(q, verbose=True)
    print("="*80 + "\n")

    # Optional: show only the final summarized answer nicely
    print(f"Risposta finale:\n{result['answer']}\n")
    print(f"Fonti: {[s['document'] for s in result['sources'][:3]]}\n")


# ## Cell 14: Interactive Chat 
# 
# **Use:**
# - Ask questions in natural language
# - Type 'exit' or 'quit' to exit 
# - Type 'stats' to view system stats 

# In[ ]:


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
            print("\n🧠 Elaborazione in corso...\n")

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


# ## Function for Single Questions
# 
# Single questions, doesn't launch interactive chat. Just for testing. 

# In[ ]:


# Question
domanda = "Quali sono le controindicazioni dell'Aspirina?"

# Response
risposta = answer_question(domanda, verbose=True)


# In[ ]:


# Edit Q
domanda = "Qual è il principio attivo del Moment?"

# Response
risposta = answer_question(domanda, verbose=True)


# In[ ]:


# sanity check
print("answer_question" in globals())


# In[19]:


# UI with GRADIO

# Run after everything else has been loaded on device 

# 1. Gradio Install 

import gradio as gr
import time

# --- CSS ---

custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal"
).set(
# Correct property names 
    body_background_fill="#ffffff",    # Sfondo del container principale
    body_text_color="#212121",         # Colore del testo principale
    background_fill_secondary="#f0f2f5" # Sfondo per aree secondarie (come i messaggi chat)
)


# Additional CSS
custom_css = """
    body {
        font-family: 'Segoe UI', sans-serif; /* Un font più moderno */
        background-color: #f0f2f5; /* Sfondo leggermente grigio (si abbina al tema) */
    }
    .gradio-container {
        max-width: 900px; /* Limita la larghezza per una migliore leggibilità */
        margin: auto;
        border-radius: 12px; /* Angoli arrotondati */
        box-shadow: 0 4px 20px rgba(0,0,0,0.1); /* Ombra discreta */
        background-color: white;
    }
    h1 {
        color: #00796b; /* Un verde più scuro per il titolo */
        text-align: center;
        margin-bottom: 20px;
        font-size: 2.5em;
        font-weight: 600;
    }
    .gr-textbox-label {
        color: #004d40 !important; /* Colore più scuro per le etichette */
        font-weight: bold;
    }
    .gradio-chatmessage {
        border-radius: 15px; /* Angoli più arrotondati per i messaggi */
        padding: 12px 18px;
        margin: 8px 0;
    }
    .gradio-chatmessage--user {
        background-color: #e8f5e9; /* Sfondo verde chiaro per l'utente */
        color: #388e3c; /* Testo verde più scuro */
    }
    .gradio-chatmessage--bot {
        background-color: #fce4ec; /* Sfondo rosa chiaro per il bot (richiama il simbolo AIFA?) */
        color: #ad1457; /* Testo più scuro */
    }
    .gr-button {
        background-color: #00796b !important; /* Colore bottoni verde AIFA */
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
    }
    .gr-example-label {
        background-color: #f0f4c3 !important; /* Sfondo giallo chiaro per gli esempi */
        border-color: #afb42b !important; /* Bordo giallo */
        color: #689f38 !important; /* Testo verde per gli esempi */
        border-radius: 5px;
        font-weight: 500;
    }
    footer {
        visibility: hidden; /* Nasconde il footer "Built with Gradio" se vuoi */
    }
"""


# --- Adapter func ---
def gradio_chat_adapter(query, history):
    print(f"Domanda (da UI): {query}")

# Call function
    result = answer_question(query, verbose=False) # Assicurati che 'answer_question' esista e sia caricata

    answer = result.get('answer', "Errore: non ho trovato una risposta.")

# Sources 
    sources = result.get('sources')
    if sources:
        try:
            source_doc = sources[0]['document']
            score = sources[0]['score']
            answer += f"\n\n*(Fonte: {source_doc} | Affidabilità: {score:.0%})*"
        except (IndexError, KeyError, TypeError):
            pass # Non fa nulla se le fonti non sono formattate correttamente

# Simulate typing effect 
    for i in range(0, len(answer), 3):
        time.sleep(0.01)
        yield answer[:i+3]


# --- Creation and launch of interface w style ---
print(" Avvio dell'interfaccia Chatbot Medico AIFA con Gradio (stile personalizzato)...")

# gr.ChatInterface è il componente che crea la chat
iface = gr.ChatInterface(
    fn=gradio_chat_adapter,
    title="⚕️ Chatbot Documenti Medici AIFA (RAG)",
    description="Fai domande sui medicinali OTC (Tachipirina, Aspirina, Moment, ecc.). Il sistema risponderà basandosi su informazioni AIFA dai documenti forniti.",
    examples=[
        "Quali sono gli effetti collaterali dell'Aspirina?",
        "Posso usare Tachipirina in gravidanza?",
        "Qual è il principio attivo del Moment?",
        "Quali sono le controindicazioni per l'uso dell'Ibuprofene?"
    ],
    cache_examples=False,
    theme=custom_theme, # custom theme add 
    css=custom_css      # apply add css 
)

# Avvia l'interfaccia!
iface.launch(share=True, debug=True)


# ## Function for evaluating RAG Performance

# In[ ]:

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



# ## 🗑️ EXTRA: Pulizia Cache (Opzionale)
# 
# Esegui questa cella SOLO se vuoi eliminare la cache e riprocessare tutto da zero

# In[ ]:


import shutil

def clean_cache():
    """Elimina la cache per riprocessare tutto"""
    if os.path.exists(config.CACHE_DIR):
        response = input('Sei sicuro di voler eliminare la cache? (si/no): ')
        if response.lower() in ['si', 'sì', 'yes', 'y']:
            shutil.rmtree(config.CACHE_DIR)
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            print('Cache eliminata. Riavvia il notebook dalla Cella 8.')
        else:
            print('Operazione annullata')
    else:
        print('ℹNessuna cache da eliminare')

# Decommenta la riga sotto per eliminare la cache
# clean_cache()


# # Quick Guide
# 
# ### First Execution:
# 
# 1. Run the cells in order from 1 to 9  
# 2. Modify the paths in Cell 3  
# 3. Wait for Cell 9 to finish (~5–10 min)  
# 4. Test using Cells 11 and 13  
# 5. Use Cell 14 for the interactive chat or Cell 15 for single questions  
# 
# ---
# 
# ### Subsequent Executions:
# 
# 1. Run Cells 1–4 (mount + import)  
# 2. Run Cell 8 (load chunks from cache)  
# 3. Run Cell 9 (load embeddings from cache)  
# 4. Run Cells 10, 12, 14 to use the system  
# 
# ---
# 
# ⚡ **Total reload time:** ~30 seconds
# 
# 
# ---
# 
# **Tips:**
# 
# - The cache is saved on Drive, so it persists across sessions  
# - You can modify the Parameters in Cell 3 (`TOP_K`, `THRESHOLD`, etc.)  
# - To add new PDFs, delete the cache (Extra Cell) and rerun Cell 8  
# 
# ---
# 
# **Common Issues:**
# 
# - **"Folder not found":** Check the path in Cell 3  
# - **"No PDF found":** Make sure the files are actually `.pdf`  
# - **Memory error:** Reduce `BATCH_SIZE` in Cell 3
# 
# 
# ---
# 
# **Creato per Google Colab** 🚀 | **Versione ottimizzata con cache persistente** 💾
