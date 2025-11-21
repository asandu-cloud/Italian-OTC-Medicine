#!/usr/bin/env python
"""
Script reads pdfs from medicinali
Embeds all 
Outputs to cache folder 
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm.auto import tqdm
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

class Config:
    PDF_FOLDER = PROJECT_ROOT / "medicinali"
    CACHE_DIR  = PROJECT_ROOT / ".cache"

    CHUNK_SIZE        = 700
    CHUNK_OVERLAP     = 100
    EMBEDDING_MODEL   = "text-embedding-3-small"


config = Config()

def clean_text(text: str) -> str:
    """Cleans text by removing non-printable characters (from your notebook)."""
    if not text or not isinstance(text, str):
        return ''
    cleaned = ''.join(c for c in text if c.isprintable() or c == '\n')
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract clean text from a single PDF (from your notebook)."""
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
            except Exception:
                continue
        return '\n'.join(content).strip()
    except Exception as e:
        print(f'PDF reading error for {pdf_path}: {e}')
        return ''

def build_chunks(config: Config) -> List[Dict]:
    """
    Read all PDFs in config.PDF_FOLDER and turn them into a list of
    dicts: {"text": ..., "document": pdf_filename, "chunk_id": int}
    """

    if not config.PDF_FOLDER.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {config.PDF_FOLDER}. "
            "Create it and put your medicinali PDFs inside."
        )

    pdf_files = sorted(
        f for f in os.listdir(config.PDF_FOLDER)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise RuntimeError(f"No PDFs found in {config.PDF_FOLDER}")

    print(f"Found {len(pdf_files)} PDFs to process")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    all_chunks: List[Dict] = []
    errors = 0

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = config.PDF_FOLDER / pdf_file
        raw_text = extract_text_from_pdf(str(file_path))

        if not raw_text:
            print(f"No valid text extracted from {pdf_file}, skipping.")
            errors += 1
            continue

        pieces = splitter.split_text(raw_text)
        if not pieces:
            print(f"No chunks produced for {pdf_file}, skipping.")
            errors += 1
            continue

        for i, chunk in enumerate(pieces):
            all_chunks.append(
                {
                    "text": chunk,
                    "document": pdf_file,
                    "chunk_id": i,
                }
            )

    print(f"\nChunking done:")
    print(f"  - Total chunks: {len(all_chunks)}")
    print(f"  - PDFs with issues: {errors}")

    return all_chunks

def embed_texts(
    texts: List[str],
    client: OpenAI,
    model: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embed a list of texts using OpenAI and return a float32 numpy array
    of shape (N, D).
    """
    all_vecs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_vecs = [np.array(e.embedding, dtype=np.float32) for e in resp.data]
        all_vecs.append(np.vstack(batch_vecs))

    embeddings = np.vstack(all_vecs)
    return embeddings


def build_and_save_index(config: Config) -> None:

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    client = OpenAI()

    # checks for cache dir 
    config.CACHE_DIR.mkdir(exist_ok=True)

    # chunk building
    chunks = build_chunks(config)
    texts = [c["text"] for c in chunks]

    # embed
    print("\nStarting embeddings...")
    embeddings = embed_texts(texts, client, config.EMBEDDING_MODEL)
    print(f"Embeddings shape: {embeddings.shape}")

    # normalize + build FAISS index (inner product)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index ready → {index.ntotal} vectors, dim={dim}")

    # save to cache
    emb_cache  = config.CACHE_DIR / "embeddings.npy"
    idx_cache  = config.CACHE_DIR / "faiss_index.idx"
    meta_cache = config.CACHE_DIR / "metadata.pkl"

    np.save(emb_cache, embeddings)
    faiss.write_index(index, str(idx_cache))
    with open(meta_cache, "wb") as f:
        pickle.dump(chunks, f)

    print("\nSaved all in .cache:")


if __name__ == "__main__":
    build_and_save_index(config)
