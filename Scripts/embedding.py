#!/usr/bin/env python
from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm.auto import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss

from dotenv import load_dotenv
from openai import OpenAI

# ----------------- Paths & Config -----------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


class Config:
    # Folder with JSON files created by your "extract_to_json" script
    JSON_FOLDER = PROJECT_ROOT / "kb_json"

    # Where to store FAISS index + embeddings + metadata
    CACHE_DIR = PROJECT_ROOT / ".cache"

    # Chunking
    CHUNK_SIZE = 700
    CHUNK_OVERLAP = 100

    # Embedding model
    EMBEDDING_MODEL = "text-embedding-3-small"


config = Config()


# ----------------- Chunk building from JSON -----------------

def build_chunks(config: Config) -> List[Dict]:
    """
    Read all JSON files in config.JSON_FOLDER and turn them into a list of
    dicts: {"text": ..., "document": filename, "chunk_id": int, "source": path}
    """

    if not config.JSON_FOLDER.exists():
        raise FileNotFoundError(
            f"JSON folder not found: {config.JSON_FOLDER}. "
            "Run your PDF→JSON extraction script first."
        )

    json_files = sorted(
        f for f in os.listdir(config.JSON_FOLDER)
        if f.lower().endswith(".json")
    )

    if not json_files:
        raise RuntimeError(f"No JSON files found in {config.JSON_FOLDER}")

    print(f"Found {len(json_files)} JSON docs to process")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    all_chunks: List[Dict] = []
    errors = 0

    for json_file in tqdm(json_files, desc="Processing JSON docs"):
        file_path = config.JSON_FOLDER / json_file

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            errors += 1
            continue

        raw_text = data.get("text", "") or ""
        filename = data.get("filename", json_file)
        source_path = data.get("source_path", filename)

        if not raw_text.strip():
            print(f"No valid text in {json_file}, skipping.")
            errors += 1
            continue

        pieces = splitter.split_text(raw_text)
        if not pieces:
            print(f"No chunks produced for {json_file}, skipping.")
            errors += 1
            continue

        for i, chunk in enumerate(pieces):
            all_chunks.append(
                {
                    "text": chunk,
                    "document": filename,   # original PDF name
                    "chunk_id": i,
                    "source": source_path,  # path to original PDF (optional but useful)
                }
            )

    print(f"\nChunking done:")
    print(f"  - Total chunks: {len(all_chunks)}")
    print(f"  - JSON files with issues: {errors}")

    return all_chunks


# ----------------- Embedding -----------------

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


# ----------------- Build index & save to .cache -----------------

def build_and_save_index(config: Config) -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    client = OpenAI()

    # checks for cache dir
    config.CACHE_DIR.mkdir(exist_ok=True)

    # chunk building from JSON
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
    emb_cache = config.CACHE_DIR / "embeddings.npy"
    idx_cache = config.CACHE_DIR / "faiss_index.idx"
    meta_cache = config.CACHE_DIR / "metadata.pkl"

    np.save(emb_cache, embeddings)
    faiss.write_index(index, str(idx_cache))
    with open(meta_cache, "wb") as f:
        pickle.dump(chunks, f)

    print("\nSaved all in .cache:")
    print(f"  - {emb_cache}")
    print(f"  - {idx_cache}")
    print(f"  - {meta_cache}")


if __name__ == "__main__":
    build_and_save_index(config)
