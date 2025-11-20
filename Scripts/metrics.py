import numpy as np
from openai import OpenAI

client = OpenAI()

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def similarity_to_reference(answer: str, reference: str) -> float:
    """
    0–1 similarity between model answer and reference answer.
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[answer, reference],
    )
    e_ans = resp.data[0].embedding
    e_ref = resp.data[1].embedding
    return cosine(e_ans, e_ref)
