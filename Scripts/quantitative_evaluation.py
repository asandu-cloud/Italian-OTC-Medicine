from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from Scripts.gpt4o_model import answer_with_rag as gpt4o_answer
from Scripts.gpt5_model import answer_with_rag as gpt5_answer
from Scripts.metrics import similarity_to_reference

EVAL_FILE = Path("eval_questions.csv")   # id;question;reference_answer;category

@dataclass
class RAGVariant:
    name: str
    answer_fn: callable

variants = [
    RAGVariant("gpt5", gpt5_answer),
    RAGVariant("gpt4o",gpt4o_answer)
]

def load_eval_questions():
    return pd.read_csv(EVAL_FILE, sep=";")

def main():
    questions = load_eval_questions()
    rows = []

    for var in variants:
        print(f"\n=== Evaluating {var.name} ===")
        for _, row in questions.iterrows():
            qid = row["id"]
            qtext = row["question"]
            ref   = row["reference_answer"]

            out = var.answer_fn(qtext) 

            sim = similarity_to_reference(out["answer"], ref)

            rows.append({
                "model": var.name,
                "id": qid,
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "total_tokens": out["total_tokens"],
                "t_retrieval": out["t_retrieval"],
                "t_generation": out["t_generation"],
                "t_total": out["t_total"],
                "sim_ref": sim,
            })

    df = pd.DataFrame(rows)
    df.to_csv("rag_eval_quantitative.csv", index=False)

    # summary stats per model
    summary = df.groupby("model").agg({
        "prompt_tokens": ["mean", "sum"],
        "completion_tokens": ["mean", "sum"],
        "total_tokens": ["mean", "sum"],
        "t_total": ["mean"],
        "sim_ref": ["mean"],
    })
    print(summary)

if __name__ == "__main__":
    main()

