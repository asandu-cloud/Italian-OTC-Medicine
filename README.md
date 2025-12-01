## Project Structure

```text
Italian-OTC-Medicine/
│
├── medicinali/                     # Source PDF documents used to build the Knowledge Base
│
├── Notebooks/
│   ├── Mistral PM.ipynb            # Standalone RAG notebook (Mistral)
│   ├── gpt-5.1_model.ipynb         # Feeds into Python scripts for GPT-5.1 RAG
│   ├── gpt4o_model.ipynb           # Feeds into Python scripts for GPT-4o RAG
│   └── .cache/                     # Notebook-specific temporary cache
│
├── Scripts/
│   ├── clear_cache.py              # Clears the hidden embedding + FAISS cache
│   ├── embedding.py                # Builds embeddings + FAISS index for all models
│   ├── gemini3_model.py            # Full RAG implementation using Gemini 3
│   ├── streamlit_app.py            # Streamlit interface running Gemini RAG
│   ├── telegram_bot.py             # Telegram bot running Gemini RAG
│   ├── quantitative_evaluation.py  # Evaluation pipeline for all models
│   ├── metrics.py                  # Computes evaluation metrics
│   ├── gpt4o_model.py              # Python script version of gpt4o model. Used only for the quantitative evaluation
│   ├── gpt5_model.py               # Python script version of gpt5 model. Used only for the quantitative evaluation and the telegram bot.
│
├── .cache/                         # Shared hidden cache (embeddings, FAISS index)
├── .venv/                          # Virtual environment (not tracked)
├── .env                            # API keys and environment variables
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

## Installation
```text
git clone https://github.com/asandu-cloud/Italian-OTC-Medicine.git
cd Italian-OTC-Medicine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Create your .env file:
```text
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

## How to Run
### Embed PDFs by running embedding.py script
```text
Content of PDFs used for KB are stored under the kb_json folder.
python Scripts/embedding.py
```
## For GPT4o, GPT5 and Mistral
```text
Run respective notebooks 
- GPT4o -> gpt4o_model.ipynb
- GPT5.1 -> gpt5_model.ipynb
- Mistral PM -> Mistal PM.ipynb

Before running Mistral run clear_cache.py due to conflicting embedding
After running Mistral run clear_cache.py again followed by embedding.py
```
## Gemini 
```text
python Scripts/streamlit_app_gemini.py
```

## Step 4 — Run Model Evaluation
```text
python Scripts/quantitative_evaluation.py
```