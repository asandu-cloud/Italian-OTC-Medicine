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
│   └── scrap_for_gpt_script.py     # Utility script (WIP)
│
├── .cache/                         # Shared hidden cache (embeddings, FAISS index)
├── .venv/                          # Virtual environment (not tracked)
├── .env                            # API keys and environment variables
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation

## Installation
git clone https://github.com/asandu-cloud/Italian-OTC-Medicine.git
cd Italian-OTC-Medicine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## Create your .env file:

OPENAI_API_KEY=...
GEMINI_API_KEY=...
TELEGRAM_TOKEN=...

## How to Run
### Step 1 — Embed PDFs
python Scripts/embedding.py

## Step 2 — Launch Streamlit App
### streamlit run Scripts/streamlit_app.py

## Step 3 — Start Telegram Bot
### python Scripts/telegram_bot.py

## Step 4 — Run Model Evaluation
### python Scripts/quantitative_evaluation.py