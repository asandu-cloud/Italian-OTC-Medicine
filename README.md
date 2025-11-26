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
│   ├── clear_cache.py              # Deletes hidden cache directory
│   ├── embedding.py                # Generates embeddings + FAISS index for all models
│   ├── gemini3_model.py            # Full RAG pipeline using Gemini 3
│   ├── streamlit_app.py            # Streamlit interface running Gemini RAG
│   ├── telegram_bot.py             # Telegram bot running Gemini RAG
│   ├── quantitative_evaluation.py  # Runs evaluation jobs across all models
│   ├── metrics.py                  # Computes quantitative metrics (accuracy, recall, etc.)
│   └── scrap_for_gpt_script.py     # Utility scripts generated from notebooks (TBD)
│
├── .cache/                         # Hidden shared cache (embeddings, FAISS index, metadata)
├── .venv/                          # Virtual environment (not tracked)
├── .env                            # API keys + environment variables
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation (this file)
