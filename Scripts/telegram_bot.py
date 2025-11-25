# Uses Gemini3 as backend

import os
from collections import defaultdict

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from gemini3_model import answer_question, chunks, index  

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
user_histories = defaultdict(list)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start command handler.
    Sends a welcome message in Italian.
    """
    await update.message.reply_text(
        "Ciao! Sono il chatbot sui farmaci da banco (OTC) basato sui documenti AIFA.\n"
        "Scrivimi una domanda, ad esempio:\n"
        "- Quali sono gli effetti collaterali della Tachipirina?\n"
        "- Posso prendere Moment a stomaco vuoto?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /help command handler.
    """
    await update.message.reply_text(
        "Puoi farmi domande sui farmaci da banco (OTC) italiani. "
        "Cercherò la risposta nei foglietti illustrativi AIFA caricati nel sistema.\n\n"
        "Esempi:\n"
        "- Posso usare Tachipirina in gravidanza?\n"
        "- Qual è il principio attivo del Moment?"
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /stats command handler.
    Shows some basic RAG stats.
    """
    num_chunks = len(chunks)
    num_docs = len(set(c.get("document", "?") for c in chunks))
    idx_size = index.ntotal

    await update.message.reply_text(
        f"Statistiche del sistema RAG:\n"
        f"- Chunk totali: {num_chunks}\n"
        f"- Documenti unici: {num_docs}\n"
        f"- Dimensione indice FAISS: {idx_size} vettori"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Main message handler.
    - Takes the user message
    - Looks up the per-user history
    - Calls answer_question(query, chat_history=history)
    - Sends back the answer in Italian
    - Updates the history
    """
    if update.message is None or update.message.text is None:
        return

    user_id = update.effective_user.id
    question = update.message.text.strip()

    # Retrieve history for this user (list[[user_msg, bot_msg], ...])
    history = user_histories[user_id]

    # Call the RAG backend
    try:
        result = answer_question(
            query=question,
            chat_history=history,
            verbose=False,
        )
    except Exception as e:
        await update.message.reply_text(
            f"Si è verificato un errore interno: {e}"
        )
        return

    answer = result.get("answer", "Errore: non sono riuscito a generare una risposta.")

    # Append sources/confidence (optional)
    sources = result.get("sources", [])
    confidence = result.get("confidence")

    if isinstance(confidence, (int, float)):
        answer += f"\n\n*(Affidabilità media del contesto: {confidence:.0%})*"

    if sources:
        answer += "\n\nFonti consultate:"
        for src in sources:
            answer += f"\n- {src}"

    # Send answer
    await update.message.reply_text(answer)

    # Update history for this user
    history.append([question, answer])
    user_histories[user_id] = history


def main():
    """
    Entry point for the Telegram bot.
    """
    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not found in environment. "
            "Add it to your .env file."
        )

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats))

    # Text messages (non-command)
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    print("Telegram bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
