from pathlib import Path
import os
import sys  # <-- NEW

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---- Paths / env ----
if "__file__" in globals():
    SCRIPTS_DIR = Path(__file__).resolve().parent
    ROOT = SCRIPTS_DIR.parent
else:
    ROOT = Path.cwd().resolve()
    SCRIPTS_DIR = ROOT / "Scripts"

# Make sure Scripts/ is on sys.path so we can `import gpt5_model`
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

load_dotenv(ROOT / ".env")

TELEGRAM_BOT_TOKEN_5 = os.getenv("TELEGRAM_BOT_TOKEN_5")
if not TELEGRAM_BOT_TOKEN_5:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

from gpt5_model import answer_question, config  # type: ignore



# ---- Handlers ----

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Ciao! 👋 Sono il bot per i farmaci OTC italiani.\n"
        "Scrivimi una domanda (es. 'Posso prendere Tachipirina e Moment insieme?')."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Sono un assistente sui farmaci da banco italiani (OTC).\n\n"
        "Esempi di domande:\n"
        "• Qual è la dose massima giornaliera di Tachipirina 1000?\n"
        "• Posso prendere Moment a stomaco vuoto?\n"
        "• Quali sono le controindicazioni di Momendol?\n\n"
        "Rispondo solo in base ai foglietti illustrativi caricati."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    # Optional: quick command-style exit
    if user_text.lower() in {"exit", "quit"}:
        await update.message.reply_text("Ok, a presto! 👋")
        return

    # Call your RAG model
    try:
        result = answer_question(user_text, verbose=False)
        answer = result.get("answer", "Si è verificato un errore nella generazione della risposta.")

        # Telegram has a 4096 char limit → truncate if necessary
        MAX_LEN = 4000
        if len(answer) > MAX_LEN:
            answer = answer[:MAX_LEN] + "\n\n[Risposta tagliata perché troppo lunga.]"

        await update.message.reply_text(answer)

    except Exception as e:
        # Basic error handling; you can log more details to console
        await update.message.reply_text(
            "Si è verificato un errore interno mentre preparavo la risposta. "
            "Riprova tra poco."
        )
        print(f"[ERROR] handle_message: {e}")


# ---- Main ----

def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN_5).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )

    print("Telegram bot GPT-5.1 RAG in esecuzione…")
    application.run_polling()


if __name__ == "__main__":
    main()