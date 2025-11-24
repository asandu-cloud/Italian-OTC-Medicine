import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import streamlit as st

# === IMPORT YOUR RAG LOGIC HERE ============================================
# Adjust "gemini3_model" to the actual module name where answer_question lives.
# That module should contain the code you currently have in the notebook
# (config, OpenAI embeddings, Gemini client, FAISS loading, etc.).

from gemini3_model import answer_question  # type: ignore

# If you want sidebar stats and you have these available, you can also import:
# from gemini3_model import config, chunks, index  # type: ignore


# === STREAMLIT PAGE CONFIG ==================================================
st.set_page_config(
    page_title="Chatbot Documenti Medici AIFA (RAG)",
    page_icon="⚕️",
    layout="centered",
)


# === HEADER ================================================================
st.title("⚕️ Chatbot Documenti Medici AIFA (RAG)")
st.write(
    "Fai domande sui medicinali OTC (Tachipirina, Aspirina, Moment, ecc.). "
    "Il sistema risponde basandosi sui documenti AIFA caricati nel sistema."
)


# === OPTIONAL SIDEBAR (if you import config/chunks/index) ===================
with st.sidebar:
    st.header("Impostazioni e informazioni")
    st.markdown(
        "- **Modello generativo**: Gemini 3\n"
        "- **Embeddings**: OpenAI `text-embedding-3-small`\n"
        "- **Tipo di sistema**: RAG sui foglietti illustrativi AIFA"
    )


# === SESSION STATE FOR CHAT HISTORY ========================================
# chat_history is a list of [user_msg, bot_msg], same format expected by format_history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# === RENDER PREVIOUS MESSAGES ==============================================
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)


# === NEW USER INPUT ========================================================
user_input = st.chat_input(
    "Scrivi qui la tua domanda sui medicinali OTC...",
    key="otc_chat_input"  # unique key for this chat input
)


if user_input:
    # Show the new user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call your RAG + Gemini pipeline
    with st.chat_message("assistant"):
        with st.spinner("Sto elaborando la risposta..."):
            result = answer_question(
                query=user_input,
                chat_history=st.session_state.chat_history,
                verbose=False,
            )

            answer = result.get("answer", "Errore: non ho trovato una risposta.")

            # Add confidence and sources, if present
            sources = result.get("sources")
            confidence = result.get("confidence")

            if isinstance(confidence, (int, float)):
                answer += f"\n\n*(Affidabilità media del contesto: {confidence:.0%})*"

            if sources:
                try:
                    # Case: list of strings
                    if isinstance(sources[0], str):
                        answer += "\n\n**Fonti consultate:**\n" + "\n".join(
                            f"- {s}" for s in sources
                        )
                    # Fallback: list of dicts with a 'document' field
                    elif isinstance(sources[0], dict) and "document" in sources[0]:
                        answer += "\n\n**Fonti consultate:**\n" + "\n".join(
                            f"- {s.get('document', 'sconosciuto')}" for s in sources
                        )
                except Exception:
                    # Do not break the UI just because sources are in an unexpected format
                    pass

            st.markdown(answer)

    # Update session chat history
    st.session_state.chat_history.append([user_input, answer])
