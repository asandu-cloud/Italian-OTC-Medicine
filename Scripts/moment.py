from Scripts.gemini3_model import retrieve_relevant_chunks

q = "Quali sono le differenze principali tra moment e tachipirina?"
retrieved, scores = retrieve_relevant_chunks(q, top_k=10, verbose=True)
