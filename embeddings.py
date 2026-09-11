"""Thin wrapper around Ollama's embeddings endpoint."""
import ollama

EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]