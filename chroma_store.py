"""
Builds a per-CV ChromaDB collection of evidence chunks (skills, experience
bullets, projects, certifications, achievements), each with a precomputed
embedding from Ollama. Reused by both the Matcher and the later
Grounding/Evidence Retrieval layer — one vector store, two consumers.
"""
import chromadb
from embeddings import embed_text
from schemas import CVData

_client = chromadb.PersistentClient(path="./chroma_store")


def _cv_chunks(cv: CVData) -> list[dict]:
    chunks = []
    idx = 0

    for edu in cv.education:
        text = f"{edu.degree}, {edu.institution}"
        if edu.dates:
            text += f" ({edu.dates})"
        chunks.append({"id": f"chunk_{idx}", "text": text, "source": "education"})
        idx += 1

    for skill in cv.skills:
        chunks.append({"id": f"chunk_{idx}", "text": skill, "source": "skills"})
        idx += 1

    for exp in cv.experience:
        for bullet in exp.description:
            chunks.append({
                "id": f"chunk_{idx}",
                "text": f"{exp.role} at {exp.company}: {bullet}",
                "source": "experience",
            })
            idx += 1

    for proj in cv.projects:
        chunks.append({"id": f"chunk_{idx}", "text": f"{proj.name}: {proj.description}", "source": "projects"})
        idx += 1

    for cert in cv.certifications:
        chunks.append({"id": f"chunk_{idx}", "text": cert, "source": "certifications"})
        idx += 1

    for ach in cv.achievements:
        chunks.append({"id": f"chunk_{idx}", "text": ach, "source": "achievements"})
        idx += 1

    if cv.personal_summary:
        chunks.append({"id": f"chunk_{idx}", "text": cv.personal_summary, "source": "summary"})
        idx += 1

    return chunks


def build_cv_collection(cv: CVData, collection_name: str = "cv_evidence"):
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass

    collection = _client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    chunks = _cv_chunks(cv)
    if not chunks:
        return collection

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]
    embeddings = [embed_text(doc) for doc in documents]

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return collection


def query_evidence(collection, requirement_text: str, top_k: int = 3):
    query_embedding = embed_text(requirement_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    hits = []
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        hits.append({"text": doc, "similarity": 1 - dist, "source": meta.get("source")})
    return hits