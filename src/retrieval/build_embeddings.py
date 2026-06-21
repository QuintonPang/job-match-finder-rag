"""
build_embeddings.py

Purpose:
    Convert each cleaned job posting into an embedding (a list of
    numbers capturing its meaning) and store it in Qdrant, our vector
    database, so we can later search for jobs by meaning rather than
    exact keyword matching.

Why we combine role_summary + skills for embedding (not the whole job):
    We want the embedded text to represent the JOB'S SUBSTANCE --
    what the role actually involves and what skills it needs. Fields
    like job title or seniority are better used as filters/metadata
    alongside the vector search, not blended into the text we embed,
    since exact-match fields and meaning-based fields serve different
    purposes.

Input:
    data/processed/cleaned_jobs.json

Output:
    Job embeddings stored in a running Qdrant instance
    (expects Qdrant running at http://localhost:6333)
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "job_postings"


def build_embedding_text(job: dict) -> str:
    """
    Combines the fields we want to be SEARCHABLE BY MEANING into one
    block of text for a single job posting.

    Why we use RemoteOK's own "tags" field instead of our LLM-extracted
    "skills" field:
        We measured a 96% empty rate on the LLM-extracted skills list
        across our real 100-job dataset -- the small model was simply
        too unreliable at this specific task, despite extensive prompt
        engineering. RemoteOK already provides its own "tags" field
        (e.g. ["python", "django", "aws"]) as clean, structured data,
        with zero LLM calls needed. This is a good general lesson:
        before reaching for more AI to solve a data problem, check
        whether the structured data you need already exists in your
        source.
    """
    cleaned = job.get("cleaned", {})
    role_summary = cleaned.get("role_summary", "")
    tags = job.get("tags", [])

    tags_text = ", ".join(tags) if tags else ""

    return f"Role: {role_summary}\nSkills: {tags_text}"


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "cleaned_jobs.json"

    print(f"Loading cleaned jobs from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} cleaned job postings.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dimension = embedder.get_sentence_embedding_dimension()
    print(f"Embedding model ready. Vector size: {embedding_dimension}")

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    # Create (or recreate) the collection that will hold our job
    # embeddings. A "collection" in Qdrant is roughly like a table
    # in a normal database -- a named group of stored vectors.
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' ready.")

    print("Embedding and uploading jobs...")
    points = []
    for i, job in enumerate(jobs):
        text = build_embedding_text(job)
        vector = embedder.encode(text).tolist()

        # We store the original job data as "payload" -- extra info
        # attached to the vector, so when we later find a match, we
        # can immediately show the real job details, not just a number.
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "position": job.get("position", ""),
                    "company": job.get("company", ""),
                    "role_summary": job.get("cleaned", {}).get("role_summary", ""),
                    "tags": job.get("tags", []),
                    "seniority_level": job.get("cleaned", {}).get("seniority_level", "unspecified"),
                    "apply_url": job.get("apply_url", ""),
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uploaded {len(points)} job embeddings to Qdrant.")


if __name__ == "__main__":
    main()