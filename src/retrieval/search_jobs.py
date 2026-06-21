"""
search_jobs.py

Purpose:
    Take a search query (e.g. a resume summary, or a simple sentence
    describing someone's skills) and find the most semantically
    similar job postings stored in Qdrant.

This is the core "retrieval" step of our RAG system: given a query,
retrieve the most relevant stored items by MEANING, not exact keyword
matching.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "job_postings"


def search_jobs(query: str, retrieval_k: int = 20):
    """
    Stage 1: Embeds the query and retrieves a SHORTLIST of candidate
    jobs from Qdrant using fast vector similarity.

    We retrieve more than we ultimately want to show (retrieval_k=20
    by default) because this stage is a fast but imprecise filter --
    the more accurate reranking step (Stage 2) needs a wide enough
    shortlist to have good candidates to actually re-sort.
    """
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Embedding query: \"{query[:80]}...\"" if len(query) > 80 else f"Embedding query: \"{query}\"")
    query_vector = embedder.encode(query).tolist()

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    print(f"Retrieving top {retrieval_k} candidates (Stage 1: vector search)...\n")
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=retrieval_k,
    ).points

    return results


def rerank_results(query: str, results, top_k: int = 5):
    """
    Stage 2: Re-scores the Stage 1 shortlist using a cross-encoder,
    which reads the query and each job TOGETHER as a pair, rather
    than comparing independently-made vectors. This catches cases
    where Stage 1's fast vector search ranked something too high due
    to superficial similarity (e.g. tag-heavy postings with little
    real relevance -- see our documented "tag soup" limitation).

    Args:
        query: the original search text
        results: the Stage 1 shortlist from search_jobs()
        top_k: how many final results to return after reranking

    Returns:
        A list of (result, rerank_score) tuples, sorted by the new,
        more accurate cross-encoder score.
    """
    print(f"Loading reranker model '{RERANKER_MODEL_NAME}'...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    # The cross-encoder needs (query, job_text) PAIRS -- one pair per
    # candidate job. We build the job-side text the same way we did
    # for embedding, so the reranker sees comparable information.
    pairs = []
    for result in results:
        payload = result.payload
        job_text = f"Role: {payload.get('role_summary', '')}\nSkills: {', '.join(payload.get('tags', []))}"
        pairs.append((query, job_text))

    print(f"Reranking {len(pairs)} candidates (Stage 2: cross-encoder)...\n")
    scores = reranker.predict(pairs)

    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda pair: pair[1], reverse=True)

    return scored_results[:top_k]


def print_results(scored_results):
    """
    Pretty-prints reranked results, showing BOTH the original vector
    similarity score and the new cross-encoder rerank score, so we
    can see how reranking changed the ordering.
    """
    for rank, (result, rerank_score) in enumerate(scored_results, start=1):
        payload = result.payload
        print(f"#{rank}  (rerank score: {rerank_score:.3f} | original vector score: {result.score:.3f})")
        print(f"    Position: {payload.get('position')}")
        print(f"    Company:  {payload.get('company')}")
        print(f"    Summary:  {payload.get('role_summary')}")
        print(f"    Tags:     {', '.join(payload.get('tags', []))}")
        print(f"    Seniority: {payload.get('seniority_level')}")
        print()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add the sibling "parsing" folder to Python's import path, so we
    # can reuse parse_resume() instead of duplicating that logic here.
    parsing_folder = Path(__file__).resolve().parents[1] / "parsing"
    sys.path.insert(0, str(parsing_folder))
    from parse_resume import parse_resume

    if len(sys.argv) >= 2:
        # A PDF path was given -- parse the real resume and use its
        # text as the search query.
        pdf_path = sys.argv[1]
        print(f"Parsing resume: {pdf_path}")
        query = parse_resume(pdf_path)
    else:
        # No resume given -- fall back to a typed sample query, same
        # as before, so this script still works without arguments.
        query = "Experienced Python developer with backend and cloud infrastructure skills"

    # Stage 1: fast vector search retrieves a wide shortlist
    shortlist = search_jobs(query, retrieval_k=20)

    # Stage 2: cross-encoder reranks that shortlist for better precision
    final_results = rerank_results(query, shortlist, top_k=5)

    print_results(final_results)

    # Save the resume text + reranked results to a JSON file, so the
    # slow LLM explanation step (Stage 3) can run separately in Colab,
    # which has no access to this machine's local Qdrant instance.
    import json

    output_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "reranked_results.json"
    output_data = {
        "resume_text": query,
        "results": [
            {"payload": result.payload, "rerank_score": float(score)}
            for result, score in final_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved reranked results to: {output_path}")
    print("(This file can also be uploaded to Colab if local generation is too slow.)")

    # Stage 3: generate grounded fit explanations + skill gaps locally.
    # The file above was already saved BEFORE this runs, so even if
    # this step is slow, crashes, or you cancel it (Ctrl+C), your
    # reranked results are safe and can still be run in Colab instead.
    print("\nGenerating explanations locally (this may take a few minutes on CPU)...\n")

    generation_folder = Path(__file__).resolve().parents[1] / "generation"
    sys.path.insert(0, str(generation_folder))
    from explain_matches import explain_all_matches, print_explained_results

    try:
        explained = explain_all_matches(query, final_results)
        print("\n=== WITH EXPLANATIONS ===\n")
        print_explained_results(explained)
    except Exception as e:
        print(f"\nLocal explanation generation failed or was too slow: {e}")
        print(f"Your reranked results are still saved at: {output_path}")
        print("You can upload that file to Colab to generate explanations there instead.")