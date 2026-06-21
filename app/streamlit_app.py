"""
streamlit_app.py

Purpose:
    A simple web UI for the job-matching RAG system. Lets a user
    upload a resume PDF, see ranked job matches, and optionally click
    "Explain this match" to get an LLM-generated fit explanation and
    skill gaps for a SINGLE job at a time (kept on-demand and
    per-job, rather than upfront for all results, since LLM
    generation is slow on typical hardware).

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make our existing src/ modules importable from this app file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "parsing"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "generation"))

from parse_resume import parse_resume
from search_jobs import search_jobs, rerank_results
from explain_matches import generate_explanation


@st.cache_resource
def load_explanation_model():
    """
    Loads the Phi-3-mini model ONCE per app session, instead of on
    every single button click. @st.cache_resource tells Streamlit:
    "run this function once, then hand back the same already-loaded
    object on every future call" -- turning a slow multi-minute reload
    into an instant lookup after the first click.
    """
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model="microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",
    )


st.set_page_config(page_title="Job Match RAG", page_icon="🔎")
st.title("🔎 Job Match Finder")
st.write("Upload your resume to find the best-matching remote job postings.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Streamlit gives us the uploaded file in memory; we save it to a
    # temp path on disk since parse_resume() expects a file path.
    temp_path = PROJECT_ROOT / "data" / "raw" / "uploaded_resume.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading your resume..."):
        resume_text = parse_resume(str(temp_path))

    with st.spinner("Searching for matching jobs..."):
        shortlist = search_jobs(resume_text, retrieval_k=20)
        final_results = rerank_results(resume_text, shortlist, top_k=5)

    st.success(f"Found {len(final_results)} matching jobs.")

    # st.session_state lets us remember which jobs the user has
    # already clicked "Explain" on, across Streamlit's re-runs.
    if "explanations" not in st.session_state:
        st.session_state.explanations = {}

    for rank, (result, rerank_score) in enumerate(final_results, start=1):
        payload = result.payload
        job_key = f"{payload.get('position')}_{payload.get('company')}_{rank}"

        with st.container(border=True):
            st.subheader(f"#{rank}  {payload.get('position')}")
            st.write(f"**Company:** {payload.get('company')}")
            st.write(f"**Summary:** {payload.get('role_summary')}")
            st.write(f"**Tags:** {', '.join(payload.get('tags', []))}")
            st.write(f"**Seniority:** {payload.get('seniority_level')}")

            if job_key in st.session_state.explanations:
                explanation = st.session_state.explanations[job_key]
                st.info(f"**Why it fits:** {explanation['fit_explanation']}")
                st.warning(f"**Skill gaps:** {explanation['skill_gaps']}")
            else:
                if st.button("Explain this match", key=f"explain_{job_key}"):
                    with st.spinner("Generating explanation (first click loads the model, may take a minute)..."):
                        generator = load_explanation_model()
                        explanation = generate_explanation(generator, resume_text, payload)
                        st.session_state.explanations[job_key] = explanation
                    st.rerun()