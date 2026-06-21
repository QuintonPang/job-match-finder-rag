"""
explain_matches.py

Purpose:
    For each reranked job match, generate two grounded pieces of text:
    1. WHY this job fits, based on the actual retrieved job data
    2. WHAT skills are missing, comparing the resume against the job's
       real tags/requirements

This is the "Generation" half of our RAG system. Critically, the LLM
is only ever given REAL, RETRIEVED data (the resume text + the actual
job's role_summary/tags) -- it is never asked to invent facts about a
job it wasn't shown, which is what keeps this grounded rather than
hallucinated.

Design choice, informed by Lesson 2's hard-won lessons:
    We use two separate, focused calls per job (fit explanation, then
    gaps) rather than one combined call, and we ask for plain prose,
    not structured JSON/lists -- this avoids nearly all the formatting
    fragility we fought through when cleaning job postings.
"""

from transformers import pipeline

EXPLANATION_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Tags that describe the WORK ARRANGEMENT or SENIORITY, not an actual
# skill -- a resume can never "have" or "lack" these, so we exclude
# them before checking for gaps at all.
NON_SKILL_TAGS = {
    "senior", "junior", "mid", "lead", "exec", "executive",
    "digital nomad", "full time", "part time", "remote", "freelance",
    "non tech", "technical",
}


def find_skill_gaps(resume_text: str, tags: list[str]) -> list[str]:
    """
    Deterministic, simple check: for each REAL skill tag (after
    excluding generic descriptors), does that word appear anywhere in
    the resume text?

    Why plain string matching instead of asking the LLM:
        Checking "does this exact word appear in this text" is a
        precise lookup task, not a task requiring interpretation or
        judgment -- exactly the kind of thing small LLMs are
        unreliable at (we observed this firsthand: the model missed
        an obvious Golang gap during testing) and plain code does
        perfectly, every time. We reserve the LLM for genuinely fuzzy
        judgment calls, not exact-match lookups.
    """
    resume_lower = resume_text.lower()
    gaps = []

    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower in NON_SKILL_TAGS:
            continue
        if tag_lower not in resume_lower:
            gaps.append(tag)

    return gaps


def build_fit_prompt(resume_text: str, job_payload: dict) -> str:
    return f"""Candidate's resume:
{resume_text[:1500]}

Job posting:
Title: {job_payload.get('position')}
Company: {job_payload.get('company')}
Summary: {job_payload.get('role_summary')}
Tags: {', '.join(job_payload.get('tags', []))}

In 2-3 sentences, explain why this candidate's background fits this specific job. Only use facts from the resume and job posting above. Be specific, not generic.

Explanation:"""


def build_gaps_prompt(resume_text: str, job_payload: dict) -> str:
    return f"""Candidate's resume:
{resume_text[:1500]}

Job posting:
Title: {job_payload.get('position')}
Tags: {', '.join(job_payload.get('tags', []))}

Looking only at the job's tags above, list which ones do NOT clearly appear in the candidate's resume. If all tags are covered, say "No major gaps found." Keep it to one short sentence.

Missing skills:"""


def generate_explanation(generator, resume_text: str, job_payload: dict) -> dict:
    """
    Runs both focused calls (fit + gaps) for a single job match, and
    returns plain readable strings -- no JSON parsing needed, since we
    deliberately asked for prose, not structured output.
    """
    fit_prompt = build_fit_prompt(resume_text, job_payload)
    fit_result = generator(fit_prompt, max_new_tokens=120, do_sample=False, repetition_penalty=1.1)
    fit_text = fit_result[0]["generated_text"][len(fit_prompt):].strip().split("\n")[0]

    gaps_prompt = build_gaps_prompt(resume_text, job_payload)
    gaps_result = generator(gaps_prompt, max_new_tokens=80, do_sample=False, repetition_penalty=1.1)
    gaps_text = gaps_result[0]["generated_text"][len(gaps_prompt):].strip().split("\n")[0]

    return {
        "fit_explanation": fit_text,
        "skill_gaps": gaps_text,
    }


def explain_all_matches(resume_text: str, scored_results) -> list[dict]:
    """
    Takes the final reranked results (from search_jobs.py) and adds
    a fit explanation + skill gaps to each one.
    """
    print(f"Loading explanation model '{EXPLANATION_MODEL_NAME}'...")
    generator = pipeline(
        "text-generation",
        model=EXPLANATION_MODEL_NAME,
        device_map="auto",
    )

    explained = []
    for result, rerank_score in scored_results:
        payload = result.payload
        print(f"Generating explanation for: {payload.get('position')} @ {payload.get('company')}")
        explanation = generate_explanation(generator, resume_text, payload)
        explained.append({
            "payload": payload,
            "rerank_score": rerank_score,
            **explanation,
        })

    return explained


def print_explained_results(explained: list[dict]):
    for rank, item in enumerate(explained, start=1):
        payload = item["payload"]
        print(f"#{rank}  {payload.get('position')} @ {payload.get('company')}")
        print(f"    Why it fits: {item['fit_explanation']}")
        print(f"    Skill gaps:  {item['skill_gaps']}")
        print()