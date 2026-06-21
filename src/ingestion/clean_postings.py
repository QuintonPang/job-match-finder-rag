"""
clean_postings.py

Purpose:
    Take raw job postings (full of marketing fluff like "rockstar",
    "ninja", "fast-paced dynamic environment") and use an LLM to
    extract the actual substance: a clean role summary, required
    skills, nice-to-have skills, and seniority level.

Why this matters for RAG:
    Later, we'll convert job postings into embeddings (numeric vectors)
    so we can search them. If we embed noisy, fluff-heavy text, the
    embedding may waste "attention" on marketing language instead of
    the actual skills and requirements. Cleaning first means the
    embeddings we generate later are sharper and more meaningful.

Input:
    data/raw/sample_remoteok_jobs.json  (or your real fetched file)

Output:
    data/processed/cleaned_jobs.json
"""

import json
from pathlib import Path
import torch
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList

# We use a small, free, open-source instruct model. It runs on CPU,
# just slower than a GPU would be. That's an acceptable tradeoff for
# a free, self-hosted project.
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


class StopOnBalancedJSON(StoppingCriteria):
    """
    A custom rule that tells the model's generation loop: "stop the
    moment a complete, balanced {...} JSON object has been produced."

    Why we need this:
        Small models like Phi-3-mini don't always reliably decide on
        their own when to stop. We saw it stop too early (just "{")
        and also stop too late (finishing the JSON, then hallucinating
        an entirely new, unrelated job posting). Instead of hoping the
        model "knows when to stop," we check the actual generated text
        after every new token and force a stop as soon as the braces
        balance out -- meaning a full JSON object has been written.
    """

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length  # length of input prompt, in tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Decode only the NEW tokens generated so far (skip the prompt)
        generated_ids = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # We only start counting braces once we've actually seen a "{"
        if "{" not in text:
            return False

        # Count whether every "{" has a matching "}" by this point.
        open_braces = text.count("{")
        close_braces = text.count("}")

        # Stop once braces are balanced AND we've seen at least one pair
        # (this avoids stopping on an empty/incomplete "{}" immediately)
        return open_braces > 0 and open_braces == close_braces


def build_prompt_for_field(job: dict, field_instruction: str) -> str:
    """
    Builds a focused prompt asking for ONE specific field only.

    Why we split into one field per call instead of asking for all 4
    fields in a single JSON object:
        We found that asking a small model to produce a full 4-field
        JSON object in one shot is unreliable -- it tends to write the
        first field, then close the JSON object early instead of
        continuing to the rest. Asking one simple, focused question
        at a time is a much easier task for a small model to get
        right consistently, even though it costs us more model calls.
    """
    return f"""Job Title: {job['position']}
Company: {job['company']}
Job Description:
{job['description']}

{field_instruction}
Answer with ONLY the requested value, nothing else. No explanation, no labels, no quotes around the whole answer.

Answer:"""


def get_role_summary(generator, job: dict) -> str:
    prompt = build_prompt_for_field(
        job,
        "In one factual sentence, summarize what this role actually involves. No marketing language."
    )
    # Raised from 80 -> 150: some summaries were getting cut off
    # mid-sentence for longer, more detailed job postings.
    result = generator(prompt, max_new_tokens=150, do_sample=False, repetition_penalty=1.1)
    text = result[0]["generated_text"][len(prompt):].strip()
    return text.split("\n")[0].strip()


def get_skills(generator, job: dict) -> list[str]:
    """
    Asks for ALL skills mentioned for this role, as a single numbered
    list -- no required/nice-to-have split.

    Why we dropped the required vs. nice-to-have split:
        We spent a long time trying to get the model to reliably
        separate skills into two categories, and kept hitting new
        failure modes each time we fixed the last one (duplicated
        skills across both lists, run-on text with no separators,
        the model explaining why it COULDN'T categorize thin postings
        instead of just returning an empty list). A single list
        removes the specific complexity that kept causing problems,
        at the cost of losing the required/nice-to-have distinction.
        This is a deliberate scope reduction, not a bug -- sometimes
        the right fix is a simpler goal, not a cleverer parser.
    """
    instruction = (
        "This is a public job advertisement, not a resume or a person's data. "
        "Your task is text extraction only: copy out the skill/tool/qualification "
        "keywords that already appear in the advertisement text above, as a "
        "NUMBERED list, like this:\n"
        "1. first skill\n"
        "2. second skill\n\n"
        "If the posting does not mention any specific skills at all, "
        "respond with exactly: 1. none"
    )

    prompt = build_prompt_for_field(job, instruction)
    result = generator(prompt, max_new_tokens=180, do_sample=False, repetition_penalty=1.15)
    text = result[0]["generated_text"][len(prompt):].strip()

    return _parse_numbered_list(text)


import re


def _parse_numbered_list(text: str) -> list[str]:
    """
    Parses a numbered list (1. 2. 3. ...) out of model output, even
    when real line breaks are missing and everything runs together
    on one line. Returns an empty list if no real items are found,
    including when the model writes "none" or explains that no
    skills were mentioned, rather than actually listing any.
    """
    tokens = re.split(r"(\d+\.\s*)", text)

    items: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token or re.fullmatch(r"\d+\.", token):
            continue

        cleaned = token.strip().strip('"').strip()
        if cleaned and cleaned.lower() not in ("none", "none."):
            items.append(cleaned)

    return items


def get_seniority_level(generator, job: dict) -> str:
    # Fast, reliable check first: if the job title itself contains an
    # explicit seniority word, trust that directly rather than relying
    # on the model -- we confirmed the model was returning "unspecified"
    # even when "Junior" appeared right in the title it was shown,
    # likely under-weighting the title versus the description body.
    title_lower = job.get("position", "").lower()
    if "junior" in title_lower or " jr" in title_lower or title_lower.startswith("jr"):
        return "junior"
    if "senior" in title_lower or " sr" in title_lower or title_lower.startswith("sr"):
        return "senior"

    prompt = build_prompt_for_field(
        job,
        "Including clues from the job title above, what seniority level is "
        "this role? Answer with exactly one word: junior, mid, senior, or unspecified."
    )
    result = generator(prompt, max_new_tokens=10, do_sample=False, repetition_penalty=1.1)
    text = result[0]["generated_text"][len(prompt):].strip().lower()
    first_word = text.split()[0].strip(".,") if text else "unspecified"

    if first_word not in ("junior", "mid", "senior", "unspecified"):
        return "unspecified"
    return first_word


MIN_DESCRIPTION_LENGTH = 200  # characters; postings shorter than this
                               # rarely contain real, extractable skill
                               # information -- we skip calling the
                               # model on them entirely.

# Phrases that indicate the model produced a refusal/disclaimer
# instead of real content, usually triggered by having almost nothing
# to work with. If a "skill" matches one of these, we drop it rather
# than store it as if it were a real extracted skill.
REFUSAL_PATTERNS = (
    "cannot provide",
    "i cannot",
    "privacy concerns",
    "privacy policies",
    "without having accessibility",
)


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def clean_job_posting(generator, job: dict) -> dict:
    """
    Cleans one job posting by asking the model 3 separate, focused
    questions, then assembling the answers into a dictionary ourselves
    in Python.

    Postings with very little real content (e.g. truncated API
    snippets) are skipped before calling the model at all, since we
    confirmed the model tends to produce generic refusal-sounding
    text when given almost nothing to work with -- no amount of
    prompt rewording fixed this, because the real problem is missing
    input data, not instruction wording.
    """
    if len(job.get("description", "")) < MIN_DESCRIPTION_LENGTH:
        job_with_cleaning = dict(job)
        job_with_cleaning["cleaned"] = {
            "role_summary": "",
            "skills": [],
            "seniority_level": "unspecified",
            "skipped_reason": "description too short to extract meaningful content",
        }
        return job_with_cleaning

    role_summary = get_role_summary(generator, job)
    skills = get_skills(generator, job)
    seniority_level = get_seniority_level(generator, job)

    # Defensive safety net: even on longer postings, drop any "skill"
    # that looks like a refusal/disclaimer rather than real content.
    skills = [s for s in skills if not _looks_like_refusal(s)]

    cleaned = {
        "role_summary": role_summary,
        "skills": skills,
        "seniority_level": seniority_level,
    }

    job_with_cleaning = dict(job)
    job_with_cleaning["cleaned"] = cleaned
    return job_with_cleaning


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "sample_remoteok_jobs.json"
    output_path = project_root / "data" / "processed" / "cleaned_jobs.json"

    print(f"Loading raw jobs from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} job postings.")

    print(f"Loading model '{MODEL_NAME}' (this may take a while the first time)...")
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",
        dtype=torch.float16,
    )

    cleaned_jobs = []
    success_count = 0
    failure_count = 0

    for i, job in enumerate(jobs, start=1):
        print(f"Cleaning job {i}/{len(jobs)}: {job['position']} @ {job['company']}", end=" ")

        result = clean_job_posting(generator, job)
        cleaned_jobs.append(result)

        # Check whether this specific job's "cleaned" field has an
        # "error" key (set inside clean_job_posting when JSON parsing
        # failed) or real extracted data.
        if "error" in result["cleaned"]:
            failure_count += 1
            print("-> FAILED (could not parse JSON)")
        else:
            success_count += 1
            print("-> OK")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_jobs, f, indent=2)

    print(f"\nSaved cleaned jobs to: {output_path}")
    print(f"Summary: {success_count} succeeded, {failure_count} failed, out of {len(jobs)} total.")


if __name__ == "__main__":
    main()