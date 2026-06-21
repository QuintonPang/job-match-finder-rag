"""
fetch_remoteok.py

Purpose:
    Pull live job postings from the RemoteOK public API and save them
    to disk as raw, untouched JSON. This is the "ingestion" stage of
    our RAG pipeline -- we are NOT cleaning or processing anything yet.
    We just want the real-world data sitting in our system.

Why save raw data separately?
    If we clean/transform the data and later find a bug in our cleaning
    logic, we don't want to have to re-fetch from the internet again.
    We keep the original copy untouched in data/raw/, and write cleaned
    versions to data/processed/ in a later lesson.
"""

import requests
import json
from pathlib import Path
from datetime import datetime, timezone

# RemoteOK's public API endpoint. No API key required.
REMOTEOK_API_URL = "https://remoteok.com/api"

# RemoteOK asks that you set a real User-Agent header, otherwise
# some servers block requests that look like bots with no identity.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational RAG project; portfolio demo)"
}


def fetch_jobs() -> list[dict]:
    """
    Calls the RemoteOK API and returns the raw list of job postings.

    Returns:
        A list of dictionaries, each representing one job posting.
    """
    response = requests.get(REMOTEOK_API_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()  # raises an error if the request failed

    jobs = response.json()

    # RemoteOK's API always returns a "legal notice" object as the
    # very first item in the list. It's not a real job, so we drop it.
    if jobs and "legal" in jobs[0]:
        jobs = jobs[1:]

    return jobs


def save_raw_jobs(jobs: list[dict], output_dir: Path) -> Path:
    """
    Saves the raw job list to disk as a timestamped JSON file.

    Args:
        jobs: list of job posting dictionaries
        output_dir: folder to save into

    Returns:
        The path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"remoteok_raw_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    return output_path


def main():
    print("Fetching live job postings from RemoteOK...")
    jobs = fetch_jobs()
    print(f"Fetched {len(jobs)} job postings.")

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    saved_path = save_raw_jobs(jobs, raw_dir)

    print(f"Saved raw data to: {saved_path}")


if __name__ == "__main__":
    main()