# fetch_pubmed.py
"""PubMed ingestion script.

This script fetches abstracts from PubMed using Biopython's Entrez module and writes them to a JSONL
file under the project's `data/corpus/` directory.

Two collections are retrieved:
  1. All papers authored by the specified `AUTHOR_NAME`.
  2. Up to `MAX_COUNT` most recent abstracts matching the `TOPIC_QUERY`.

The variables at the top of the file can be edited before running the script.
"""

import os
import time
import json
from typing import List, Dict

from Bio import Entrez

# ==============================
# Editable configuration variables
# ==============================
AUTHOR_NAME = "Minagar A"  # Author to fetch all publications for
TOPIC_QUERY = "multiple sclerosis neuroimmunology"  # Search query for recent abstracts
MAX_COUNT = 500  # Maximum number of abstracts to fetch for the query
# ==============================

# NCBI requires an email address for all requests.
Entrez.email = "aminagar@gmail.com"
# Optional: set your NCBI API key here to increase rate limits (if you have one).
# Entrez.api_key = "YOUR_NCBI_API_KEY"

# Rate‑limit handling – NCBI recommends no more than 3 requests per second.
REQUEST_DELAY = 0.34  # seconds (approx. 3 requests per second)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../data/corpus/pubmed_abstracts.jsonl")
OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

def _esearch(term: str, retmax: int = 10000) -> List[str]:
    """Search PubMed for a term and return a list of PMIDs.

    Args:
        term: The query string passed to Entrez.
        retmax: Maximum number of IDs to retrieve (default 10 000).
    Returns:
        List of PMID strings.
    """
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    time.sleep(REQUEST_DELAY)
    return record.get("IdList", [])

def _efetch(pmids: List[str]) -> List[Dict]:
    """Fetch details for a list of PMIDs.

    The function batches the PMIDs to avoid oversized requests.
    """
    results = []
    batch_size = 200  # Entrez can handle a few hundred IDs per request.
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        ids = ",".join(batch)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        time.sleep(REQUEST_DELAY)
        for article in records.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            # PMID
            pmid = str(medline.get("PMID", ""))
            # Title
            title = article_data.get("ArticleTitle", "")
            # Authors
            author_list = []
            for a in article_data.get("AuthorList", []):
                if "LastName" in a and "ForeName" in a:
                    author_list.append(f"{a['ForeName']} {a['LastName']}")
                elif "CollectiveName" in a:
                    author_list.append(a["CollectiveName"])
            # Journal & year
            journal = article_data.get("Journal", {}).get("Title", "")
            year = ""
            if "JournalIssue" in article_data.get("Journal", {}):
                year = article_data["Journal"]["JournalIssue"].get("PubDate", {}).get("Year", "")
            # Abstract text
            abstract_text = ""
            abstract = article_data.get("Abstract", {})
            if isinstance(abstract, dict):
                # Concatenate all AbstractText elements.
                abstract_text = " ".join([t for t in abstract.get("AbstractText", [])])
            # Skip records without an abstract.
            if not abstract_text.strip():
                continue
            results.append({
                "pmid": pmid,
                "title": title,
                "authors": author_list,
                "journal": journal,
                "year": year,
                "abstract": abstract_text,
            })
    return results

def fetch_by_author(author_name: str) -> List[Dict]:
    """Retrieve all PubMed records for a given author.

    The search term uses the `[Author]` field tag.
    """
    term = f"{author_name}[Author]"
    pmids = _esearch(term, retmax=10000)
    return _efetch(pmids)

def fetch_by_query(query: str, max_count: int) -> List[Dict]:
    """Retrieve the most recent abstracts matching a query.

    The search is sorted by most recent (`sort=date`).
    """
    # Use a broader search without field restriction to increase hits.
    term = query
    # Retrieve up to max_count recent abstracts sorted by publication date.
    handle = Entrez.esearch(
        db="pubmed",
        term=term,
        retmax=max_count,
        sort="pub date",
        usehistory="y",
    )
    record = Entrez.read(handle)
    handle.close()
    time.sleep(REQUEST_DELAY)
    pmids = record.get("IdList", [])
    return _efetch(pmids)

def main() -> None:
    # Collect records from both sources.
    print("Fetching records authored by", AUTHOR_NAME)
    author_records = fetch_by_author(AUTHOR_NAME)
    print(f"Retrieved {len(author_records)} records for author.")

    print("Fetching recent abstracts for query:", TOPIC_QUERY)
    query_records = fetch_by_query(TOPIC_QUERY, MAX_COUNT)
    print(f"Retrieved {len(query_records)} records for query.")

    # Combine and deduplicate records by PMID to avoid duplicates from author and query.
    combined = {rec["pmid"]: rec for rec in author_records + query_records}
    all_records = list(combined.values())
    
    # Ensure output directory exists.
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Writing to", OUTPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            json_line = json.dumps(rec, ensure_ascii=False)
            f.write(json_line + "\n")
    print("Done.")

if __name__ == "__main__":
    main()
