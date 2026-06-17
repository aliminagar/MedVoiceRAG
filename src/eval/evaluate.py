# src/eval/evaluate.py
"""RAGAS evaluation script for MedVoiceRAG (ragas 0.4.x).

Uses 10 curated MS / neuro-immunology questions (no ground truth).
Computes Faithfulness and AnswerRelevancy — metrics that do NOT require
reference answers.

Run with:
    poetry run python src/eval/evaluate.py
"""

import json
import os
import sys
import types
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()  # load OPENAI_API_KEY from .env

# ---------------------------------------------------------------------------
# CRITICAL: stub out langchain_community.chat_models.vertexai BEFORE ragas
# is imported anywhere. ragas.llms.base does a top-level import of ChatVertexAI
# from that module; without this stub the whole ragas package fails to load.
# ---------------------------------------------------------------------------
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {
    "__init__": lambda self, *a, **kw: None
})
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

# ---------------------------------------------------------------------------
# ragas imports — use the OLD (pre-collections) singleton metrics.
# ragas.evaluate() checks isinstance(m, Metric) against the OLD Metric class;
# the new ragas.metrics.collections classes inherit from BaseMetric and do NOT
# pass that check, so we must use the old API here.
# ---------------------------------------------------------------------------
import openai as _openai                              # noqa: E402
from ragas import evaluate as ragas_evaluate          # noqa: E402
from ragas.metrics import faithfulness, answer_relevancy  # noqa: E402
from ragas.llms import llm_factory                    # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from langchain_openai import OpenAIEmbeddings         # noqa: E402
from datasets import Dataset                          # noqa: E402

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.rag.pipeline import answer_question, _retriever  # noqa: E402

# ---------------------------------------------------------------------------
# Test set — 10 MS / neuro-immunology questions (no ground truth needed)
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "What is the role of natalizumab in multiple sclerosis?",
    "How does B-cell depletion therapy (e.g., ocrelizumab) work in MS?",
    "What are the primary mechanisms of neuroinflammation in multiple sclerosis?",
    "Which cytokines are most implicated in the pathogenesis of MS?",
    "How does myelin repair occur after demyelination in MS?",
    "What imaging biomarkers are used to monitor disease activity in MS?",
    "Can gut microbiota influence multiple sclerosis progression?",
    "What are the long-term safety concerns of alemtuzumab therapy?",
    "How do Th17 cells contribute to neuroautoimmunity in MS?",
    "What are the latest advances in remyelination therapies for MS?",
]


def _run_query(question: str) -> dict:
    """Run one question through the RAG pipeline; return answer + contexts."""
    docs = _retriever.invoke(question)
    answer, _ = answer_question(question)
    contexts = [doc.page_content for doc in docs]
    return {"question": question, "answer": answer, "contexts": contexts}


def evaluate() -> None:
    print("Running RAG pipeline on all questions...")
    rows = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {q[:70]}")
        rows.append(_run_query(q))

    # Build HuggingFace Dataset in the OLD ragas format
    # (old ragas.evaluate expects: question / answer / contexts columns)
    hf_dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer":   [r["answer"]   for r in rows],
        "contexts": [r["contexts"] for r in rows],
    })

    # Initialise OpenAI LLM + embeddings
    print("Initialising OpenAI LLM and embeddings for RAGAS...")
    _client    = _openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm        = llm_factory(model="gpt-4o-mini", client=_client)
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    print("Running RAGAS evaluation (this may take a few minutes)...")
    result = ragas_evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    result_df = result.to_pandas()
    print("RAGAS scoring complete. Columns:", list(result_df.columns))

    # Pull scores (column names may be 'faithfulness' and 'answer_relevancy')
    faith_col = "faithfulness"     if "faithfulness"     in result_df.columns else None
    rel_col   = "answer_relevancy" if "answer_relevancy" in result_df.columns else None

    for i, r in enumerate(rows):
        r["faithfulness"]     = float(result_df.loc[i, faith_col]) if faith_col else None
        r["answer_relevancy"] = float(result_df.loc[i, rel_col])   if rel_col   else None

    faith_scores = [r["faithfulness"]     for r in rows if r["faithfulness"]     is not None]
    rel_scores   = [r["answer_relevancy"] for r in rows if r["answer_relevancy"] is not None]
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else float("nan")
    avg_rel   = sum(rel_scores)   / len(rel_scores)   if rel_scores   else float("nan")

    # -----------------------------------------------------------------------
    # Write output files into src/eval/
    # -----------------------------------------------------------------------
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    md_path   = out_dir / "results.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "averages": {
                "faithfulness":     avg_faith,
                "answer_relevancy": avg_rel,
            },
            "per_question": rows,
        }, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# RAGAS Evaluation Results",
        "",
        "| # | Question | Faithfulness | Answer Relevancy |",
        "|---|----------|:------------:|:----------------:|",
    ]
    for idx, r in enumerate(rows, 1):
        q     = r["question"].replace("|", "\\|")
        f_val = f"{r['faithfulness']:.3f}"     if r["faithfulness"]     is not None else "N/A"
        r_val = f"{r['answer_relevancy']:.3f}" if r["answer_relevancy"] is not None else "N/A"
        md_lines.append(f"| {idx} | {q} | {f_val} | {r_val} |")
    md_lines += [
        "",
        f"**Average Faithfulness:** {avg_faith:.3f}",
        f"**Average Answer Relevancy:** {avg_rel:.3f}",
    ]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n✓ JSON  -> {json_path}")
    print(f"✓ MD    -> {md_path}")
    print(f"\nAverage Faithfulness:     {avg_faith:.3f}")
    print(f"Average Answer Relevancy: {avg_rel:.3f}")


if __name__ == "__main__":
    evaluate()
