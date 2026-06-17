# src/app.py
"""Gradio web UI for MedVoiceRAG.

Features
--------
1. Users can **record audio** (microphone) **or type** a question.
2. Recorded audio is automatically transcribed via OpenAI Whisper and fills the
   editable question textbox so the user can refine it.
3. On **Submit**, the RAG pipeline retrieves relevant PubMed chunks, the LLM
   generates a concise answer, and the answer is spoken back to the user using
   gTTS.
4. Citations are rendered as a markdown list; each PMID is a clickable link to
   PubMed with the title, journal and year displayed.

The script ties together:
- :func:`answer_question` from ``src.rag.pipeline``
- :func:`transcribe` from ``src.audio.transcribe``
- :func:`speak` from ``src.audio.speak``

Run with ``poetry run python src/app.py``.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple

import gradio as gr

# Local imports – the project root is two levels up from this file.
from src.rag.pipeline import answer_question
from src.audio.transcribe import transcribe as audio_transcribe
from src.audio.speak import speak as synthesize_speech

# ---------------------------------------------------------------------------
# Helper to format citations as Markdown with PubMed links
# ---------------------------------------------------------------------------

def format_citations(sources: List[Dict]) -> str:
    """Return a markdown string with each source as a clickable PubMed link.

    Each entry displays ``[PMID: xxxx] Title (Journal, Year)`` where the PMID
    links to ``https://pubmed.ncbi.nlm.nih.gov/<PMID>/``.
    """
    if not sources:
        return "*No sources returned.*"
    lines = []
    for src in sources:
        pmid = src.get("pmid", "")
        title = src.get("title", "").replace("\n", " ")
        journal = src.get("journal", "")
        year = src.get("year", "")
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#"
        lines.append(f"- [{pmid}]({link}) {title} ({journal}, {year})")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file and return the text for the question box."""
    if not audio_path:
        return ""
    try:
        return audio_transcribe(audio_path)
    except Exception as e:
        return f"[Transcription error: {e}]"


def answer_and_speak(question: str) -> Tuple[str, str, str]:
    """Run the RAG pipeline, synthesize speech and return answer, citations MD,
    and path to the spoken audio file.
    """
    if not question:
        return "", "", ""
    # 1️⃣ Retrieve answer and source metadata.
    answer, sources = answer_question(question)
    # 2️⃣ Format citations for display.
    citations_md = format_citations(sources)
    # 3️⃣ Generate spoken version of the answer.
    try:
        audio_path = synthesize_speech(answer)
    except Exception as e:
        audio_path = ""
        # Append a note to the answer if synthesis fails.
        answer += f"\n\n[Speech synthesis error: {e}]"
    return answer, citations_md, audio_path

# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# MedVoiceRAG — Voice Q&A over Neuroimmunology Literature")

    with gr.Row():
        # Left column – input area
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Speak your question",
                type="filepath",
            )
            question_input = gr.Textbox(
                label="Question (you can edit after transcription)",
                lines=2,
                placeholder="Type your question here...",
                elem_id="question_input",
            )
            submit_btn = gr.Button("Submit", variant="primary")

        # Right column – output area
        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Answer",
                lines=6,
                interactive=False,
                elem_id="answer_output",
            )
            citations_output = gr.Markdown(
                label="Citations",
                elem_id="citations_output",
            )
            audio_output = gr.Audio(
                label="Spoken answer",
                type="filepath",
                elem_id="audio_output",
            )

    # -------------------------------------------------------------------
    # Interactions
    # -------------------------------------------------------------------
    # When the user records audio, automatically transcribe and fill the box.
    audio_input.change(fn=transcribe_audio, inputs=audio_input, outputs=question_input)

    # When the user clicks Submit, run the RAG pipeline and synthesize speech.
    submit_btn.click(
        fn=answer_and_speak,
        inputs=question_input,
        outputs=[answer_output, citations_output, audio_output],
    )

if __name__ == "__main__":
    # Launch the Gradio server on the default port (7860).
    demo.launch()
