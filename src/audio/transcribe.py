# src/audio/transcribe.py
"""Utility for transcribing audio files using OpenAI's Whisper API.

The function :func:`transcribe` accepts a path to an audio file, calls the
``whisper-1`` model, and returns the plain‑text transcription.

The OpenAI API key is read from a ``.env`` file (via ``python‑dotenv``) so the
user does not need to hard‑code credentials.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load ``OPENAI_API_KEY`` from ``.env`` if present.
load_dotenv()

# Import OpenAI lazily – this avoids import errors if the package is missing
# until the function is actually used.
try:
    import openai
except ImportError as exc:
    raise ImportError(
        "The OpenAI Python client is required for transcription. Install it with "
        "`pip install openai`"
    ) from exc


def transcribe(audio_path: str) -> str:
    """Transcribe *audio_path* using the Whisper ``whisper-1`` model.

    Parameters
    ----------
    audio_path: str
        Absolute or relative path to an audio file supported by Whisper (e.g.,
        ``.wav``, ``.mp3``, ``.m4a``).

    Returns
    -------
    str
        The textual transcription returned by the OpenAI API.
    """
    audio_file = Path(audio_path).expanduser().resolve()
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # The OpenAI client reads the key from the environment variable
    # ``OPENAI_API_KEY``. ``load_dotenv`` ensures the ``.env`` file populates it.
    with audio_file.open("rb") as f:
        # Using the newer ``audio.transcriptions.create`` API (OpenAI >=1.0).
        # If the older ``Audio.transcribe`` method is available, the call still
        # works because the client provides backwards compatibility.
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    # ``response`` is a pydantic model with a ``text`` attribute.
    return response.text.strip()


if __name__ == "__main__":
    # Simple demo: transcribe a file provided as the first CLI argument.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.audio.transcribe <audio_file_path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        txt = transcribe(path)
        print("--- Transcription ---")
        print(txt)
    except Exception as e:
        print(f"Error: {e}")
