# src/audio/speak.py
"""Utility for converting text to speech using gTTS.

The function :func:`speak` takes a string, synthesises English speech with the
Google Text‑to‑Speech (gTTS) library, writes the audio to a temporary ``.mp3``
file inside the project, and returns the absolute path of that file.

The resulting file can be fed directly to a Gradio ``Audio`` component.
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# ``gTTS`` does not need an API key, but we load the `.env` anyway for
# consistency and potential future configuration.
load_dotenv()

try:
    from gtts import gTTS
except ImportError as exc:
    raise ImportError(
        "gTTS is required for speech synthesis. Install it with `pip install gtts`."
    ) from exc


def speak(text: str, language: str = "en") -> str:
    """Convert *text* to an MP3 file using gTTS.

    Parameters
    ----------
    text: str
        The text to synthesize.
    language: str, optional
        Language code for gTTS (default ``"en"`` for English).

    Returns
    -------
    str
        Absolute path to the generated ``.mp3`` file.
    """
    if not text:
        raise ValueError("`text` must be a non‑empty string.")

    # Ensure the output folder exists – we keep temporary audio inside the
    # project under ``tmp_audio`` (which is ignored via .gitignore).
    output_dir = Path(__file__).parents[2] / "tmp_audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique filename to avoid collisions.
    filename = f"speech_{uuid.uuid4().hex}.mp3"
    output_path = output_dir / filename

    # Generate speech.
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(str(output_path))

    return str(output_path.resolve())


if __name__ == "__main__":
    # Demo: read a line of text from the command line and synthesize it.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.audio.speak <text>")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    try:
        mp3_path = speak(input_text)
        print(f"Audio saved to: {mp3_path}")
    except Exception as e:
        print(f"Error: {e}")
