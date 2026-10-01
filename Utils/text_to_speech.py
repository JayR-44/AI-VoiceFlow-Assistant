from io import BytesIO

from gtts import gTTS


def text_to_speech(text):
    """Return MP3 bytes without creating a shared output file."""
    if not text.strip():
        raise ValueError("Enter text to read aloud.")
    output = BytesIO()
    gTTS(text=text, lang="en", timeout=(10, 30)).write_to_fp(output)
    audio = output.getvalue()
    if not audio:
        raise RuntimeError("Speech service returned empty audio.")
    return audio
