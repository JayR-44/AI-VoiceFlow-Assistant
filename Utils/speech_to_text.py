from pathlib import Path

from Utils.providers import with_fallback

MAX_AUDIO_BYTES = 24 * 1024 * 1024


def transcribe_audio(audio, filename):
    suffix = Path(filename).suffix.lower()
    if suffix not in (".mp3", ".wav"):
        raise ValueError("Upload an MP3 or WAV file.")
    if not audio:
        raise ValueError("The audio is empty. Record again or choose another file.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("Audio must be 24 MB or smaller. Please use a shorter recording.")
    upload = ("question" + suffix, audio, "audio/wav" if suffix == ".wav" else "audio/mpeg")
    return with_fallback(
        "Transcription",
        lambda client: client.audio.transcriptions.create(model="whisper-1", file=upload).text,
        lambda client: client.audio.transcriptions.create(model="whisper-large-v3", file=upload).text,
    )
