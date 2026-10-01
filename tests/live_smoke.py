"""Explicit opt-in live check: sends a synthetic question and may use API credit."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from Utils.text_to_speech import text_to_speech
from Utils.speech_to_text import transcribe_audio
from Utils.question_answer import generate_answer


def main():
    load_dotenv()
    print("Checking speech generation with a synthetic question...", flush=True)
    try:
        audio = text_to_speech("What is two plus two?")
        print(f"gTTS: PASS ({len(audio)} MP3 bytes)", flush=True)
    except Exception:
        audio = None
        print("gTTS: FAIL (connection or speech service unavailable)", flush=True)
    keys = {key: os.environ.get(key, "") for key in ("OPENAI_API_KEY", "GROQ_API_KEY")}
    try:
        for key, value in keys.items():
            for other in keys:
                os.environ[other] = value if other == key else ""
            if not value.strip():
                print(f"{key}: not configured", flush=True)
                continue
            if audio:
                try:
                    transcript, provider = transcribe_audio(audio, "question.mp3")
                    print(f"{provider} transcription: PASS ({transcript})", flush=True)
                except (RuntimeError, ValueError) as error:
                    print(str(error), flush=True)
            try:
                answer, provider = generate_answer("What is two plus two?")
                print(f"{provider} answer: PASS ({answer})", flush=True)
                if audio:
                    spoken = text_to_speech(answer)
                    print(f"Answer speech: PASS ({len(spoken)} MP3 bytes)", flush=True)
            except (RuntimeError, ValueError) as error:
                print(str(error), flush=True)
            except Exception:
                print("Answer speech: FAIL (connection or speech service unavailable)", flush=True)
    finally:
        os.environ.update(keys)


if __name__ == "__main__":
    main()
