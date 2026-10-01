from Utils.speech_to_text import transcribe_audio
from Utils.question_answer import generate_answer
from Utils.text_to_speech import text_to_speech


def answer_question(audio, filename="recording.wav"):
    """Preserve completed stages if a later service fails."""
    result = {"providers": []}
    try:
        question, provider = transcribe_audio(audio, filename)
        result["providers"].append(f"Transcription: {provider}")
        question = question.strip()
        if not question:
            raise ValueError("No question was detected. Please record your question again and speak clearly.")
        result["question"] = question
        answer, provider = generate_answer(question)
        result["answer"] = answer
        result["providers"].append(f"Answer: {provider}")
    except (ValueError, RuntimeError) as error:
        result["error"] = str(error)
        return result
    try:
        result["audio"] = text_to_speech(answer)
    except Exception:
        result["speech_error"] = "Your answer is ready, but speech generation failed. Check your connection and retry the audio."
    return result
