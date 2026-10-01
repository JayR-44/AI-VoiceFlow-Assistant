import streamlit as st


def setup_page():
    st.set_page_config(page_title="VoiceFlow Assistant", layout="centered")
    st.title("VoiceFlow Assistant")
    st.write("Record a question. Get an answer in text and voice.")


def recording_changed(recording_key):
    # Each recording event is consumed once, including failures.
    st.session_state["recording_pending"] = st.session_state.get(recording_key) is not None


def question_input(disabled=False):
    generation = st.session_state.get("input_generation", 0)
    recording_key = f"recording_{generation}"
    st.caption("Click the microphone, allow access, ask your question, then stop. Your question is sent automatically when you stop.")
    audio = st.audio_input("Record your question", key=recording_key, on_change=recording_changed, args=(recording_key,), disabled=disabled)
    st.caption("Use localhost on this laptop, or HTTPS when hosted online.")
    pending = st.session_state.pop("recording_pending", False)
    retry = st.button("Ask again using this recording", disabled=disabled or audio is None)
    if audio is not None and (pending or retry) and not disabled:
        return {"audio": audio.getvalue(), "filename": "recording.wav"}
    return None
