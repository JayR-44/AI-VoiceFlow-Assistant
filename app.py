import os

import streamlit as st
from dotenv import load_dotenv

from Utils.Ui import setup_page, question_input
from Utils.workflow import answer_question
from Utils.text_to_speech import text_to_speech

load_dotenv()
setup_page()


def reset():
    # A new widget identity clears browser-held recordings.
    generation = st.session_state.get("input_generation", 0) + 1
    for key in list(st.session_state):
        del st.session_state[key]
    st.session_state["input_generation"] = generation


with st.sidebar:
    st.header("VoiceFlow Assistant")
    st.caption("Ask a question and hear the answer in English.")
    configured = any(os.getenv(key, "").strip() for key in ("OPENAI_API_KEY", "GROQ_API_KEY"))
    if not configured:
        st.warning("Add OPENAI_API_KEY or GROQ_API_KEY to .env, then restart the app.")
    st.button("Reset", on_click=reset)
    st.caption("Audio/questions are sent to OpenAI, or Groq on fallback. Answer text is sent to Google's speech service.")

request = question_input(disabled=not configured)
if request:
    # Clear the previous result before any request to avoid stale answer audio.
    st.session_state["result"] = {}
    with st.spinner("Listening, thinking, and preparing your answer..."):
        st.session_state["result"] = answer_question(**request)
    st.session_state["play_answer"] = True

result = st.session_state.get("result", {})
if result.get("question"):
    st.subheader("Your question")
    st.write(result["question"])
if result.get("answer"):
    st.subheader("Answer")
    st.write(result["answer"])
if result.get("providers"):
    st.caption(" | ".join(result["providers"]))
if result.get("error"):
    st.error(result["error"])
if result.get("speech_error"):
    st.warning(result["speech_error"])
    if st.button("Retry answer audio"):
        try:
            result["audio"] = text_to_speech(result["answer"])
            result.pop("speech_error", None)
            st.session_state["play_answer"] = True
            st.rerun()
        except Exception:
            st.error("Speech service is still unavailable. Check your internet connection and retry.")
if result.get("audio"):
    st.audio(result["audio"], format="audio/mpeg", autoplay=st.session_state.pop("play_answer", False))
    st.caption("If your browser blocks automatic playback, press Play.")
    st.download_button("Download answer audio", result["audio"], "answer.mp3", "audio/mpeg")
