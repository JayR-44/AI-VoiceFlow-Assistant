# VoiceFlow Assistant

Record a question, stop recording, and receive an AI answer in text and spoken
English. Each question is independent (no chat history).

## Run on Windows

```powershell
cd D:\Github\Assessment-Collab
python -m venv .venv # Only if .venv does not already exist
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Stop with Ctrl+C. No environment activation is needed.
Create `.env` using `.env.example` as a guide, preserving your existing keys.
Configure at least one provider key, with available quota and model access.
Restart Streamlit after changing keys. Never commit real keys; any credentials
previously committed should be revoked and replaced. Ignoring `.env` does not
remove it from Git history or stop tracking an already tracked file.

## Use the microphone

Click the microphone, allow browser microphone access, ask a
question, and stop. Stopping automatically submits the recording once. If access
is denied, allow the microphone in browser site settings and Windows microphone
privacy settings. Use localhost locally or HTTPS for a hosted app. If autoplay is
blocked, click Play on the answer audio. Reset clears the inputs and answer.
Use short, clear recordings; audio is limited to 24 MB.

## Services and data flow

1. Browser records your question as WAV audio.
2. OpenAI `whisper-1` transcribes it; failures fall back to Groq `whisper-large-v3`.
3. OpenAI Responses with `gpt-4.1-mini` answers the transcript; failures fall back
   to Groq `openai/gpt-oss-120b`. Optional model overrides are in `.env.example`.
4. gTTS sends the answer to Google's speech service and returns English MP3 bytes.
5. Streamlit retains the current question, answer, and audio in session memory.

Internet access is required. API providers may charge for requests. The UI shows
which provider succeeded and actionable errors for rejected keys, quota, and
model access. A voice-service failure preserves the written answer and offers an
audio-only retry. This app has no live web search. Browser microphone capture and
speaker playback require a manual device test.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests mock external services; they do not spend API credits.

To explicitly check real credentials and services with a synthetic question
(requires internet and may use API credit):

```powershell
.\.venv\Scripts\python.exe -B tests/live_smoke.py
```

Verified on September 30, 2026: the automated tests passed. After correcting the
Groq model setting, live Groq transcription, answer generation, and gTTS answer
audio all passed. OpenAI was not configured in this check. No API keys were
changed by these checks.

Groq retired `llama-3.3-70b-versatile` for free/developer accounts on August 16,
2026. This project now defaults to `openai/gpt-oss-120b`, hosted by Groq and using
your Groq key. See [Groq's deprecation notice](https://console.groq.com/docs/deprecations).
Restart the app after updating `GROQ_CHAT_MODEL` in `.env`.

References: [Streamlit recorder](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input),
[OpenAI text generation](https://developers.openai.com/api/docs/guides/text),
[Groq answer model](https://console.groq.com/docs/model/openai/gpt-oss-120b).
