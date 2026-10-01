import os
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from Utils import providers, speech_to_text, question_answer, text_to_speech, workflow, Ui


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai", "GROQ_API_KEY": "test-groq"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_openai_success_does_not_call_groq(self):
        with patch.object(providers, "OpenAI") as primary, patch.object(providers, "Groq") as fallback:
            primary.return_value.__enter__.return_value.responses.create.return_value.output_text = " Four. "
            self.assertEqual(question_answer.generate_answer("What is 2+2?"), ("Four.", "OpenAI"))
            fallback.assert_not_called()

    def test_answer_falls_back_to_groq(self):
        with patch.object(providers, "OpenAI", side_effect=RuntimeError("secret-key")), patch.object(providers, "Groq") as fallback:
            client = fallback.return_value.__enter__.return_value
            client.chat.completions.create.return_value.choices = [SimpleNamespace(message=SimpleNamespace(content="Four."))]
            self.assertEqual(question_answer.generate_answer("2+2?"), ("Four.", "Groq"))

    def test_missing_primary_uses_groq(self):
        os.environ["OPENAI_API_KEY"] = ""
        with patch.object(providers, "OpenAI") as primary, patch.object(providers, "Groq"):
            self.assertEqual(providers.with_fallback("Test", lambda c: "a", lambda c: "b"), ("b", "Groq"))
            primary.assert_not_called()

    def test_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Add OPENAI_API_KEY"):
                question_answer.generate_answer("Hello")

    def test_errors_do_not_leak_raw_credentials(self):
        with patch.object(providers, "OpenAI", side_effect=RuntimeError("secret-key")), patch.object(providers, "Groq", side_effect=RuntimeError("secret-key")):
            with self.assertRaises(RuntimeError) as error:
                question_answer.generate_answer("Hello")
            self.assertNotIn("secret-key", str(error.exception))
            self.assertIn("OpenAI", str(error.exception))
            self.assertIn("Groq", str(error.exception))

    def test_actionable_status_messages(self):
        for status, phrase in [(401, "key was rejected"), (403, "permissions"), (429, "quota"), (404, "model unavailable")]:
            error = RuntimeError("private error")
            error.status_code = status
            self.assertIn(phrase, providers.error_summary(error))

    def test_transcription_preserves_mp3_and_wav_types(self):
        for suffix, mime in [("mp3", "audio/mpeg"), ("wav", "audio/wav")]:
            with patch.object(providers, "OpenAI") as primary:
                client = primary.return_value.__enter__.return_value
                client.audio.transcriptions.create.return_value.text = "A question?"
                self.assertEqual(speech_to_text.transcribe_audio(b"audio", "input." + suffix), ("A question?", "OpenAI"))
                self.assertEqual(client.audio.transcriptions.create.call_args.kwargs["file"], ("question." + suffix, b"audio", mime))

    def test_empty_transcript_falls_back(self):
        with patch.object(providers, "OpenAI") as primary, patch.object(providers, "Groq") as fallback:
            primary.return_value.__enter__.return_value.audio.transcriptions.create.return_value.text = " "
            fallback.return_value.__enter__.return_value.audio.transcriptions.create.return_value.text = "Question?"
            self.assertEqual(speech_to_text.transcribe_audio(b"audio", "a.wav"), ("Question?", "Groq"))

    def test_invalid_audio_rejected_before_network(self):
        for audio, filename in [(b"", "a.wav"), (b"a", "a.exe"), (b"a" * (speech_to_text.MAX_AUDIO_BYTES + 1), "a.wav")]:
            with self.assertRaises(ValueError):
                speech_to_text.transcribe_audio(audio, filename)


class WorkflowTests(unittest.TestCase):
    def test_recording_to_spoken_answer(self):
        with patch.object(workflow, "transcribe_audio", return_value=("2+2?", "Groq")) as transcribe, patch.object(workflow, "generate_answer", return_value=("Four.", "OpenAI")) as answer, patch.object(workflow, "text_to_speech", return_value=b"mp3") as speak:
            result = workflow.answer_question(audio=b"wav", filename="recording.wav")
            transcribe.assert_called_once_with(b"wav", "recording.wav")
            answer.assert_called_once_with("2+2?")
            speak.assert_called_once_with("Four.")
            self.assertEqual(result["audio"], b"mp3")
            self.assertEqual(len(result["providers"]), 2)

    @patch.object(workflow, "transcribe_audio", return_value=("2+2?", "Groq"))
    def test_speech_failure_preserves_answer(self, transcribe):
        with patch.object(workflow, "generate_answer", return_value=("Four.", "Groq")), patch.object(workflow, "text_to_speech", side_effect=RuntimeError()):
            result = workflow.answer_question(audio=b"wav")
            self.assertEqual(result["answer"], "Four.")
            self.assertIn("speech_error", result)

    @patch.object(workflow, "transcribe_audio", return_value=("2+2?", "Groq"))
    def test_answer_failure_preserves_question(self, transcribe):
        with patch.object(workflow, "generate_answer", side_effect=RuntimeError("Quota exhausted")), patch.object(workflow, "text_to_speech") as speak:
            result = workflow.answer_question(audio=b"wav")
            self.assertEqual(result["question"], "2+2?")
            self.assertIn("error", result)
            speak.assert_not_called()

    @patch.object(workflow, "transcribe_audio", return_value=(" ", "Groq"))
    def test_empty_question_stops_pipeline(self, transcribe):
        with patch.object(workflow, "generate_answer") as answer:
            self.assertIn("error", workflow.answer_question(audio=b"wav"))
            answer.assert_not_called()

    def test_speech_is_returned_in_memory(self):
        with patch.object(text_to_speech, "gTTS") as tts:
            tts.return_value.write_to_fp.side_effect = lambda fp: fp.write(b"MP3")
            self.assertEqual(text_to_speech.text_to_speech("Hello"), b"MP3")
            tts.return_value.save.assert_not_called()


class UiTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "GROQ_API_KEY": ""}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dotenv = patch("dotenv.load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)

    def submit_recording(self, app, question="2+2?"):
        with patch.object(Ui, "question_input", return_value={"audio": b"wav", "filename": "recording.wav"}), patch.object(workflow, "transcribe_audio", return_value=(question, "Groq")):
            app.run()
        self.assertFalse(app.exception)

    def test_initial_render_has_only_recording_flow(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("audio_input")), 1)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertEqual(len(app.radio), 0)
        self.assertEqual(len(app.text_area), 0)
        self.assertEqual(len(app.expander), 0)

    def test_recorded_answer_persists_without_duplicate_calls_and_resets(self):
        with patch.object(workflow, "generate_answer", return_value=("Four.", "OpenAI")) as answer, patch.object(workflow, "text_to_speech", return_value=b"MP3"):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.submit_recording(app)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["result"]["answer"], "Four.")
            app.run()
            answer.assert_called_once()
            next(b for b in app.button if b.label == "Reset").click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("result", app.session_state)

    def test_recording_event_is_consumed_once(self):
        with patch.object(Ui, "st") as st:
            st.session_state = {"recording": BytesIO(b"wav")}
            st.audio_input.return_value = st.session_state["recording"]
            st.button.return_value = False
            Ui.recording_changed("recording")
            self.assertEqual(Ui.question_input(), {"audio": b"wav", "filename": "recording.wav"})
            self.assertIsNone(Ui.question_input())
            st.session_state["recording"] = None
            st.audio_input.return_value = None
            Ui.recording_changed("recording")
            self.assertIsNone(Ui.question_input())

    def test_reset_recreates_empty_recorder_on_every_click(self):
        with patch.object(workflow, "generate_answer") as answer:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            previous_id = app.get("audio_input")[0].proto.id
            for generation in (1, 2):
                app.session_state["result"] = {"question": "Old question", "answer": "Old answer", "audio": b"MP3"}
                app.session_state["recording_pending"] = True
                next(b for b in app.button if b.label == "Reset").click().run()
                self.assertFalse(app.exception)
                recorder = app.get("audio_input")[0]
                self.assertNotEqual(recorder.proto.id, previous_id)
                previous_id = recorder.proto.id
                self.assertIsNone(app.session_state[f"recording_{generation}"])
                self.assertNotIn("result", app.session_state)
                self.assertNotIn("recording_pending", app.session_state)
                self.assertTrue(next(b for b in app.button if b.label == "Ask again using this recording").disabled)
            answer.assert_not_called()

    def test_failed_new_question_clears_previous_audio(self):
        with patch.object(workflow, "generate_answer", side_effect=[("Four.", "OpenAI"), RuntimeError("Quota exhausted")]), patch.object(workflow, "text_to_speech", return_value=b"MP3"):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.submit_recording(app)
            self.assertIn("audio", app.session_state["result"])
            self.submit_recording(app, "3+3?")
            self.assertFalse(app.exception)
            self.assertNotIn("audio", app.session_state["result"])
            self.assertNotIn("answer", app.session_state["result"])
            self.assertEqual(app.session_state["result"]["question"], "3+3?")

    def test_speech_retry_does_not_regenerate_answer(self):
        with patch.object(workflow, "generate_answer", return_value=("Four.", "Groq")) as answer, patch.object(workflow, "text_to_speech", side_effect=RuntimeError()):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.submit_recording(app)
            with patch.object(text_to_speech, "text_to_speech", return_value=b"MP3"):
                next(b for b in app.button if b.label == "Retry answer audio").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["result"]["audio"], b"MP3")
            answer.assert_called_once()

    def test_missing_keys_disable_recording(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "GROQ_API_KEY": ""}, clear=False):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.get("audio_input")[0].proto.disabled)
            self.assertTrue(app.warning)


if __name__ == "__main__":
    unittest.main()
