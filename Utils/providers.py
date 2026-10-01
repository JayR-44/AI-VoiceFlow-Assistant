import os

from openai import OpenAI
from groq import Groq


def error_summary(error):
    status = getattr(error, "status_code", None)
    if status == 401:
        return "API key was rejected; update it in .env and restart"
    if status == 403:
        return "access denied; check account and model permissions"
    if status == 429:
        return "quota or rate limit reached; check billing/limits or retry later"
    if status == 404:
        return "model unavailable; check the configured model name"
    if status == 400:
        return "request rejected; check the audio file or model configuration"
    if isinstance(error, ValueError):
        return "no usable text returned"
    return "service unavailable; check your connection and retry"


def with_fallback(operation, openai_call, groq_call, api_key=None):
    failures = []
    for provider, factory, key, call in (
        ("OpenAI", OpenAI, api_key or os.getenv("OPENAI_API_KEY", ""), openai_call),
        ("Groq", Groq, os.getenv("GROQ_API_KEY", ""), groq_call),
    ):
        if not key.strip():
            continue
        try:
            with factory(api_key=key.strip(), timeout=30.0, max_retries=0) as client:
                text = call(client)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response")
            return text.strip(), provider
        except Exception as error:
            # Raw provider exceptions may contain credentials; don't expose them.
            failures.append(f"{provider}: {error_summary(error)}")
    if not failures:
        raise RuntimeError("Add OPENAI_API_KEY or GROQ_API_KEY to .env and restart the app.")
    raise RuntimeError(f"{operation} failed. " + "; ".join(failures))
