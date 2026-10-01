import os

from Utils.providers import with_fallback

INSTRUCTIONS = (
    "You are a helpful voice assistant. Answer the user's question in English, "
    "using concise, natural sentences suitable for speaking aloud."
    "Only answer what is asked, Don't add more relative information but even if you want to then Usually use "
    "two to four sentences. Avoid markdown formatting unless necessary. "
    "Be honest about uncertainty. You do not have live web access."
)


def generate_answer(question):
    question = question.strip()
    if not question:
        raise ValueError("Please ask a question first.")
    if len(question) > 12000:
        raise ValueError("Please shorten your question to 12,000 characters or fewer.")
    return with_fallback(
        "Answer generation",
        lambda client: client.responses.create(
            model=os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini",
            instructions=INSTRUCTIONS, input=question, max_output_tokens=400,
            store=False,
        ).output_text,
        lambda client: client.chat.completions.create(
            model=os.getenv("GROQ_CHAT_MODEL") or "openai/gpt-oss-120b",
            messages=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": question}],
            max_completion_tokens=400,
        ).choices[0].message.content,
    )
