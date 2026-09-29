"""Generate a text answer for a single incoming message, using the same
vector store as the voice agent (rag/store.py). Kept separate from
agent.py since this is a plain request/response text flow, not a
LiveKit voice session.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from dotenv import load_dotenv
from store import VectorStore
load_dotenv(".env.example")
GEMINI_MODEL = os.getenv("WHATSAPP_LLM_MODEL", "gemini-2.5-flash")
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful WhatsApp assistant. Answer using the provided "
    "knowledge base context whenever it's relevant -- don't guess. If "
    "the context doesn't cover the question, say so honestly and "
    "answer from general knowledge if you can. Keep answers short and "
    "conversational, suited to a chat message rather than an essay."
)

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


async def generate_reply(store: VectorStore, question: str) -> str:
    results = await store.search(question, top_k=TOP_K)
    if results:
        context = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk, _score in results)
    else:
        context = "(no relevant knowledge base entries found)"

    prompt = f"Knowledge base context:\n{context}\n\nUser question: {question}"

    response = await _client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return (response.text or "").strip() or "Sorry, I couldn't come up with an answer to that."
