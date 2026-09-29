"""LiveKit voice agent with RAG over a local knowledge base.

Setup:
    pip install -r requirements.txt
    cp .env.example .env.local   # fill in your keys
    python -m rag.ingest --source ./knowledge_base --out ./index

Run:
    python agent.py console   # local mic/speaker, no LiveKit server needed
    python agent.py dev       # connect to a real LiveKit room

This is a standard STT-LLM-TTS pipeline, not the Gemini Live
speech-to-speech model: LLM is Gemini (google.LLM, text-only, plain
GOOGLE_API_KEY from Google AI Studio -- free tier, no GCP project or
billing account). STT/TTS are Groq (Whisper for STT, Orpheus for TTS),
also plain API key, no GCP involved anywhere. Google Cloud Speech/TTS
were deliberately left out here since those always require a real GCP
service account regardless of which model you point them at.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import google, groq, silero

from store import VectorStore

load_dotenv(".env.local")

logger = logging.getLogger("rag-agent")
logger.setLevel(logging.INFO)

INDEX_PATH = "./index"
TOP_K = 4

# Swap these to change models.
STT_MODEL = "whisper-large-v3-turbo"
STT_LANGUAGE = "en"  # e.g. "ar" for Arabic
GEMINI_MODEL = "gemini-2.5-flash"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "autumn"


def prewarm(proc: JobProcess) -> None:
    # Loaded once per worker process, reused across every session --
    # keeps per-call connection latency down.
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["store"] = VectorStore.load(INDEX_PATH)


class RagAssistant(Agent):
    def __init__(self, store: VectorStore) -> None:
        self._store = store
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. For any question that "
                "could be answered from the knowledge base, call "
                "search_knowledge_base first instead of guessing. If the "
                "tool returns nothing relevant, say so honestly rather "
                "than making something up. Keep answers short and "
                "conversational, since they are spoken aloud."
            )
        )

    @function_tool()
    async def search_knowledge_base(self, context: RunContext, query: str) -> str:
        """Search the knowledge base for information relevant to the user's question.

        Args:
            query: A focused search query capturing what the user wants to know.
        """
        results = await self._store.search(query, top_k=TOP_K)
        if not results:
            return "No relevant information found in the knowledge base."
        return "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk, _score in results)


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name="rag-agent")
async def entrypoint(ctx: JobContext) -> None:
    store: VectorStore = ctx.proc.userdata["store"]

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=groq.STT(model=STT_MODEL, language=STT_LANGUAGE),
        llm=google.LLM(model=GEMINI_MODEL),
        tts=groq.TTS(model=TTS_MODEL, voice=TTS_VOICE),
    )

    await session.start(
        room=ctx.room,
        agent=RagAssistant(store),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user briefly and ask what they'd like to know."
    )


if __name__ == "__main__":
    cli.run_app(server)