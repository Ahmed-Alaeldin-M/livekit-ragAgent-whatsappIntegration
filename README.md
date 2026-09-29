LiveKit voice agent with RAG

What's here: a voice agent (agent.py) that answers questions by calling a search_knowledge_base tool, which does cosine-similarity search over a local vector index (rag/store.py). Documents are chunked and embedded offline by rag/ingest.py.

Setup

    pip install -r requirements.txt
    cp .env.example .env.local
    # fill in LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
    # / GOOGLE_API_KEY (free key from https://aistudio.google.com/apikey)
    # / GROQ_API_KEY

    mkdir knowledge_base
    # drop in .txt / .md / .pdf files

    python -m rag.ingest --source ./knowledge_base --out ./index

Run

    python agent.py console

This runs fully local with your mic/speakers, no LiveKit room needed. Use `python agent.py dev` to connect to a real room instead.

Notes

- Re-run rag/ingest.py any time the knowledge_base folder changes; it rebuilds ./index from scratch.
- rag/store.py embeds locally with sentence-transformers (BAAI/bge-m3 by default) for strong multilingual (Arabic/English) retrieval, no API key needed. First run downloads and caches the model; override with the EMBEDDING_MODEL env var.
- This is a standard STT-LLM-TTS pipeline (not the Gemini Live speech-to-speech model). LLM is Gemini (google.LLM, text-only) via a plain Gemini API key — no GCP project or billing account needed. STT/TTS are Groq (Whisper + Orpheus), also plain API key. Google Cloud Speech-to-Text/Text-to-Speech are intentionally not used here: those always need a real GCP service account regardless of model, so they're a separate setup step if you ever want them.
- Swap STT_MODEL / STT_LANGUAGE / GEMINI_MODEL / TTS_MODEL / TTS_VOICE at the top of agent.py to change models.
- The vector store is a flat numpy array with cosine similarity — fine up to tens of thousands of chunks. Swap rag/store.py for pgvector/Qdrant if you outgrow it; the add/save/load/search interface is what agent.py depends on, so nothing else needs to change.

WhatsApp

whatsapp/webhook.py is a second front door to the same knowledge base — a FastAPI app that receives WhatsApp Cloud API messages, answers them with rag/store.py + Gemini, and sends the reply back. Run it separately from the voice agent:

    uvicorn whatsapp.webhook:app --reload --port 8000

For local testing, Meta needs a public HTTPS URL to reach this — tunnel it (ngrok, Cloudflare Tunnel, etc.) and paste that URL into the Meta App Dashboard under WhatsApp > Configuration > Webhook, subscribed to the "messages" field. Fill in WEBHOOK_VERIFY_TOKEN / WHATSAPP_APP_SECRET / WHATSAPP_ACCESS_TOKEN in .env.local — details on where to find each are in .env.example.

Notes specific to the webhook:
- Every incoming POST is verified against X-Hub-Signature-256 using WHATSAPP_APP_SECRET; requests that don't match are rejected with 403. This isn't optional — it's the only thing stopping someone else from POSTing fake messages (and fake replies) to your endpoint.
- Meta retries a delivery if it doesn't get a fast 2xx back, so the webhook responds immediately and answers the message in a FastAPI background task. Message IDs are deduped in memory so a retry doesn't trigger two replies — that dedupe is per-process and resets on restart, so if you ever run more than one worker, swap it for a shared store (Redis, a DB row) instead.
- Only text messages are handled; images, audio, location, etc. are currently ignored. Extend extract_text_messages() in whatsapp/webhook.py to add other types.
- The LLM here is plain Gemini via whatsapp/rag_reply.py (WHATSAPP_LLM_MODEL env var to change it) — independent of whichever LLM the voice agent uses, since this is a synchronous text Q&A call, not a LiveKit session.
