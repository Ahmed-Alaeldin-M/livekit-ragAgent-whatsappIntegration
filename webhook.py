"""FastAPI webhook for WhatsApp Cloud API: verifies the subscription
handshake, accepts incoming messages, answers them from the RAG
knowledge base, and sends the reply back via the Graph API.

Run:
    uvicorn whatsapp.webhook:app --reload --port 8000

For local testing, Meta needs a public HTTPS URL to call -- tunnel this
port with ngrok/Cloudflare Tunnel/similar and put that URL + WEBHOOK_VERIFY_TOKEN
into the Meta App Dashboard's WhatsApp > Configuration > Webhook screen,
subscribed to the "messages" field.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from store import VectorStore
from rag_reply import generate_reply

load_dotenv(".env.example")

logger = logging.getLogger("whatsapp-webhook")
logger.setLevel(logging.INFO)

INDEX_PATH = os.getenv("RAG_INDEX_PATH", "./index")
WHATSAPP_API_VERSION = "v26.0"

# Meta retries a webhook delivery if it doesn't get a fast 2xx, which can
# otherwise cause the same message to be answered twice. A bounded
# in-memory set is enough to dedupe those retries for a single-process
# deployment; it resets on restart and isn't shared across workers, so
# swap this for a shared store (Redis, a DB table) before running more
# than one worker process.
_seen_message_ids: deque[str] = deque(maxlen=2000)
_seen_message_id_set: set[str] = set()


def _mark_seen(message_id: str) -> bool:
    """Returns True if this message_id is new, False if already processed."""
    if message_id in _seen_message_id_set:
        return False
    if len(_seen_message_ids) == _seen_message_ids.maxlen:
        _seen_message_id_set.discard(_seen_message_ids[0])
    _seen_message_ids.append(message_id)
    _seen_message_id_set.add(message_id)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = VectorStore.load(INDEX_PATH)
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="WhatsApp RAG Webhook", lifespan=lifespan)


def verify_token(token: str) -> None:
    expected_token = os.getenv("WEBHOOK_VERIFY_TOKEN")
    if not expected_token or not hmac.compare_digest(token, expected_token):
        raise HTTPException(status_code=403, detail="Invalid verify token")


def verify_signature(raw_body: bytes, signature_header: str | None) -> None:
    app_secret = os.getenv("WHATSAPP_APP_SECRET")
    if not app_secret:
        raise HTTPException(
            status_code=500, detail="WHATSAPP_APP_SECRET is not configured on the server"
        )
    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=403, detail="Missing signature")

    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=403, detail="Invalid signature")


def extract_text_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull out text messages from a (possibly batched) webhook payload.
    Non-text messages and status updates (delivery/read receipts) are
    skipped here -- extend this if you need to handle other types."""
    out = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            phone_number_id = value.get("metadata", {}).get("phone_number_id")
            for msg in value.get("messages", []):
                if msg.get("type") == "text":
                    out.append(
                        {
                            "message_id": msg.get("id"),
                            "from": msg.get("from"),
                            "text": msg.get("text", {}).get("body", ""),
                            "phone_number_id": phone_number_id,
                        }
                    )
    return out


async def send_whatsapp_message(
    http_client: httpx.AsyncClient, phone_number_id: str, to: str, body: str
) -> None:
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    if not access_token:
        logger.error("WHATSAPP_ACCESS_TOKEN is not configured, cannot send reply")
        return

    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{phone_number_id}/messages"
    resp = await http_client.post(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body},
        },
    )
    if resp.status_code >= 400:
        logger.error("WhatsApp send failed (%s): %s", resp.status_code, resp.text)


async def process_message(
    store: VectorStore, http_client: httpx.AsyncClient, message: dict[str, Any]
) -> None:
    try:
        reply_text = await generate_reply(store, message["text"])
    except Exception:
        logger.exception("Failed to generate a reply for message %s", message["message_id"])
        reply_text = "Sorry, something went wrong answering that -- please try again."

    await send_whatsapp_message(
        http_client, message["phone_number_id"], message["from"], reply_text
    )


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> PlainTextResponse:
    """Verify a webhook subscription and return its challenge."""
    if hub_mode != "subscribe" or not hub_verify_token or hub_challenge is None:
        raise HTTPException(status_code=400, detail="Invalid webhook verification request")

    verify_token(hub_verify_token)
    return PlainTextResponse(content=hub_challenge, status_code=200)


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Accept incoming webhook events, verify authenticity, and answer any
    new text messages in the background so Meta gets a fast 2xx."""
    raw_body = await request.body()
    verify_signature(raw_body, request.headers.get("X-Hub-Signature-256"))

    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    for message in extract_text_messages(payload):
        if not message["message_id"] or not _mark_seen(message["message_id"]):
            continue  # missing id, or already processed (Meta retry)
        background_tasks.add_task(
            process_message, request.app.state.store, request.app.state.http_client, message
        )

    return JSONResponse(content={"status": "received"}, status_code=200)
