# main.py
"""
Simple AI Server
- Provides two endpoints:
    1. /v1/chat/completions : Proxy to OpenAI or OpenRouter ChatCompletion API
    2. /v1/prefill           : Extract structured invoice data from an email and save to CSV
"""

import os
import json
import re
import csv
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# --- Environment Setup ---
# Load API keys from .env file
load_dotenv(find_dotenv(), override=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# API base URLs
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Create FastAPI app
app = FastAPI(title="AI Server", version="1.0.0")


# ---------- Helpers ----------
def normalize_for_gpt5(body: dict) -> dict:
    """
    Some GPT-5 models use `max_completion_tokens` instead of `max_tokens`.
    This function ensures compatibility by renaming the parameter if needed.
    """
    model = body.get("model", "")
    if model.startswith("gpt-5") and "max_tokens" in body and "max_completion_tokens" not in body:
        body = dict(body)
        body["max_completion_tokens"] = body.pop("max_tokens")
    return body


def is_openrouter_model(model: str) -> bool:
    """
    Identify whether the given model string refers to an OpenRouter model.
    Convention: contains "/" or ends with ":free"
    Example: "qwen/qwen3-235b-a22b:free"
    """
    return "/" in model or model.endswith(":free")


async def forward_to_upstream(body: dict) -> httpx.Response:
    """
    Forward request to the correct upstream provider:
      - OpenAI if model is gpt-* 
      - OpenRouter if model looks like vendor/...:free
    """
    model = body.get("model", "")       
    use_openrouter = is_openrouter_model(model)

    headers = {"Content-Type": "application/json"}

    if use_openrouter:
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        upstream_url = OPENROUTER_URL
    else:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        upstream_url = OPENAI_URL

    # Forward the request to the appropriate provider
    async with httpx.AsyncClient(timeout=60.0) as client:
        return await client.post(upstream_url, headers=headers, json=body)


def ensure_csv_header(path: str, fieldnames):
    """
    Ensure that the CSV file exists and has the correct header row.
    If the file is new/empty, write the header.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _strip_code_fences(text: str) -> str:
    """
    Remove Markdown code fences from LLM responses if present.
    Example: ```json {...} ``` → {...}
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def safe_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor for LLM responses.
    Tries multiple strategies:
      1. Direct json.loads()
      2. Greedy {...} regex match
      3. Stack-based balanced braces
    Raises ValueError if parsing fails.
    """
    if text is None:
        raise ValueError("Empty model response")

    text = _strip_code_fences(text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    # Attempt direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Regex greedy parse
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Stack-based brace parsing
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break

    raise ValueError("Could not parse JSON from model response")


def canonicalize_record(d: Dict[str, Any]) -> Dict[str, str]:
    """
    Ensure extracted invoice record has all required fields
    and that all values are strings.
    """
    fields = ["amount", "currency", "due_date", "description", "company", "contact"]
    out: Dict[str, str] = {}
    for k in fields:
        v = d.get(k, "")
        if v is None:
            v = ""
        if k == "amount" and isinstance(v, (int, float)):
            v = str(v)
        out[k] = str(v).strip()
    return out


# ---------- Endpoint: Chat Completions ----------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy endpoint for OpenAI/OpenRouter ChatCompletion API.
    Passes through the model and messages payload,
    normalizing parameters for GPT-5 if necessary.
    """
    body = await request.json()

    if "model" not in body or "messages" not in body:
        raise HTTPException(status_code=400, detail="Fields 'model' and 'messages' are required")

    # Compatibility shim for gpt-5*
    body = normalize_for_gpt5(body)
    # print("Using model:", body.get("model", ""))           #print which model is running

    try:
        upstream = await forward_to_upstream(body)

        # Try to parse and return JSON
        try:
            data = upstream.json()
            return JSONResponse(status_code=upstream.status_code, content=data)
        except json.JSONDecodeError:
            # If provider didn’t return JSON
            raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e!r}")


# ---------- Endpoint: Prefill ----------
class PrefillIn(BaseModel):
    """Request schema for /v1/prefill"""
    email_text: str
    model: Optional[str] = None


@app.post("/v1/prefill")
async def prefill(payload: PrefillIn):
    """
    Extract billing fields (amount, currency, due_date, description, company, contact)
    from a raw email using an LLM, then save results into data.csv.
    """
    email_text = (payload.email_text or "").strip()
    if not email_text:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "email_text is required"},
        )

    # Prompt design to force JSON output
    system = (
        "Extract structured billing fields from the email. "
        "Return ONLY a single JSON object with exactly these keys: "
        "amount, currency, due_date, description, company, contact. "
        "If unknown, use an empty string. "
        "due_date must be YYYY-MM-DD; amount must be numeric characters only (no currency symbols). "
        "No code fences, no extra text."
    )
    user = (
        "Email:\n---\n"
        f"{email_text}\n"
        "---\n\n"
        "Example JSON:\n"
        "{"
        "\"amount\":\"123.45\","
        "\"currency\":\"EUR\","
        "\"due_date\":\"2025-08-31\","
        "\"description\":\"August invoice for cloud services\","
        "\"company\":\"Acme GmbH\","
        "\"contact\":\"billing@acme.com\""
        "}"
    )

    # Default model is gpt-5-mini if none provided
    model = payload.model or "gpt-5-mini"
    # print(model)         #prints which model is running
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1000,
    }
    body = normalize_for_gpt5(body)

    try:
        upstream = await forward_to_upstream(body)

        if upstream.status_code != 200:
            return JSONResponse(
                status_code=upstream.status_code,
                content={"success": False, "message": "something went wrong"},
            )

        resp = upstream.json()
        raw_text = resp["choices"][0]["message"]["content"]

        # Parse the JSON safely
        try:
            parsed = safe_json_from_text(raw_text)
        except Exception:
            return JSONResponse(
                status_code=502,
                content={"success": False, "message": "something went wrong"},
            )

        record = canonicalize_record(parsed)

        # Save record to CSV
        csv_path = "data.csv"
        fields = ["amount", "currency", "due_date", "description", "company", "contact"]
        ensure_csv_header(csv_path, fields)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(record)

        # Response matches README
        return {"success": True, "message": "data extracted and written"}

    except httpx.HTTPError:
        return JSONResponse(
            status_code=502,
            content={"success": False, "message": "something went wrong"},
        )
