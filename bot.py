"""
Data-Analyst Telegram Bot
=========================
Receives plain-text data-analysis questions and replies with:
  {"answer": <answer>, "log_url": "<public gist url>"}

Environment variables required:
  TELEGRAM_TOKEN   – Telegram bot token from @BotFather
  GEMINI_API_KEY   – Google Gemini API key
  GITHUB_TOKEN     – GitHub PAT with `gist` scope (for log uploads)

Optional:
  CONVERSATION_TTL – seconds to keep conversation history (default 300)
"""

import asyncio
import json
import logging
import os
import re
import textwrap
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()  # loads .env from the project root before anything else

import requests
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# New google-genai SDK (pip install google-genai)
from google import genai
from google.genai import types as genai_types

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bot")

# ─── Configuration ───────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN     = os.environ["GITHUB_TOKEN"]
CONVERSATION_TTL = int(os.environ.get("CONVERSATION_TTL", "300"))

# Initialise the new genai client
client = genai.Client(api_key=GEMINI_API_KEY)

# Thread pool for running sync work without blocking the async event loop
_executor = ThreadPoolExecutor(max_workers=4)

# ─── Tool implementations ────────────────────────────────────────────────────

def fetch_url(url: str, max_bytes: int = 200_000) -> str:
    """Download a URL and return its text content (HTML stripped to text)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (data-analyst-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30, stream=True)
        resp.raise_for_status()
        content = b""
        for chunk in resp.iter_content(chunk_size=8192):
            content += chunk
            if len(content) >= max_bytes:
                break
        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            soup = BeautifulSoup(content, "lxml")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)[:50_000]
        return content.decode("utf-8", errors="replace")[:50_000]
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def run_python(code: str) -> str:
    """
    Execute Python code in a restricted namespace and return stdout + result.
    Available: pandas, requests, json, re, math, statistics, datetime, itertools.
    """
    import io, contextlib, math, statistics, itertools, signal
    import pandas as pd

    namespace = {
        "pd": pd,
        "pandas": pd,
        "requests": requests,
        "json": json,
        "re": re,
        "math": math,
        "statistics": statistics,
        "datetime": datetime,
        "itertools": itertools,
        "fetch_url": fetch_url,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<agent>", "exec"), namespace)  # noqa: S102
        out = buf.getvalue()
        # also grab last expression value
        if "_result" in namespace:
            out += f"\n_result = {namespace['_result']!r}"
        return out[:10_000] if out else "(no output)"
    except Exception as exc:
        return f"EXCEPTION: {exc}"


def dispatch_tool(name: str, args: dict) -> str:
    if name == "fetch_url":
        return fetch_url(**args)
    if name == "run_python":
        return run_python(**args)
    return f"Unknown tool: {name}"


# ─── Tool declarations for Gemini ────────────────────────────────────────────

FETCH_URL_DECL = genai_types.FunctionDeclaration(
    name="fetch_url",
    description=(
        "Fetch the text content of a public URL. "
        "Use this to download datasets, MOSPI pages, Wikipedia, CSV files, etc. "
        "Returns plain text (HTML tags stripped). Max 50 000 chars."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "url": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="The full URL to fetch.",
            )
        },
        required=["url"],
    ),
)

RUN_PYTHON_DECL = genai_types.FunctionDeclaration(
    name="run_python",
    description=(
        "Execute Python code and return stdout. "
        "pandas, requests, json, re, math, statistics, datetime, itertools are pre-imported. "
        "fetch_url(url) is also available inside the code. "
        "Use for data wrangling, CSV parsing, calculations, and analysis. "
        "Always print your findings so they appear in the output."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "code": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Python source code to execute.",
            )
        },
        required=["code"],
    ),
)

TOOLS = [genai_types.Tool(function_declarations=[FETCH_URL_DECL, RUN_PYTHON_DECL])]

# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data analyst agent. You receive data-analysis questions—sometimes with
inline data, sometimes pointing at public datasets (MOSPI, data.gov.in, Wikipedia, etc.).

Your job:
1. Understand the question thoroughly.
2. Use fetch_url and run_python tools as needed to retrieve and analyse data.
3. Produce the final answer in the EXACT JSON shape the question requests.

CRITICAL RULES:
- Your FINAL reply must be EXACTLY one JSON object and nothing else.
  It must have two keys: "answer" and "log_url".
  The "log_url" value will be filled in by the system—output the placeholder text
  __LOG_URL_PLACEHOLDER__ for log_url in your final JSON, e.g.:
  {"answer": {"state": "Assam"}, "log_url": "__LOG_URL_PLACEHOLDER__"}
- Do NOT wrap the JSON in markdown fences.
- Do NOT add explanation text outside the JSON.
- If the question asks for a specific JSON shape for "answer", follow it precisely.
- Use tools iteratively until you are confident in the answer.
- For MOSPI data: try https://mospi.gov.in and search for the specific dataset.
  Common datasets: SRS Statistical Reports, Health/MMR data, Census data.
- When fetching CSV/Excel data via run_python, use pd.read_csv() or pd.read_excel().
- If a URL fails, try an alternative source (Wikipedia, data.gov.in, etc.).
""").strip()


# ─── Gemini agent ────────────────────────────────────────────────────────────

def run_agent(question: str, run_log: list) -> str:
    """
    Run the Gemini agent on `question`. Appends to run_log.
    Returns the final text response.
    Uses the new google-genai SDK with proper multi-turn tool calling.
    """
    run_log.append({
        "event": "user_message",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": question,
    })

    # Build the initial conversation history
    contents: list[genai_types.Content] = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=question)],
        )
    ]

    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,
        temperature=0.1,
    )

    max_iterations = 2
    for iteration in range(max_iterations):
        logger.info(f"Agent iteration {iteration}")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
#minor change
        if not response.candidates:
            run_log.append({
                "event": "error",
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "response.candidates is empty (fully blocked response)",
            })
            return '{"answer": "Response was blocked by safety filters", "log_url": "__LOG_URL_PLACEHOLDER__"}'

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        logger.info(f"Finish reason: {finish_reason}")

        # candidate.content can be None when the response is blocked
        # (SAFETY, RECITATION) or has no parts (some MAX_TOKENS cases).
        if candidate.content is None:
            # Try response.text as a last resort
            fallback = ""
            try:
                fallback = response.text or ""
            except Exception:
                pass
            run_log.append({
                "event": "model_response",
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": fallback,
                "tool_calls": [],
                "finish_reason": str(finish_reason),
                "note": "candidate.content was None",
            })
            run_log.append({
                "event": "final_answer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": fallback,
            })
            return fallback or '{"answer": "No response from model", "log_url": "__LOG_URL_PLACEHOLDER__"}'

        # Collect text parts and tool calls from the response
        text_parts = []
        tool_calls = []
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            if getattr(part, "function_call", None):
                fc = part.function_call
                tool_calls.append({
                    "name": fc.name,
                    "args": dict(fc.args),
                })

        run_log.append({
            "event": "model_response",
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "finish_reason": str(finish_reason),
        })

        # Append the model's turn to the conversation
        contents.append(candidate.content)

        if not tool_calls:
            # No tool calls → this is the final answer
            final_text = "\n".join(text_parts).strip()
            run_log.append({
                "event": "final_answer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": final_text,
            })
            return final_text

        # Execute tools and feed results back as a "tool" turn
        tool_result_parts = []
        for tc in tool_calls:
            logger.info(f"Calling tool {tc['name']} with args {list(tc['args'].keys())}")
            result = dispatch_tool(tc["name"], tc["args"])
            run_log.append({
                "event": "tool_result",
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tc["name"],
                "result_preview": result[:500],
            })
            tool_result_parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=tc["name"],
                        response={"result": result},
                    )
                )
            )

        # Append tool results as a "user" turn (required by the new SDK)
        contents.append(
            genai_types.Content(
                role="user",
                parts=tool_result_parts,
            )
        )

    # Max iterations hit
    run_log.append({
        "event": "max_iterations_reached",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return '{"answer": "Could not complete analysis within iteration limit", "log_url": "__LOG_URL_PLACEHOLDER__"}'


# ─── GitHub Gist upload ──────────────────────────────────────────────────────

def upload_gist(run_log: list) -> str:
    """Upload run_log as a public GitHub Gist and return the raw URL."""
    filename = f"run_{uuid.uuid4().hex[:8]}.jsonl"
    content = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in run_log)
    payload = {
        "description": "Data-analyst bot run log",
        "public": True,
        "files": {filename: {"content": content}},
    }
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.post(
        "https://api.github.com/gists",
        json=payload,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    gist_id = data["id"]
    owner = data["owner"]["login"]
    # Use the permanent (non-versioned) raw URL
    permanent = f"https://gist.githubusercontent.com/{owner}/{gist_id}/raw/{filename}"
    logger.info(f"Gist uploaded: {permanent}")
    return permanent


# ─── Conversation history ────────────────────────────────────────────────────

# Maps chat_id -> list of (timestamp, text) tuples
conversation_store: dict[int, list] = defaultdict(list)


def get_conversation_context(chat_id: int, new_message: str) -> str:
    """
    Build a context-aware prompt from conversation history + new message.
    Prunes entries older than CONVERSATION_TTL seconds.
    """
    now = time.time()
    history = conversation_store[chat_id]
    # Prune old entries
    history[:] = [(ts, msg) for ts, msg in history if now - ts < CONVERSATION_TTL]

    history.append((now, new_message))

    if len(history) == 1:
        return new_message

    # Build multi-turn context
    turns = "\n".join(f"[Turn {i+1}]: {msg}" for i, (_, msg) in enumerate(history))
    return (
        f"This is a multi-turn conversation. Here are all messages so far:\n\n"
        f"{turns}\n\n"
        f"Please answer the LAST turn (Turn {len(history)}) above."
    )


# ─── Telegram handler ────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    text = message.text.strip()

    logger.info(f"Received message from chat_id={chat_id}: {text[:100]}")

    run_log: list = []
    try:
        # Build prompt with conversation history
        prompt = get_conversation_context(chat_id, text)

        # Run the agent in a thread pool so we don't block the event loop
        loop = asyncio.get_running_loop()
        raw_answer = await loop.run_in_executor(
            _executor, lambda: run_agent(prompt, run_log)
        )

        # Upload log to Gist (also in executor)
        try:
            log_url = await loop.run_in_executor(
                _executor, lambda: upload_gist(run_log)
            )
        except Exception as e:
            logger.error(f"Gist upload failed: {e}")
            log_url = "https://github.com/upload-failed"

        # Inject log_url into the answer
        final_json = raw_answer.replace("__LOG_URL_PLACEHOLDER__", log_url)

        # Validate it's proper JSON (best-effort)
        try:
            parsed = json.loads(final_json)
            if "log_url" not in parsed:
                parsed["log_url"] = log_url
            final_json = json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            json_match = re.search(r'\{.*\}', final_json, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if "log_url" not in parsed:
                        parsed["log_url"] = log_url
                    final_json = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    final_json = json.dumps({
                        "answer": raw_answer,
                        "log_url": log_url,
                    }, ensure_ascii=False)
            else:
                final_json = json.dumps({
                    "answer": raw_answer,
                    "log_url": log_url,
                }, ensure_ascii=False)

        logger.info(f"Sending reply: {final_json[:200]}")
        await message.reply_text(final_json)

    except Exception as exc:
        logger.exception(f"Error processing message: {exc}")
        error_resp = json.dumps({
            "answer": f"Error: {exc}",
            "log_url": "https://error",
        })
        await message.reply_text(error_resp)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Starting Data-Analyst Telegram Bot…")
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    # Handle all non-command text messages
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    # Python 3.10+ no longer auto-creates a default event loop.
    # Python 3.14 raises RuntimeError in asyncio.get_event_loop() when none exists.
    # python-telegram-bot calls get_event_loop() internally, so we must create
    # and register one before handing control to PTB.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main()
