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

from dotenv import load_dotenv
load_dotenv()  # loads .env from the project root before anything else

import httpx
import requests
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

import google.generativeai as genai

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bot")

# ─── Configuration ───────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN   = os.environ["GITHUB_TOKEN"]
CONVERSATION_TTL = int(os.environ.get("CONVERSATION_TTL", "300"))

genai.configure(api_key=GEMINI_API_KEY)

# ─── Tool definitions ────────────────────────────────────────────────────────

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
    import io, contextlib, math, statistics, itertools
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


# ─── Gemini agent ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch the text content of a public URL. "
            "Use this to download datasets, MOSPI pages, Wikipedia, etc. "
            "Returns plain text (HTML tags stripped)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."}
            },
            "required": ["url"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute Python code and return stdout. "
            "pandas, requests, json, re, math, statistics, datetime are available. "
            "Use this for data wrangling, calculations, and analysis. "
            "Call fetch_url() inside the code if you need to download something. "
            "Print your final answer clearly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute.",
                }
            },
            "required": ["code"],
        },
    },
]

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
- For MOSPI data, try https://mospi.gov.in or search for the specific dataset.
  Common datasets: SRS Statistical Reports, Health/MMR data, Census data, etc.
- When fetching CSV/Excel data via run_python, use pd.read_csv() or pd.read_excel().
""").strip()


def dispatch_tool(name: str, args: dict) -> str:
    if name == "fetch_url":
        return fetch_url(**args)
    if name == "run_python":
        return run_python(**args)
    return f"Unknown tool: {name}"


def run_agent(question: str, run_log: list) -> str:
    """
    Run the Gemini agent on `question`. Appends to run_log.
    Returns the final text response.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,
    )

    run_log.append({
        "event": "user_message",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": question,
    })

    chat = model.start_chat(enable_automatic_function_calling=False)

    # Build messages from question
    messages = [question]

    max_iterations = 15
    for iteration in range(max_iterations):
        logger.info(f"Agent iteration {iteration}")

        response = chat.send_message(messages[-1] if iteration == 0 else messages)
        candidate = response.candidates[0]

        # Collect all parts
        text_parts = []
        tool_calls = []
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
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
        })

        if not tool_calls:
            # Final response
            final_text = "\n".join(text_parts).strip()
            run_log.append({
                "event": "final_answer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": final_text,
            })
            return final_text

        # Execute tools and feed results back
        tool_results = []
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
            tool_results.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tc["name"],
                        response={"result": result},
                    )
                )
            )

        messages = [genai.protos.Content(parts=tool_results, role="tool")]

    # Max iterations hit – return whatever we have
    run_log.append({"event": "max_iterations_reached", "timestamp": datetime.now(timezone.utc).isoformat()})
    return '{"answer": "Could not complete analysis within iteration limit", "log_url": "__LOG_URL_PLACEHOLDER__"}'


# ─── GitHub Gist upload ──────────────────────────────────────────────────────

def upload_gist(run_log: list) -> str:
    """Upload run_log as a public GitHub Gist and return the raw URL."""
    filename = f"run_{uuid.uuid4().hex[:8]}.jsonl"
    content = "\n".join(json.dumps(entry) for entry in run_log)
    payload = {
        "description": "Data-analyst bot run log",
        "public": True,
        "files": {filename: {"content": content}},
    }
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.post("https://api.github.com/gists", json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    raw_url = data["files"][filename]["raw_url"]
    # Make URL permanent (not versioned)
    gist_id = data["id"]
    permanent = f"https://gist.githubusercontent.com/{data['owner']['login']}/{gist_id}/raw/{filename}"
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

        # Run the agent
        raw_answer = run_agent(prompt, run_log)

        # Upload log to Gist
        try:
            log_url = upload_gist(run_log)
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
    # Python 3.10+ no longer auto-creates an event loop; 3.14 raises RuntimeError
    # without one, so we create it explicitly before handing off to PTB.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main()
