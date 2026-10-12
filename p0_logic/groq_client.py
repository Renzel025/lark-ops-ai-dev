"""
Groq API: chat completion, vision OCR, translation to Chinese.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import requests

from . import config as _config
from .perf_log import perf_log
from . import text_processing as _text

log = logging.getLogger("lark-ops-ai")

GROQ_API_KEY = _config.GROQ_API_KEY
GROQ_BASE = _config.GROQ_BASE
GROQ_MODEL = _config.GROQ_MODEL
GROQ_VISION_MODEL = _config.GROQ_VISION_MODEL

# Groq's current chat models are reasoning models. Their thinking counts against max_tokens, so the
# small budgets used here (120-500) could be spent thinking and return no answer, and Qwen writes
# <think> into the content by default (breaks JSON parsing). Turn reasoning down/hidden for them.
_REASONING_MIN_MAX_TOKENS = 1024
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _reasoning_params(model: str) -> dict:
    m = (model or "").strip().lower()
    if m.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    if m.startswith("qwen/qwen3"):
        return {"reasoning_effort": "none", "reasoning_format": "hidden"}
    return {}


def _timeout_kw():
    return _config.timeout_kw()


def _groq_runtime() -> Tuple[str, str, str]:
    """Fresh key/models after ``reload_env_runtime`` (module-level GROQ_* is import-time only)."""
    _config.reload_env_runtime()
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    model = (os.getenv("GROQ_MODEL") or GROQ_MODEL or "qwen/qwen3.8-27b").strip()
    vision = (os.getenv("GROQ_VISION_MODEL") or GROQ_VISION_MODEL or "llama-3.2-11b-vision-preview").strip()
    return key, model, vision


def groq_chat_once(system_prompt: str, user_content: str, max_tokens: int, model: Optional[str] = None) -> str:
    api_key, default_model, _vision = _groq_runtime()
    if not api_key:
        return ""
    use_model = (model or default_model).strip()
    url = f"{GROQ_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": use_model,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    reasoning = _reasoning_params(use_model)
    if reasoning:
        payload.update(reasoning)
        payload["max_tokens"] = max(int(max_tokens or 0), _REASONING_MIN_MAX_TOKENS)
    t0 = time.perf_counter()
    try:
        try:
            r = requests.post(url, headers=headers, json=payload, **_timeout_kw())
            if r.status_code == 400 and reasoning:
                # Reasoning knobs rejected for this model — retry plain rather than fail the call.
                log.warning(
                    "Groq rejected reasoning params for model=%s (%s) — retrying without them",
                    use_model,
                    (r.text or "")[:160],
                )
                for k in reasoning:
                    payload.pop(k, None)
                r = requests.post(url, headers=headers, json=payload, **_timeout_kw())
        except Exception as e:
            log.error("Groq request error: %s", e)
            return ""
        if r.status_code != 200:
            log.error("Groq API error: %s - %s", r.status_code, (r.text or "")[:200])
            return ""
        try:
            j = r.json() if r.text else {}
            content = j.get("choices", [{}])[0].get("message", {}).get("content") or ""
            return _THINK_RE.sub("", content).strip()
        except Exception:
            return ""
    finally:
        perf_log(f"groq_chat_once model={use_model}", t0)


def groq_vision_ocr(image_bytes: bytes) -> str:
    api_key, _default_model, vision_model = _groq_runtime()
    if not api_key or not image_bytes:
        return ""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    url = f"{GROQ_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_prompt = (
        "You extract ONLY visible text from screenshots.\n"
        "Rules:\n"
        "- Output only text visible in the image.\n"
        "- Preserve names, @mentions, IDs, timestamps, numbers, and line breaks.\n"
        "- Do not summarize.\n"
        "- Do not infer.\n"
        "- Do not create templates.\n"
        "- Do not add labels like Player ID, Game Name, Error Code unless they are literally visible.\n"
        "- If a part is unreadable, skip that part instead of inventing."
    )
    payload = {
        "model": vision_model,
        "temperature": 0.0,
        "max_tokens": 2200,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the screenshot text exactly as visible."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    t0 = time.perf_counter()
    try:
        try:
            r = requests.post(url, headers=headers, json=payload, **_timeout_kw())
        except Exception as e:
            log.error("Groq vision request error: %s", e)
            return ""
        if r.status_code != 200:
            log.error("Groq vision error HTTP=%s head=%s", r.status_code, (r.text or "")[:300])
            return ""
        try:
            j = r.json() if r.text else {}
            return (j.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        except Exception as e:
            log.error("Groq vision parse error: %s", e)
            return ""
    finally:
        perf_log(f"groq_vision_ocr model={vision_model}", t0)


def translate_to_zh(text: str) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    if _text.is_not_specified(src):
        return "未指定"
    if _text.looks_like_chinese(src):
        return _text.normalize_gaming_zh(_text.clean_single_line_translation(src))
    system_prompt = (
        "You are a professional translator for gaming incident reports.\n"
        "Translate the user's text into Simplified Chinese.\n"
        "STRICT RULES:\n"
        "- Translate ONLY the provided text.\n"
        "- Do NOT add explanations.\n"
        "- Do NOT add examples.\n"
        "- Do NOT summarize.\n"
        "- Do NOT expand the content.\n"
        "- Do NOT create incident IDs, templates, bullets, or extra paragraphs.\n"
        "- Keep numbers, IDs, acronyms, product names, and timestamps exactly as they are.\n"
        "- Use 玩家 for game players, not 球员.\n"
        "- Output exactly ONE concise line of Simplified Chinese only.\n"
    )
    out = groq_chat_once(system_prompt, src, max_tokens=120)
    cleaned = _text.clean_single_line_translation(out)
    if not cleaned:
        return src
    return _text.normalize_gaming_zh(cleaned)


def translate_issue_impact_pair_to_zh(en_issue: str, en_impact: str) -> Tuple[str, str]:
    """
    Fallback when one-shot overview is off or fails: two Groq calls in parallel.
    """
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fi = pool.submit(translate_to_zh, en_issue)
            fj = pool.submit(translate_to_zh, en_impact)
            return fi.result(), fj.result()
    finally:
        perf_log("groq translate_issue_impact_pair", t0)


def _repair_truncated_json_object(text: str) -> Optional[dict]:
    """Salvage a JSON object the model got cut off mid-way through (hit ``max_tokens``).

    Trims back to the last COMPLETE ``"key": value`` pair at the top level and closes the brace, so
    a response ending in ``…"summary": "…", "issue_fingerprint":`` still yields the fields that did
    arrive. Returns ``None`` when nothing usable survives.
    """
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    depth = 0
    in_str = False
    esc = False
    last_pair_end = -1
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 1:
            last_pair_end = i
    if last_pair_end <= 0:
        return None
    try:
        obj = json.loads(s[:last_pair_end] + "}")
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_json_object(raw: str) -> Optional[dict]:
    if not (raw or "").strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
        # No closing brace (or an unparsable one) — most often a max_tokens cut-off.
        repaired = _repair_truncated_json_object(text)
        if repaired is not None:
            log.warning("JSON was truncated — salvaged %s field(s): %s", len(repaired), sorted(repaired))
        return repaired


def groq_p0_keyword_declares_new_bridge(message_text: str) -> Optional[bool]:
    """
    Classify whether **message_text** is a **real P0 declaration** for the *current* incident/workflow
    (someone asserts severity is P0 / declares escalation / asks to start P0 handling),
    vs. merely mentioning \"p0\" in passing (chat about an existing bridge, RCA story, generic plural \"p0 issues\", etc.).

    Note: JSON field name is historical (``declares_new_p0``); semantics are **P0 declaration / escalation**, not only \"open VC\".

    Returns:
        ``True`` / ``False`` when JSON is parsed; ``None`` on missing key/API/parse failure
        — callers should **fail-open** (preserve keyword trigger) so an outage does not block real incidents.
    """
    if not GROQ_API_KEY:
        return None
    t = (message_text or "").strip()
    if not t:
        return None
    t = t[:4500]
    system_prompt = (
        "You triage Lark chat lines for an on-call bot. The bot starts a P0 incident flow when someone "
        "**clearly declares or assigns P0** to the situation they are talking about (right now), not when "
        "they only mention the letters \"P0\" casually.\n\n"
        "declares_new_p0=true when the speaker **asserts** the current issue/situation **is** P0 / **should be "
        "handled as** P0 / **we treat this as** P0 / **escalat** to P0 / **declare** P0 / needs P0 **now**. "
        "Short confirmations count, e.g. \"this is p0\", \"this issue is p0\", \"yes team this issue is p0\", "
        "\"declaring p0\", \"escalated to p0\".\n\n"
        "declares_new_p0=false when:\n"
        "- the line is mainly a **question** or **asking permission** (\"is this p0?\", \"can we tag as p0?\", "
        "\"may we…\") without a firm declaration;\n"
        "- **negation** or \"not p0\" / only P1;\n"
        "- **status inside** an already-running P0 meeting/call with **no new** declaration "
        '(e.g. \"checking in the p0 meeting\", \"discussing on the p0 bridge\");\n'
        "- **generic or plural** talk (\"p0 issues\", \"the p0 process\") with **no** specific incident declared P0;\n"
        "- **past/historical** narrative only (\"last week was p0\", \"this was a p0 in 2024\") with no present declaration.\n\n"
        "When unsure, prefer **true** if the speaker sounds like they are **assigning P0 to the current issue**.\n\n"
        "Output ONLY valid JSON: {\"declares_new_p0\": true} or {\"declares_new_p0\": false}"
    )
    user_prompt = f"MESSAGE:\n{t}"
    raw = groq_chat_once(system_prompt, user_prompt, max_tokens=120)
    obj = _parse_json_object(raw or "")
    if not obj:
        log.warning("groq_p0_keyword: JSON parse failed head=%s", (raw or "")[:200])
        return None
    v = obj.get("declares_new_p0")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return None


_PRIORITY_KEYWORD_CLASSIFY_SYSTEM = (
    "You triage Lark **incident group** chat for an on-call bot.\n"
    "The message contains **P0** and/or **P1**. Decide what the speaker wants **right now**.\n\n"
    "Output ONLY valid JSON:\n"
    '{"intent":"...", "reason":"one short sentence"}\n\n'
    "intent values:\n"
    "- **declare_p0** — speaker asserts the **current** situation **is / should be handled as P0 NOW** "
    '(e.g. "p0", "this is p0", "declaring p0", "escalate to p0", "we treat this as p0").\n'
    "- **declare_p1** — same for **P1** / priority 1 **now**.\n"
    "- **handoff** — sharing **context** (meegle, ticket, story link, case ID) and P0/P1 is only a **label** "
    'on the case, NOT asking to open a bridge (e.g. "here is the meegle on this P0 case").\n'
    "- **mention_only** — casual/historical/generic mention (RCA, p0 issues last week, status inside "
    "existing meeting, process talk) with **no new** declaration.\n"
    "- **question** — asking if something is P0/P1 or asking permission, not declaring.\n"
    "- **negation** — clearly **not** P0/P1 or no escalation needed.\n\n"
    "If both P0 and P1 appear, pick the priority the speaker is **actually assigning** to the live incident.\n"
    "If the message mentions **only P0** (no P1), never return **declare_p1**. If only P1 (no P0), never return **declare_p0**.\n"
    "Phrases like **can we consider this (one) as P0** are **question** (asking permission), not declare_p0.\n"
    "When unsure between declare vs handoff, prefer **handoff** if they are mainly **sharing a link/ticket**.\n"
    "When unsure between declare vs mention_only, prefer **declare_p0** only if they sound like they want **action now**."
)


def _parse_priority_keyword_classification(raw: str, provider: str) -> Optional[dict]:
    obj = _parse_json_object(raw or "")
    if not obj:
        log.warning("classify_priority_keyword (%s): JSON parse failed head=%s", provider, (raw or "")[:200])
        return None
    intent = str(obj.get("intent") or "").strip().lower()
    allowed = {
        "declare_p0",
        "declare_p1",
        "handoff",
        "mention_only",
        "question",
        "negation",
    }
    if intent not in allowed:
        log.warning("classify_priority_keyword (%s): unknown intent=%r", provider, intent)
        return None
    reason = str(obj.get("reason") or "").strip()[:300]
    return {"intent": intent, "reason": reason, "provider": provider}


def classify_priority_keyword(message_text: str, provider: Optional[str] = None) -> Optional[dict]:
    """
    One LLM call: declaration vs mention/handoff/question for ``p0`` / ``p1`` keyword hits.
    Failover chain (``auto``): **claude → gemini → groq**. Pass ``provider`` to force one.
    """
    from . import config as _cfg

    t = (message_text or "").strip()
    if not t:
        return None
    t = t[:4500]
    user_prompt = f"MESSAGE:\n{t}"

    def _via_claude() -> Optional[dict]:
        from .anthropic_client import anthropic_chat_once

        if not _cfg.anthropic_claude_configured():
            return None
        raw = anthropic_chat_once(_PRIORITY_KEYWORD_CLASSIFY_SYSTEM, user_prompt, max_tokens=180)
        return _parse_priority_keyword_classification(raw, "claude")

    def _via_gemini() -> Optional[dict]:
        from .gemini_client import gemini_chat_once

        if not _cfg.get_gemini_api_key():
            return None
        raw = gemini_chat_once(_PRIORITY_KEYWORD_CLASSIFY_SYSTEM, user_prompt, max_tokens=160)
        return _parse_priority_keyword_classification(raw, "gemini")

    def _via_groq() -> Optional[dict]:
        if not GROQ_API_KEY:
            return None
        raw = groq_chat_once(_PRIORITY_KEYWORD_CLASSIFY_SYSTEM, user_prompt, max_tokens=160)
        return _parse_priority_keyword_classification(raw, "groq")

    _dispatch = {"claude": _via_claude, "gemini": _via_gemini, "groq": _via_groq}

    names = (
        [(provider or "").strip().lower()]
        if provider
        else _cfg.priority_keyword_ai_provider_chain()
    )
    for name in names:
        fn = _dispatch.get(name)
        if not fn:
            continue
        result = fn()
        if result:
            if len(names) > 1 or (provider is None and len(_cfg.priority_keyword_ai_provider_chain()) > 1):
                if name != names[0]:
                    log.info("classify_priority_keyword: failover succeeded via %s", name)
            return result
        log.info("classify_priority_keyword: %s returned no result — trying next provider", name)
    return None


def groq_classify_priority_keyword(message_text: str) -> Optional[dict]:
    """Classify priority keyword intent via Groq."""
    return classify_priority_keyword(message_text)


def classify_graph_screenshot_request(message_text: str) -> Optional[dict]:
    """Natural-language Grafana screenshot triage (Claude preferred, Groq fallback)."""
    from features.screenshot.graph_screenshot_ai import classify_graph_screenshot_request as _classify

    return _classify(message_text)


def _scrub_issue_source_for_model(issue_source: str) -> str:
    src = (issue_source or "").strip()
    if not src:
        return ""
    best = _text.issue_context_for_model(src) or src
    scrubbed = re.sub(r"\[Screenshot\s+\d+\s+OCR\]", " ", best, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b\d{6,}\b", "<ID>", scrubbed)
    scrubbed = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "<DATE>", scrubbed)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    return scrubbed


def build_overview_oneshot_prompts(issue_source: str, impact_en: str) -> Optional[Tuple[str, str]]:
    """
    ``(system_prompt, user_prompt)`` for the overview issue+bilingual one-shot, or ``None`` when
    there is nothing to summarize. Shared by the Groq and Claude providers so both emit the same
    JSON shape (``issue_en`` / ``zh_issue`` / ``zh_impact``). See ``overview_ai``.
    """
    scrubbed = _scrub_issue_source_for_model(issue_source)
    if not scrubbed:
        return None
    impact_line = (impact_en or "").strip() or "Not specified"
    system_prompt = (
        "You are an on-call incident assistant.\n"
        "Output ONLY valid JSON (no markdown fences, no commentary). Keys: issue_en, zh_issue, zh_impact\n"
        "- issue_en: ONE short, direct English sentence stating WHAT is broken: name the specific "
        "game / provider / feature / page / payment method and the failure (error message, stuck, "
        "cannot log in, missing balance, …). If the text states a cause, include it. "
        "Lead with the broken thing — no filler like 'There is an issue where'.\n"
        "  Do NOT describe impact: never write how many players are affected or reporting, "
        "'multiple players are affected', or similar — impact scope is a separate field. "
        "Ignore player-count sentences when writing issue_en.\n"
        "  Good: 'Players cannot launch PG Soft games — stuck on loading screen.' "
        "Good: 'GCash deposits fail with transaction timeout after payment gateway change.' "
        "Bad: 'Multiple players are affected by this issue, with more than 5 players reporting the problem.'\n"
        "  Use only the incident text. No player IDs, ticket IDs, dates, or @names. No bullets.\n"
        "- zh_issue: ONE line Simplified Chinese: translate issue_en faithfully.\n"
        "- zh_impact: ONE line Simplified Chinese: translate the impact scope English line given below. "
        'If that line is "Not specified" or empty, set zh_impact to 未指定.\n'
        "Use 玩家 for game players, not 球员. Keep acronyms (FPMS, CPMS, …) unchanged in Chinese lines.\n"
    )
    user_prompt = (
        f"Incident text:\n{scrubbed}\n\n"
        f"Impact scope (English — translate this exact line to zh_impact):\n{impact_line}\n"
    )
    return system_prompt, user_prompt


def _salvage_overview_fields(raw: str) -> dict:
    """
    Recover ``issue_en`` / ``zh_issue`` / ``zh_impact`` from a truncated or slightly malformed reply
    by regex — each is a complete quoted string even when the JSON is cut off before its closing brace
    (``issue_en`` comes first, so it usually survives). Returns {} if issue_en can't be recovered.
    """
    out: dict = {}
    for key in ("issue_en", "zh_issue", "zh_impact"):
        m = re.search(r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', raw or "")
        if not m:
            continue
        try:
            out[key] = json.loads('"' + m.group(1) + '"')  # unescape \n, \" etc.
        except Exception:
            out[key] = m.group(1)
    return out if str(out.get("issue_en") or "").strip() else {}


def parse_overview_oneshot(raw: str) -> Optional[Tuple[str, str, str]]:
    """Parse ``issue_en`` / ``zh_issue`` / ``zh_impact`` from a model reply; ``None`` on failure."""
    obj = _parse_json_object(raw or "")
    if not obj:
        # Response may be truncated mid-JSON (token limit) — try to salvage the quoted fields.
        obj = _salvage_overview_fields(raw or "")
        if obj:
            log.info("overview one-shot: salvaged fields from truncated/partial JSON")
    if not obj:
        log.warning("overview one-shot: JSON parse failed head=%s", (raw or "")[:200])
        return None
    issue_en = str(obj.get("issue_en") or "").strip()
    zh_issue = str(obj.get("zh_issue") or "").strip()
    zh_impact = str(obj.get("zh_impact") or "").strip()
    if not issue_en:
        return None
    return issue_en, zh_issue, zh_impact


def groq_overview_issue_and_zh_bilingual(issue_source: str, impact_en: str) -> Optional[Tuple[str, str, str]]:
    """
    Single Groq round trip: English issue one-liner + zh_issue + zh_impact.
    Replaces summarize_issue + translate_issue_impact_pair (2–3 HTTP calls → 1).
    Returns None on failure so callers fall back to the legacy path.
    """
    if not GROQ_API_KEY:
        return None
    prompts = build_overview_oneshot_prompts(issue_source, impact_en)
    if not prompts:
        return None
    system_prompt, user_prompt = prompts
    t0 = time.perf_counter()
    try:
        raw = groq_chat_once(system_prompt, user_prompt, max_tokens=1200)
        return parse_overview_oneshot(raw)
    finally:
        perf_log("groq_overview_issue_zh_one_shot", t0)


_DECLARE_REPLY_MAX_CHARS = 320


def _scrub_declare_reply_context(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"(?is)\bAccount:?\s*[\d,\s]+$", "", t).strip()
    t = re.sub(r"\b\d{7,10}\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:600]


def build_issue_watch_declare_reply_prompts(
    *,
    categories_md: str,
    summary: str,
    concern_excerpt: str,
    players_count: int,
    min_reports_threshold: int,
    widespread_impact: bool,
) -> Tuple[str, str]:
    """``(system, user)`` for the declare-on-concern thread reply — provider-agnostic.

    Lives here next to the other prompt builders, but the call itself runs through the Issue Watch
    provider chain (Claude first), so this is not Groq-specific.

    ``min_reports_threshold`` / ``widespread_impact`` are accepted for call compatibility but no
    longer shape the reply: it is one fixed short sentence (issue + affected players), so threshold
    and widespread wording would only pad it.
    """
    cats = (categories_md or "").strip() or "(unspecified)"
    summ = (summary or "").strip() or "(none)"
    concern = _scrub_declare_reply_context(concern_excerpt) or "(none)"
    try:
        n = max(0, int(players_count or 0))
    except (TypeError, ValueError):
        n = 0
    players_clause = (
        f"and {n} affected player{'' if n == 1 else 's'}" if n >= 1 else "(omit the players clause)"
    )
    system_prompt = (
        "You are an on-call ops bot replying in a Lark detection-group thread after duty declares P0.\n"
        "Write EXACTLY ONE short sentence in this shape (plain text, no markdown, no bullets):\n"
        '  "We will declare this issue as P0 due to <short issue phrase> and <N> affected players."\n'
        "Rules:\n"
        "- <short issue phrase> is the symptom in a few words, taken ONLY from the facts given "
        "(e.g. 'GCash deposits failing to credit', 'players unable to log in').\n"
        "- Use the players clause EXACTLY as given in 'Players clause'. If it says to omit it, end the "
        "sentence after the issue phrase.\n"
        "- Nothing else: no greeting, no second sentence, no next steps, no escalation or investigation talk, "
        "no threshold or widespread wording, no counts other than the players clause.\n"
        "STRICT: Do not invent facts. Do not list account IDs. Do not mention AI or automation. "
        "Max ~140 characters.\n"
        "Output ONLY valid JSON: {\"reply\": \"your text\"}"
    )
    user_prompt = (
        f"Category labels:\n{cats}\n\n"
        f"Summary: {summ}\n\n"
        f"Concern (scrubbed): {concern}\n\n"
        f"Players clause: {players_clause}"
    )
    return system_prompt, user_prompt


def parse_issue_watch_declare_reply(raw: str) -> str:
    """``{"reply": "..."}`` → cleaned one/two-sentence text (``""`` when unusable). Any provider."""
    obj = _parse_json_object(raw or "")
    if not obj:
        if (raw or "").strip():
            log.warning("declare reply: JSON parse failed head=%s", (raw or "")[:200])
        return ""
    reply = str(obj.get("reply") or "").strip()
    reply = re.sub(r"\s+", " ", reply).strip(" \"'")
    if not reply:
        return ""
    if len(reply) > _DECLARE_REPLY_MAX_CHARS:
        reply = reply[:_DECLARE_REPLY_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,.;:") + "."
    return reply


def groq_issue_watch_declare_group_reply(
    *,
    categories_md: str,
    summary: str,
    concern_excerpt: str,
    players_count: int,
    min_reports_threshold: int,
    widespread_impact: bool,
) -> Optional[str]:
    """Groq-only declare reply. Kept for direct/legacy callers; the Issue Watch declare path uses
    the provider chain (Claude first) via ``build_issue_watch_declare_reply_prompts``."""
    if not GROQ_API_KEY:
        return None
    system_prompt, user_prompt = build_issue_watch_declare_reply_prompts(
        categories_md=categories_md,
        summary=summary,
        concern_excerpt=concern_excerpt,
        players_count=players_count,
        min_reports_threshold=min_reports_threshold,
        widespread_impact=widespread_impact,
    )
    t0 = time.perf_counter()
    try:
        return parse_issue_watch_declare_reply(groq_chat_once(system_prompt, user_prompt, max_tokens=200)) or None
    finally:
        perf_log("groq_issue_watch_declare_reply", t0)
