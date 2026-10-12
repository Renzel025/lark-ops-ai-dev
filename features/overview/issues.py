"""
Issue summarization and regeneration.

Both calls run through the overview provider chain (``P0_OVERVIEW_AI_PROVIDER``: Claude first,
Groq only as failover), the same chain as the overview one-shot — so one setting decides the model
for every overview-side AI call. With no provider configured they degrade to a regex first-sentence
cut rather than failing.
"""
from __future__ import annotations

import logging
import re

from p0_logic import config as _config
from p0_logic import groq_client as _groq
from p0_logic import text_processing as _text

log = logging.getLogger("lark-ops-ai")

# Issue one-liner max length for cards (was 240; hard slice caused mid-word cuts like "veri" vs "verified").
ISSUE_SUMMARY_MAX_CHARS = 480


def _issue_ai_chat(system_prompt: str, user_prompt: str, *, max_tokens: int) -> str:
    """One round trip through the overview provider chain. ``""`` when every provider fails."""
    from features.overview import overview_ai as _overview_ai

    chain = _config.overview_ai_provider_chain() or (["groq"] if _groq.GROQ_API_KEY else [])
    for provider in chain:
        try:
            out = (
                _overview_ai.run_provider(provider, system_prompt, user_prompt, max_tokens=max_tokens) or ""
            ).strip()
        except Exception as e:  # noqa: BLE001 — any provider error falls through to the next
            log.warning("issues: provider=%s raised %s", provider, e)
            continue
        if out:
            return out
        log.warning("issues: provider=%s returned nothing", provider)
    return ""


def _truncate_issue_output(s: str, max_chars: int) -> str:
    """Cap length without splitting a word — avoids ``…veri`` when the model returns a long sentence."""
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last_space = cut.rfind(" ")
    min_keep = int(max_chars * 0.55)
    if last_space >= min_keep:
        return cut[:last_space].rstrip(" ,.;:")
    return cut.rstrip(" ,.;:")


def summarize_issue(text: str) -> str:
    src = (text or "").strip()
    if not src:
        return "Not specified"
    best_issue_text = _text.issue_context_for_model(src) or src
    scrubbed = re.sub(r"\[Screenshot\s+\d+\s+OCR\]", " ", best_issue_text, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b\d{6,}\b", "<ID>", scrubbed)
    scrubbed = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "<DATE>", scrubbed)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    if not scrubbed:
        return "Not specified"
    system_prompt = (
        "You are an on-call incident assistant.\n"
        "Write ONE concise factual issue sentence for a P0/P1 overview.\n"
        "STRICT RULES:\n"
        "- Use only the provided text.\n"
        "- State directly WHAT is broken: the specific game / provider / feature / page and the failure; "
        "include the cause if the text states one.\n"
        "- Ignore names, @mentions, chat headers, screenshot labels, ticket references, and unrelated thread noise.\n"
        "- Do NOT describe impact (how many players are affected/reporting) — that is a separate field.\n"
        "- Do NOT include player IDs, account IDs, or ticket IDs.\n"
        "- Do NOT include dates or previous incident references.\n"
        "- Do NOT invent anything.\n"
        "- Output exactly one concise sentence only."
    )
    out = _issue_ai_chat(system_prompt, scrubbed, max_tokens=180)
    if not out:
        cut = re.split(r"[.\n]", scrubbed, maxsplit=1)[0].strip()
        return _truncate_issue_output(cut, ISSUE_SUMMARY_MAX_CHARS) if cut else "Not specified"
    out = re.sub(r"\b\d{6,}\b", "", out)
    out = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", out)
    out = re.sub(r"\s+", " ", out).strip(" ,.")
    return _truncate_issue_output(out, ISSUE_SUMMARY_MAX_CHARS) if out else "Not specified"


def regenerate_issue_only(old_issue: str, context_text: str = "") -> str:
    src = (old_issue or "").strip()
    ctx = (context_text or "").strip()
    if not src:
        return "Not specified"
    clean_ctx = re.sub(r"\b\d{6,}\b", "<ID>", ctx)
    clean_ctx = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "<DATE>", clean_ctx)
    clean_ctx = re.sub(r"\s+", " ", clean_ctx).strip()[:1800]
    system_prompt = (
        "You are an on-call incident assistant.\n"
        "Rewrite the issue sentence into a different but equivalent wording.\n"
        "STRICT RULES:\n"
        "- Return exactly ONE sentence.\n"
        "- Keep the same meaning.\n"
        "- Make it noticeably different from the current issue wording.\n"
        "- Focus only on the actual incident symptom.\n"
        "- Do NOT include player counts.\n"
        "- Do NOT include player IDs, account IDs, or ticket IDs.\n"
        "- Do NOT include dates or references to previous incidents.\n"
        "- Do NOT mention support teams.\n"
        "- Do NOT add assumptions.\n"
        "- Output only the rewritten issue sentence."
    )
    user_prompt = (
        f"Current issue sentence:\n{src}\n\n"
        f"Incident context:\n{clean_ctx}\n\n"
        "Rewrite it with different wording while preserving meaning."
    )
    out = _issue_ai_chat(system_prompt, user_prompt, max_tokens=140)
    if not out:
        return src
    out = re.sub(r"\b\d{6,}\b", "", out)
    out = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", out)
    out = re.sub(r"\s+", " ", out).strip(" .,")
    if not out:
        return src
    if out.lower() == src.lower():
        second_prompt = (
            f"Current issue sentence:\n{src}\n\n"
            f"Incident context:\n{clean_ctx}\n\n"
            "Rewrite it again using a clearly different sentence structure. "
            "Do not include counts, IDs, or dates."
        )
        out2 = _issue_ai_chat(system_prompt, second_prompt, max_tokens=140)
        if out2:
            out2 = re.sub(r"\b\d{6,}\b", "", out2)
            out2 = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", out2)
            out2 = re.sub(r"\s+", " ", out2).strip(" .,")
            if out2 and out2.lower() != src.lower():
                return _truncate_issue_output(out2, ISSUE_SUMMARY_MAX_CHARS)
    return _truncate_issue_output(out, ISSUE_SUMMARY_MAX_CHARS)
