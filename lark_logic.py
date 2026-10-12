import os
import re
import time
import secrets
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from wiki_ai_logic import handle_wiki_ai
from p0_logic import text_processing as _text
from p0_logic.config import (
    get_incident_group_chat_ids,
    get_p0_keyword_ai_triage,
    resolve_priority_keyword_ai_provider,
    get_session_meeting_card_post_chat_id,
    get_p0_trigger_ignore_open_ids,
    get_p0_issue_watch_enabled,
    get_p0_keyword_confirm_dm_enabled,
    get_dm_instruction_open_ids,
    p0_group_typed_meeting_commands_enabled,
    HELP_RE,
    RING_CMD_RE,
)
from p0_logic.groq_client import classify_priority_keyword
from features.session.session import notify_p1_mentioned
from p0_logic.cards import build_help_commands_card, build_p0_keyword_confirm_dm_card
from p0_logic.lark_client import (
    post_card_to_chat,
    post_text_to_chat,
    post_card_to_open_id,
    urgent_message_for_users,
)
from features.screenshot.graph_screenshot_request import (
    try_handle_graph_screenshot_request,
    _strip_leading_mentions,
    _mentions_our_bot,
)
from features.issue_watch.issue_watch import try_handle_issue_watch
from p0_logic.config import (
    get_p0_edit_rescan_enabled,
    get_p0_keyword_buzz_enabled,
    get_p0_keyword_lark_urgent_mode,
    get_p0_command_declare_enabled,
    get_p0_command_open_ids,
    parse_p0_declare_command,
)
from p0_logic import (
    start_p0,
    cancel_p0_session,
    end_p0_session,
    clear_p0_cooldown,
    P0_SESSIONS,
    chat_has_active_session,
    handle_dm_generate_overview,
    resolve_source_incident_chat_for_session_command,
)

log = logging.getLogger("lark-ops-ai")

WIKI_GROUP_CHAT_ID = os.getenv("WIKI_GROUP_CHAT_ID", "").strip()

# Lark may deliver ``im.message.receive_v1`` twice for one user send (e.g. main feed + thread copy)
# with different ``message_id`` but the same ``create_time`` and text — dedupe keyword VC triggers.
_KEYWORD_TRIGGER_DEDUPE: Dict[str, float] = {}
_KEYWORD_TRIGGER_DEDUPE_LOCK = threading.Lock()
_KEYWORD_TRIGGER_DEDUPE_TTL_SEC = 25.0


def _keyword_trigger_dedupe_key(
    chat_id: str,
    user_open_id: str,
    message_id: str,
    message_create_time: str,
    text_raw: str,
) -> str:
    norm = " ".join((text_raw or "").split())
    ct = (message_create_time or "").strip()
    if ct:
        return f"kw:{chat_id}:{user_open_id}:{ct}:{norm}"
    mid = (message_id or "").strip()
    return f"kw:{chat_id}:{user_open_id}:mid:{mid}:{norm}"


def _try_consume_keyword_trigger_dedupe(key: str) -> bool:
    """
    True = first delivery for this key within TTL (proceed with P0/P1 keyword action).
    False = duplicate webhook (skip — avoids double ``start_p0`` / double P1 card).
    """
    now = time.monotonic()
    with _KEYWORD_TRIGGER_DEDUPE_LOCK:
        for k, t in list(_KEYWORD_TRIGGER_DEDUPE.items()):
            if now - t > _KEYWORD_TRIGGER_DEDUPE_TTL_SEC:
                del _KEYWORD_TRIGGER_DEDUPE[k]
        if key in _KEYWORD_TRIGGER_DEDUPE:
            return False
        _KEYWORD_TRIGGER_DEDUPE[key] = now
        return True


# --- P0 keyword confirm DM (P0_KEYWORD_CONFIRM_DM_ENABLED) ---------------------------------------
# When a group ``p0`` mention is NOT auto-declared (AI/Groq says it is not a fresh declaration),
# instead of silently dropping it we DM the duty a Yes/No card. The pending entry is keyed by a
# generated ``nonce`` and looked up / consumed when the duty clicks Yes or No (in handlers.py).
_P0_KEYWORD_CONFIRM_LOCK = threading.RLock()
# nonce -> { source_incident_chat_id, trigger_open_id, trigger_lark_user_id, source_chat_name,
#            phrase, created_at (epoch float) }
_P0_KEYWORD_CONFIRM_PENDING: Dict[str, Dict[str, Any]] = {}
_P0_KEYWORD_CONFIRM_TTL_SEC = 3600.0  # prune entries older than 1h
# Dedupe the DM itself: at most one confirm DM per source ``message_id`` (Lark redelivers events).
_P0_KEYWORD_CONFIRM_DM_DEDUPE: Dict[str, float] = {}
_P0_KEYWORD_CONFIRM_DM_DEDUPE_TTL_SEC = 600.0


def _p0_keyword_confirm_prune_locked() -> None:
    now = time.time()
    for k, v in list(_P0_KEYWORD_CONFIRM_PENDING.items()):
        if now - float(v.get("created_at") or 0) > _P0_KEYWORD_CONFIRM_TTL_SEC:
            _P0_KEYWORD_CONFIRM_PENDING.pop(k, None)


def _try_consume_p0_keyword_confirm_dm_dedupe(message_id: str) -> bool:
    """True = first DM for this source message_id; False = duplicate webhook delivery (skip DM)."""
    mid = (message_id or "").strip()
    if not mid:
        return True
    now = time.monotonic()
    with _P0_KEYWORD_CONFIRM_LOCK:
        for k, t in list(_P0_KEYWORD_CONFIRM_DM_DEDUPE.items()):
            if now - t > _P0_KEYWORD_CONFIRM_DM_DEDUPE_TTL_SEC:
                del _P0_KEYWORD_CONFIRM_DM_DEDUPE[k]
        if mid in _P0_KEYWORD_CONFIRM_DM_DEDUPE:
            return False
        _P0_KEYWORD_CONFIRM_DM_DEDUPE[mid] = now
        return True


def p0_keyword_confirm_lookup(nonce: str) -> Optional[Dict[str, Any]]:
    """Return a copy of the pending entry (None if missing/expired). Does NOT consume it."""
    key = (nonce or "").strip()
    if not key:
        return None
    with _P0_KEYWORD_CONFIRM_LOCK:
        _p0_keyword_confirm_prune_locked()
        v = _P0_KEYWORD_CONFIRM_PENDING.get(key)
        return dict(v) if v else None


def p0_keyword_confirm_consume(nonce: str) -> Optional[Dict[str, Any]]:
    """Atomically pop the pending entry (idempotent: a second Yes/No click returns None)."""
    key = (nonce or "").strip()
    if not key:
        return None
    with _P0_KEYWORD_CONFIRM_LOCK:
        _p0_keyword_confirm_prune_locked()
        v = _P0_KEYWORD_CONFIRM_PENDING.pop(key, None)
        return dict(v) if v else None


def _send_p0_keyword_confirm_dm(
    *,
    source_incident_chat_id: str,
    trigger_open_id: str,
    trigger_lark_user_id: str,
    source_chat_name: str,
    phrase: str,
    token: str,
    message_id: str,
) -> None:
    """
    DM the duty a Yes/No "create a P0 meeting?" card for a ``p0`` mention that was NOT auto-declared.
    Stores the pending entry keyed by a fresh nonce BEFORE sending; deduped per source ``message_id``.
    """
    if not _try_consume_p0_keyword_confirm_dm_dedupe(message_id):
        log.info(
            "Incident group: P0 keyword confirm DM skipped (duplicate Lark delivery) chat_id=%s message_id=%s",
            source_incident_chat_id,
            message_id,
        )
        return
    recipients = [x for x in (get_dm_instruction_open_ids() or []) if x]
    if not recipients:
        log.warning(
            "Incident group: P0 keyword confirm DM has no recipients (P0_DM_INSTRUCTION_OPEN_IDS unset) chat_id=%s",
            source_incident_chat_id,
        )
        return
    nonce = secrets.token_hex(16)
    # Resolve the concern NOW (buffer is fresh) so, if the duty clicks "Create meeting", the auto-overview
    # is built from the real issue this "p0" refers to — not the bare "p0". Kept separate from ``phrase``
    # (which still shows the triggering text on the confirm card). Falls back to phrase.
    _concern_text = (phrase or "").strip()
    try:
        from features.overview import concern_context as _concern_ctx

        _concern_text = _concern_ctx.resolve_declaration_concern(
            (source_incident_chat_id or "").strip(), decl_message_id=message_id, decl_text=phrase
        ) or _concern_text
    except Exception as _cc_err:  # noqa: BLE001
        log.warning("concern_context: confirm-DM resolve failed err=%s", _cc_err)
    entry: Dict[str, Any] = {
        "source_incident_chat_id": (source_incident_chat_id or "").strip(),
        "trigger_open_id": (trigger_open_id or "").strip(),
        "trigger_lark_user_id": (trigger_lark_user_id or "").strip(),
        "source_chat_name": (source_chat_name or "").strip(),
        "phrase": (phrase or "").strip()[:300],
        "concern_text": _concern_text.strip()[:1200],
        "source_message_id": (message_id or "").strip(),
        "created_at": time.time(),
    }
    with _P0_KEYWORD_CONFIRM_LOCK:
        _p0_keyword_confirm_prune_locked()
        _P0_KEYWORD_CONFIRM_PENDING[nonce] = entry
    card = build_p0_keyword_confirm_dm_card(
        nonce,
        entry["phrase"],
        entry["source_chat_name"],
        source_incident_chat_id=entry["source_incident_chat_id"],
        source_message_id=entry["source_message_id"],
    )
    # The card carries no buttons by default, so the buzz IS the page — without it a "p0" mention
    # would sit unread in a DM. Same 加急 path the Major-P0 alert already uses.
    buzz_on = get_p0_keyword_buzz_enabled()
    urgent_mode = get_p0_keyword_lark_urgent_mode()
    tails: List[str] = []
    buzzed = 0
    for oid in recipients:
        st, body, mid = post_card_to_open_id(oid, token, card)
        tails.append(oid[-8:] if len(oid) > 8 else oid)
        if st != 200:
            log.warning(
                "Incident group: P0 keyword confirm DM post HTTP=%s oid_tail=%s body=%r",
                st,
                oid[-8:] if len(oid) > 8 else oid,
                (body or "")[:200],
            )
            continue
        if buzz_on and urgent_mode != "off" and mid:
            uok, udetail = urgent_message_for_users(token, mid, [oid], mode=urgent_mode)
            if uok:
                buzzed += 1
            else:
                log.warning(
                    "Incident group: P0 keyword buzz urgent_%s failed oid_tail=%s detail=%s "
                    "(enable im:message.urgent on the bot app?)",
                    urgent_mode,
                    oid[-8:] if len(oid) > 8 else oid,
                    (udetail or "")[:300],
                )
    log.info(
        "Incident group: P0 keyword confirm DM sent recipients=%s nonce=%s chat_id=%s "
        "buzz_enabled=%s urgent_%s=%s",
        tails,
        nonce,
        source_incident_chat_id,
        buzz_on,
        urgent_mode,
        buzzed,
    )


def _handle_p0_declare_command(
    *,
    priority: str,
    chat_id: str,
    notify_chat: str,
    token: str,
    user_id: str,
    sender_lark_user_id: str,
    source_chat_name: str,
    text_raw: str,
    message_id: str,
    message_create_time: str,
) -> None:
    """
    ``/p0`` / ``/p1`` typed in an incident group — the only path allowed to create a meeting.

    Restricted to the OM duty accounts in ``P0_COMMAND_OPEN_IDS`` (falls back to the duty DM list).
    An unresolvable allowlist lets the command through rather than bricking declaration outright.
    """
    slash = "/" + priority.lower()
    allowed = get_p0_command_open_ids()
    if not allowed:
        log.warning(
            "Incident group: %s allowlist is EMPTY (P0_COMMAND_OPEN_IDS and P0_DM_INSTRUCTION_OPEN_IDS "
            "both unset) — letting the command through chat_id=%s",
            slash,
            chat_id,
        )
    elif (user_id or "").strip() not in allowed:
        log.info(
            "Incident group: %s refused (not OM duty) chat_id=%s user_tail=%s",
            slash,
            chat_id,
            (user_id or "")[-8:],
        )
        if token:
            post_text_to_chat(
                notify_chat,
                token,
                "⚠️ Only OM duty can declare. Ask OM duty to type **{}** in this group.".format(slash),
            )
        return

    kw_dedupe = _keyword_trigger_dedupe_key(
        chat_id, user_id, message_id, message_create_time, text_raw
    )
    if not _try_consume_keyword_trigger_dedupe(kw_dedupe):
        log.info("Incident group: %s skipped (duplicate Lark delivery) chat_id=%s", slash, chat_id)
        return

    # The command is just the trigger. The overview is built from the issues discussed above
    # (reply-parent, then AI-pick, then most-recent) — same as the old typed "p0".
    concern = ""
    try:
        from features.overview import concern_context as _concern_ctx

        concern = _concern_ctx.resolve_declaration_concern(
            chat_id, decl_message_id=message_id, decl_text=""
        )
    except Exception as _cc_err:  # noqa: BLE001
        log.warning("concern_context: %s resolve failed chat_id=%s err=%s", slash, chat_id, _cc_err)

    log.info(
        "Incident group: %s declared by OM duty chat_id=%s user_tail=%s concern=%r",
        slash,
        chat_id,
        (user_id or "")[-8:],
        (concern or "")[:160],
    )
    start_p0(
        chat_id,
        token,
        user_id,
        priority=priority,
        source_chat_name=source_chat_name,
        trigger_lark_user_id=sender_lark_user_id,
        declaration_text=concern,
        via_command=True,
    )


def _maybe_p0_keyword_confirm_dm(
    *,
    chat_id: str,
    token: str,
    user_id: str,
    sender_lark_user_id: str,
    source_chat_name: str,
    text_raw: str,
    message_id: str,
) -> None:
    """
    Gate + guards for the not-auto-declared P0 mention DM. No-op unless the toggle is ON;
    skipped when a meeting is already active in the source group.
    """
    if not get_p0_keyword_confirm_dm_enabled():
        return
    # Rule: anything that is NOT a firm auto-declaration must still ASK the duty — questions,
    # hedges ("possible p0"), handoffs, and past/informational references ("the p0 yesterday",
    # "may we ask ... p0") all get the Yes/No DM. Only an EXPLICIT negation ("this is not p0")
    # stays silent, since asking "create a meeting?" right after "this is NOT p0" is nonsensical.
    if _is_explicit_p0_negation(text_raw):
        log.info(
            "Incident group: P0 keyword confirm DM skipped (explicit negation) chat_id=%s",
            chat_id,
        )
        return
    if chat_has_active_session(chat_id):
        log.info(
            "Incident group: P0 keyword confirm DM skipped (session already active) chat_id=%s",
            chat_id,
        )
        return
    _send_p0_keyword_confirm_dm(
        source_incident_chat_id=chat_id,
        trigger_open_id=user_id,
        trigger_lark_user_id=sender_lark_user_id,
        source_chat_name=source_chat_name,
        phrase=text_raw,
        token=token,
        message_id=message_id,
    )


# Keyword anywhere in the sentence (e.g. "this is p0", "we tag this as a P0", "is this p0?") —
# case-insensitive, no declare-vs-question distinction: every hit just notifies duty.
# NOTE: \b is useless here — Python counts CJK as word characters, so "这属于p0" has no
# boundary before the "p" and never matched. Bound on ASCII alphanumerics/underscore instead
# so "p0" glued to Chinese text is seen, while "ap0"/"p00"/"p0_issue" stay ignored as before.
P0_KEYWORD_RE = re.compile(r"(?<![0-9A-Za-z_])(?:p0|priority\s*0)(?![0-9A-Za-z_])", re.IGNORECASE)
P1_KEYWORD_RE = re.compile(r"(?<![0-9A-Za-z_])(?:p1|priority\s*1)(?![0-9A-Za-z_])", re.IGNORECASE)

# Shared subject: "it", "this", "this one", "this issue", "that outage", …
_P0_SUBJECT = r"(?:it|(?:this|that)(?:\s+(?:one|issue|incident|outage|problem|ticket|case))?)"
_PRIO01 = r"(?:p0|p1|priority\s*0|priority\s*1)"
_P0_ONLY = r"(?:p0|priority\s*0)"


def _is_pasted_meeting_invite_footer(text: str) -> bool:
    """
    Ignore copy-paste of the red meeting-card footer (starts with ``P0 declared -`` / ``P1 declared -``)
    so it does not start another VC.
    """
    t = (text or "").strip().lower()
    return t.startswith("p0 declared - created a meeting") or t.startswith("p1 declared - created a meeting")


# Lark mobile/desktop composer may append ``Message <chat display name>`` to im.message text.
# e.g. deposit paste + ``Message p0 detection dev`` falsely matches ``\\bp0\\b`` keyword trigger.
# Footer may be on its own line OR same line as the last account id (no leading newline).
_LARK_COMPOSER_MESSAGE_FOOTER_RE = re.compile(r"(?is)\s*Message\s+.+?\s*$")


def _strip_lark_composer_message_footer(text: str, *, chat_label: str = "") -> str:
    """Remove trailing ``Message <group name>`` Lark UI suffix before keyword / triage."""
    t = (text or "").strip()
    label = (chat_label or "").strip()
    if label:
        labeled = re.compile(rf"(?is)\s*Message\s+{re.escape(label)}\s*$")
        t = labeled.sub("", t).rstrip()
    m = _LARK_COMPOSER_MESSAGE_FOOTER_RE.search(t)
    if m:
        t = t[: m.start()].rstrip()
    return t


def _text_for_priority_keyword_trigger(text: str, *, chat_label: str = "") -> str:
    return _strip_lark_composer_message_footer(text, chat_label=chat_label)


# Moved to p0_logic/text_processing.py as is_manual_p0_incident_overview_template() — Issue Watch
# needs the same detector and importing lark_logic.py from a features/ module would invert the
# dependency direction (features should not reach up into the top-level router).


# Incident keyword: message contains ``p0`` but explicitly says no / not P0 / no escalation (no Groq).
# Interrogative: a trailing "?" or an opening question word (after any @mentions). Used to stop a
# question being mistaken for a denial.
_P0_QUESTION_RE = re.compile(
    r"(?is)(?:\?\s*$)"
    r"|^\W*(?:@[\w_]+\s*)*"
    r"(?:is|isn'?t|are|aren'?t|was|were|do|does|did|should|shouldn'?t|can|can'?t|could|"
    r"why|what|how|any\s+chance)\b"
)

_P0_NEGATION_SUBSTRINGS = (
    "no p0 escalation",
    "not a p0",
    "not p0",
    "is not p0",
    "is not a p0",
    "not in p0",
    "non-p0",
    "non p0",
    "without p0 escalation",
    "without a p0",
    "without p0",
    "p0 escalation is not required",
    "p0 escalation is not needed",
    "no need for p0",
    "no need for a p0",
    "no p0 needed",
    "no p0 required",
)


def _is_explicit_p0_negation(text: str) -> bool:
    """
    True when the text contains ``p0`` but clearly *declines* P0 / escalation (e.g. "No P0 escalation is required").
    Skips the naive ``\\bp0\\b`` keyword VC trigger — no LLM required.
    """
    t = (text or "").strip()
    if not t:
        return False
    if not P0_KEYWORD_RE.search(t):
        return False
    # A QUESTION is never a negation. "Is this not P0 ?" asks whether it SHOULD be one — the
    # opposite of declining it — but it contains the substring "not p0" and was being swallowed
    # silently, so duty was never asked. Under the routing policy only an explicit negation is
    # silent; everything else, questions included, must reach the duty confirm-DM.
    if _P0_QUESTION_RE.search(t):
        return False
    low = " ".join(t.lower().split())
    for s in _P0_NEGATION_SUBSTRINGS:
        if s in low:
            return True
    if re.search(r"(?is)\bno\s+p0\s+escalation\b", t):
        return True
    if re.search(r"(?is)\b(?:not|without)\s+(?:a\s+)?p0\b", t):
        return True
    # "will not be consider(ed) as p0" — ``not`` and ``p0`` are not adjacent (old pattern missed this).
    if re.search(
        r"(?is)\b(?:will|would)\s+not\s+(?:be\s+)?consider\w*(?:\s+\w+){0,3}\s+as\s+(?:a\s+)?p0\b",
        t,
    ):
        return True
    if re.search(r"(?is)\bnot\s+consider\w*(?:\s+\w+){0,3}\s+as\s+(?:a\s+)?p0\b", t):
        return True
    if re.search(
        r"(?is)\b(?:does|do|did)\s+not\s+(?:qualify|count)\s+(?:as\s+)?(?:a\s+)?p0\b",
        t,
    ):
        return True
    return False


def _regex_priority_keyword_intent_override(text_raw: str) -> Optional[str]:
    """
    Deterministic declare vs question when phrasing clearly matches regex (overrides Groq mislabels).
    Returns ``declare_p0``, ``question``, or None (defer to Groq).
    """
    t = (text_raw or "").strip()
    if not t or not P0_KEYWORD_RE.search(t):
        return None
    if _is_explicit_p0_negation(t):
        return "question"
    if _is_p0_thread_confirm_question(t):
        return "question"
    if _is_p0_conditional_or_confirm_question(t):
        return "question"
    if _is_explicit_direct_p0_declaration(t):
        return "declare_p0"
    return None


def _sanitize_priority_keyword_ai_triage(text_raw: str, result: Dict[str, str]) -> Dict[str, str]:
    """Correct common Groq mislabels when only P0 or only P1 appears in the text."""
    override = _regex_priority_keyword_intent_override(text_raw)
    if override:
        groq_intent = str((result or {}).get("intent") or "").strip().lower()
        if groq_intent != override:
            log.info(
                "Priority keyword AI triage: regex override groq=%s -> %s text_head=%r",
                groq_intent or "(none)",
                override,
                (text_raw or "")[:200],
            )
        return {
            **(result or {}),
            "intent": override,
            "reason": f"regex_override:{override}",
            "provider": (result or {}).get("provider") or "regex",
        }
    intent = str(result.get("intent") or "").strip().lower()
    has_p0 = bool(P0_KEYWORD_RE.search(text_raw or ""))
    has_p1 = bool(P1_KEYWORD_RE.search(text_raw or ""))
    if has_p0 and not has_p1 and intent == "declare_p1":
        log.warning(
            "Priority keyword AI triage: Groq intent=declare_p1 but text is P0-only — remapping to question text_head=%r",
            (text_raw or "")[:200],
        )
        return {**result, "intent": "question", "reason": "P0-only text; declare_p1 mislabel"}
    if has_p1 and not has_p0 and intent == "declare_p0":
        log.warning(
            "Priority keyword AI triage: Groq intent=declare_p0 but text is P1-only — remapping to question text_head=%r",
            (text_raw or "")[:200],
        )
        return {**result, "intent": "question", "reason": "P1-only text; declare_p0 mislabel"}
    return result


def _priority_keyword_ai_triage(text_raw: str, groq_key: str) -> Optional[Dict[str, str]]:
    """Run LLM classifier (Claude → Gemini → Groq) when ``P0_KEYWORD_AI_TRIAGE`` is on."""
    if not get_p0_keyword_ai_triage():
        return None
    override = _regex_priority_keyword_intent_override(text_raw)
    if override:
        result = {
            "intent": override,
            "reason": f"regex_override:{override}",
            "provider": "regex",
        }
        log.info(
            "Priority keyword AI triage provider=regex intent=%s reason=%r text_head=%r",
            override,
            result["reason"],
            (text_raw or "")[:200],
        )
        return result
    if not resolve_priority_keyword_ai_provider():
        return None
    try:
        result = classify_priority_keyword(text_raw, provider=None)
        if result:
            result = _sanitize_priority_keyword_ai_triage(text_raw, result)
            log.info(
                "Priority keyword AI triage provider=%s intent=%s reason=%r text_head=%r",
                result.get("provider") or "-",
                result.get("intent"),
                result.get("reason"),
                (text_raw or "")[:200],
            )
        else:
            override = _regex_priority_keyword_intent_override(text_raw)
            if override:
                result = {
                    "intent": override,
                    "reason": f"regex_override:{override}",
                    "provider": "regex",
                }
                log.info(
                    "Priority keyword AI triage: LLM empty — regex override intent=%s text_head=%r",
                    override,
                    (text_raw or "")[:200],
                )
            else:
                log.info(
                    "Priority keyword AI triage: no usable Groq result (legacy/GROQ_GATE path may apply) text_head=%r",
                    (text_raw or "")[:200],
                )
        return result
    except Exception as e:
        log.warning("Priority keyword AI triage failed: %s", e)
        return None


def _is_explicit_direct_p0_declaration(text: str) -> bool:
    """
    Short, unmistakable **declarations** (not questions). Groq gate often false-negatives these.

    Examples: "this is p0", "it's p0", whole-line "p0", "we tag this issue as p0".
    """
    t = (text or "").strip()
    if not t or not P0_KEYWORD_RE.search(t):
        return False
    if _is_explicit_p0_negation(t):
        return False
    # A conditional / confirmation QUESTION that merely contains "this is p0" is NOT a declaration —
    # e.g. "may we confirm if this is p0?". Defer to the confirm path (ask first), never hard-declare.
    if _is_p0_conditional_or_confirm_question(t):
        return False
    if re.search(r"(?is)\bthis\s+is\s+(?:a\s+)?(?:p0|priority\s*0)\b", t):
        return True
    if re.search(r"(?is)\bthat\s+is\s+(?:a\s+)?(?:p0|priority\s*0)\b", t):
        return True
    if re.search(r"(?is)\bit(?:'s|\s+is)\s+(?:a\s+)?(?:p0|priority\s*0)\b", t):
        return True
    # "yes team this issue is p0" — Groq often misses; not "is this issue p0?" (question path).
    if re.search(
        r"(?is)\b(?:this|that|the)\s+issue\s+is\s+(?:a\s+)?(?:p0|priority\s*0)\b",
        t,
    ):
        return True
    if re.match(r"(?is)^(?:p0|priority\s*0)\s*[!?.…]*\s*$", t):
        return True
    # "we tag … as p0" — modal + we + tag stays with Groq / thread-confirm ("can we tag…").
    if not re.search(
        r"(?is)\b(?:can|could|should|may|would|shall)\s+we\s+"
        r"(?:tag|treat|consider|declare|escalate|raise|elevate)",
        t,
    ):
        # Declaration verbs: "we declare this as p0", "declare this issue as p0", "escalate to p0",
        # "raise this to p0", "elevate to p0" (and -ing/-ed forms). Modal questions ("should we
        # escalate to p0?") are excluded by the guard above.
        if re.search(
            r"(?is)\b(?:(?:i|we|let's|lets)\s+)?(?:hereby\s+)?(?:will\s+)?(?:are\s+)?"
            r"(?:declar(?:e|ed|es|ing)|escalat(?:e|ed|es|ing)|rais(?:e|ed|es|ing)|elevat(?:e|ed|es|ing))\s+"
            r"(?:this|that|it|the\s+issue|this\s+issue)?\s*(?:as\s+|to\s+)?(?:a\s+)?(?:p0|priority\s*0)\b",
            t,
        ):
            return True
        if re.search(
            rf"(?is)\b(?:i|we)\s+(?:will\s+)?consider(?:ed|ing)?\s+{_P0_SUBJECT}\s+(?:as\s+)?(?:a\s+)?(?:p0|priority\s*0)\b",
            t,
        ):
            return True
        if re.search(
            rf"(?is)\b(?:confirm|confirmed)\s+(?:as\s+)?(?:a\s+)?(?:p0|priority\s*0)\b",
            t,
        ):
            return True
        if re.search(
            r"(?is)\b(?:i|we)\s+treat(?:ed|ing)?\s+(?:this|that|it|the\s+issue|this\s+issue)\s+as\s+(?:a\s+)?(?:p0|priority\s*0)\b",
            t,
        ):
            return True
        if re.search(
            r"(?is)\b(?:i|we)\s+tag(?:ged|ging)?\s+(?:this|that|it|the\s+issue|this\s+issue)\s+as\s+(?:a\s+)?(?:p0|priority\s*0)\b",
            t,
        ):
            return True
        # Passive / subjectless tag: "this issue tagged as p0", "tagged as p0", "marked as p0"
        # (no I/we subject). These are real declarations the LLM triage keeps mislabeling as
        # "handoff" — make them deterministic so a declaration never depends on the classifier.
        # Modal questions ("should we tag … as p0") are already excluded by the guard above.
        if re.search(
            r"(?is)\b(?:tagged|treated|marked|flagged|classified|labell?ed)\s+as\s+(?:a\s+)?(?:p0|priority\s*0)\b",
            t,
        ):
            return True
    return False


# Cancel commands: optional free-text reason after the phrase (e.g. "cancel meeting no need yet")
# Order: longer prefixes first so "cancel meeting" wins over "cancel".
# The keyword must be a WHOLE word — the ``(?=\s|$)`` lookahead stops arbitrary letters gluing onto
# the short ``cm`` / ``cancel`` triggers (e.g. a screenshot caption ``cmsdb`` must NOT cancel the P0
# meeting; only ``cm``, ``cm <reason>``, ``cancel``, ``cancel meeting <reason>`` should).
CANCEL_WITH_OPTIONAL_REASON_RE = re.compile(
    r"^\s*(cancel\s+meeting|cancel\s+p0|cancel\s+p1|cancel|cm)(?=\s|$)\s*(.*)$",
    re.IGNORECASE,
)


def _matches_typed_end_meeting_command(text_raw: str) -> bool:
    """Operator guide: whole-line **end meeting**; **p0 end** / **end p0** / … may appear in prose."""
    t = (text_raw or "").strip()
    if not t:
        return False
    if re.match(r"(?is)^\s*(?:em|end\s+meeting|close\s+meeting)\s*$", t):
        return True
    if re.match(r"(?is)^\s*(?:pe|p0e)\s*$", t):
        return True
    if re.match(r"(?is)^\s*(?:p1e|1e)\s*$", t):
        return True
    if re.search(r"(?is)\b(?:p0|p1)\s+end\b", t):
        return True
    if re.search(r"(?is)\bend\s+(?:p0|p1)\b", t):
        return True
    if re.search(r"(?is)\bclose\s+p0\b", t):
        return True
    if re.search(r"(?is)\bp0\s+resolved\b", t):
        return True
    return False


# Clear cooldown only (no new VC). Whole line only. / 仅清除冷却，不新建会议
COOLDOWN_RESET_RE = re.compile(
    r"^\s*(p0\s+cooldown\s+reset|cooldown\s+reset|reset\s+cooldown|clear\s+cooldown|cr)\s*$",
    re.IGNORECASE,
)


# Detects a *question* about P0 ("is this p0", "can we tag this as p0") vs a declaration.
# Used only by _is_p0_thread_confirm_question to keep questions from auto-declaring.
# (The old designated-asker thread-confirm flow was removed.)
# ``?`` optional (not only questions with ``?``); phrase may follow @mentions ("@QA is this P0?").
P0_THREAD_CONFIRM_QUESTION_RE = re.compile(
    rf"(?is)(?:"
    rf"is\s+this\s+(?:an?\s+)?{_P0_ONLY}\b|"
    rf"is\s+that\s+(?:an?\s+)?{_P0_ONLY}\b|"
    rf"is\s+it\s+(?:an?\s+)?{_P0_ONLY}\b|"
    rf"is\s+this\s+[^\n?]+\s+is\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"is\s+this\s+[^\n?]+\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"if\s+this\s+[^\n?]+\s+is\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"if\s+that\s+[^\n?]+\s+is\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"is\s+this\s+[^\n?]+\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"is\s+that\s+[^\n?]+\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"can\s+we\s+tag\s+(?:{_P0_SUBJECT}\s+)?as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"could\s+we\s+tag\s+(?:{_P0_SUBJECT}\s+)?as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"shall\s+we\s+tag\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"can\s+this\s+be\s+tagged\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"can\s+that\s+be\s+tagged\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"can\s+we\s+refer\s+(?:{_P0_SUBJECT}\s+)?as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"could\s+we\s+refer\s+(?:{_P0_SUBJECT}\s+)?as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"is\s+this\s+possible\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"is\s+that\s+possible\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"should\s+we\s+declare\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"should\s+i\s+declare\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"can\s+we\s+declare\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"could\s+we\s+declare\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"can\s+we\s+consider\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"could\s+we\s+consider\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b|"
    rf"shall\s+we\s+consider\s+{_P0_SUBJECT}\s+as\s+(?:a\s+)?{_P0_ONLY}\b"
    rf")"
)


def _is_p0_thread_confirm_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(P0_THREAD_CONFIRM_QUESTION_RE.search(t))


def _is_p0_conditional_or_confirm_question(text: str) -> bool:
    """A P0 keyword wrapped in a **conditional / confirmation question**, e.g.
    "may we confirm if this is p0?", "checking if this is p0", "let's verify whether it's a p0".

    The literal ``this/that/it is p0`` substring must NOT hard-declare a meeting when it sits under an
    ``if``/``whether`` clause or a ``confirm``/``check``/``verify`` question — those are asking, not
    declaring. Routes to the duty confirm-DM (ask first), never auto-start.
    """
    t = (text or "").strip()
    if not t:
        return False
    # "... if/whether (this|that|it) is (a) p0" — the p0 phrase is the object of a conditional.
    if re.search(
        r"(?is)\b(?:if|whether)\s+(?:this|that|it)\s+(?:is|'s)\s+(?:a\s+)?(?:p0|priority\s*0)\b",
        t,
    ):
        return True
    # A "confirm/check/verify/clarify … if/whether …" question that mentions a P0 keyword anywhere.
    # (Guards against "confirmed as p0" declarations — those have no if/whether.)
    if (
        re.search(r"(?is)\b(?:confirm|confirming|check|checking|verify|verifying|clarify|clarifying)\b", t)
        and re.search(r"(?is)\b(?:if|whether)\b", t)
        and P0_KEYWORD_RE.search(t)
    ):
        return True
    return False




def _clean_mention_names(raw_mentions: Any) -> List[str]:
    out: List[str] = []
    if not raw_mentions:
        return out

    if isinstance(raw_mentions, list):
        for m in raw_mentions:
            if isinstance(m, str):
                name = m.strip()
                if name:
                    out.append(name)
                continue

            if isinstance(m, dict):
                name = (
                    m.get("name")
                    or m.get("title")
                    or m.get("display_name")
                    or m.get("full_name")
                    or ""
                )
                name = str(name).strip()
                if name:
                    out.append(name)

    seen = set()
    deduped: List[str] = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(x)

    return deduped


def _get_active_session_chat_id() -> str:
    if not P0_SESSIONS:
        return ""
    return list(P0_SESSIONS.keys())[-1]


def _parse_mixed_commands(ring_raw: str) -> Tuple[List[str], List[str]]:
    """Ordered, deduped commands from ONE message, allowing a MIX in a single line — e.g.
    ``/srebac sfpms cpms`` or ``/scpms /fpms /c @Juan``. Returns ``(game_cmds, ring_cmds)``: game_cmds
    are SRE-game escalation commands (srebac …); ring_cmds are duty/direct ring commands (scpms, fpms,
    dba, c, …), fired into the active meeting.

    Only ONE leading slash is needed — bare (slashless) command tokens are accepted, BUT only when the
    whole message is a PURE command list (every token is a known command). If ANY token is prose
    (e.g. "@bot pms is down"), only slash-prefixed tokens count, so casual chat can never page.
    (@mentions are already stripped from ``ring_raw``; tagged targets come from mention_open_ids.)
    """
    from features.recording.sre_game import is_sre_game_command, is_po_game_command

    # Drop @mention tokens (the /c targets, e.g. "@Name" / "@_user_2") — they're captured separately
    # via mention_open_ids, so they must NOT count as "prose" (which would force a slash on every
    # command). This lets "/cpms fpms /c @A @B" work with a single leading slash.
    toks = [t for t in (ring_raw or "").split() if t and not t.startswith("@")]

    def _cmd(tok: str) -> str:
        return tok.lstrip("/").strip().lower()

    def _known(c: str) -> bool:
        return bool(is_sre_game_command(c) or is_po_game_command(c) or c == "c" or RING_CMD_RE.match(c))

    has_prose = any(not _known(_cmd(t)) for t in toks)
    game: List[str] = []
    ring: List[str] = []
    gseen: set = set()
    rseen: set = set()
    for tok in toks:
        if has_prose and not tok.startswith("/"):
            continue  # with prose present, require an explicit slash per command
        c = _cmd(tok)
        if not c:
            continue
        if is_sre_game_command(c) or is_po_game_command(c):
            if c not in gseen:
                gseen.add(c)
                game.append(c)
        elif (c == "c" or RING_CMD_RE.match(c)) and c not in rseen:
            rseen.add(c)
            ring.append(c)
    return game, ring


def process_message_edit(
    chat_id: str,
    message_id: str,
    sender_open_id: str,
    token: str,
    text: str = "",
    *,
    source_chat_name: str = "",
    message_create_time: str = "",
    mention_open_ids: Optional[List[str]] = None,
) -> None:
    """Re-run **Issue Watch detection** on an edited message (``im.message.updated_v1``).

    Why only detection: OM commonly posts a report and then EDITS it to append the affected player
    IDs. The bot classified the original (0 players) and a major P0 needs 4+, so the alert that
    matters would never fire. Re-classifying the edited text closes that hole.

    Why NOT the declaration routing: re-running the full router would let an edit fire typed commands
    (end meeting, ring commands, wiki Q&A) a second time. Declaring stays a deliberate act — type
    the keyword in a new message.
    """
    cid = (chat_id or "").strip()
    mid = (message_id or "").strip()
    body = (text or "").strip()
    if not cid or not mid or not body:
        return
    if not get_p0_edit_rescan_enabled():
        return
    if cid not in get_incident_group_chat_ids():
        return
    log.info(
        "message edit: re-running issue watch chat_id=%s mid_tail=%s text_head=%r",
        cid,
        mid[-12:],
        body[:100],
    )
    try:
        from features.issue_watch.issue_watch import cancel_deferred_alert_for_sender

        # A deferred alert from the ORIGINAL text is now stale — the edit is the better version.
        cancel_deferred_alert_for_sender(cid, (sender_open_id or "").strip())
    except Exception as e:  # noqa: BLE001
        log.warning("message edit: could not cancel deferred alert: %s", e)
    try:
        try_handle_issue_watch(
            _strip_lark_composer_message_footer(body, chat_label=source_chat_name),
            cid,
            (sender_open_id or "").strip(),
            token,
            source_chat_name=source_chat_name,
            message_id=mid,
            message_create_time=message_create_time,
            mention_open_ids=mention_open_ids or [],
        )
    except Exception as e:  # noqa: BLE001
        log.error("message edit: issue watch failed chat_id=%s err=%s", cid, e, exc_info=True)


def process_message(
    incoming_text: str,
    chat_id: str,
    user_id: str,
    token: str,
    lark_client: Any,
    groq_key: str,
    source_chat_name: str = "",
    **kwargs: Any,
) -> None:
    text_raw = (incoming_text or "").strip()

    message_type = str(kwargs.get("message_type") or "").strip().lower()
    message_id = str(kwargs.get("message_id") or "").strip()
    message_create_time = str(kwargs.get("message_create_time") or "").strip()
    parent_id = str(kwargs.get("parent_id") or "").strip()
    root_id = str(kwargs.get("root_id") or "").strip()
    thread_id = str(kwargs.get("thread_id") or "").strip()
    image_key = str(kwargs.get("image_key") or kwargs.get("file_key") or "").strip()
    mention_names = _clean_mention_names(kwargs.get("mention_names") or kwargs.get("mentions"))
    tenant_token = str(kwargs.get("tenant_token") or token or "").strip()
    chat_type = str(kwargs.get("chat_type") or "").strip().lower()
    sender_lark_user_id = str(kwargs.get("sender_lark_user_id") or "").strip()

    active_chat_id = _get_active_session_chat_id()
    has_active_session = bool(active_chat_id)

    log.info(
        "process_message route: chat_id=%s chat_type=%s user_id=%s message_type=%s text=%r mentions=%s active_chat_id=%s has_active_session=%s",
        chat_id,
        chat_type,
        user_id,
        message_type,
        text_raw[:200] if text_raw else "",
        mention_names,
        active_chat_id,
        has_active_session,
    )

    # ---------------------------------------------------------
    # INCIDENT GROUP + prompt/mirror session commands
    # ---------------------------------------------------------
    is_detection = chat_id in get_incident_group_chat_ids()
    mirror_session_source = (
        "" if is_detection else resolve_source_incident_chat_for_session_command(chat_id)
    )
    if is_detection or mirror_session_source:
        if not text_raw:
            return

        if is_detection:
            session_source = chat_id
        else:
            session_source = mirror_session_source

        # Buffer the message so a typed "p0" can later resolve WHICH concern to build the overview from
        # (reply-parent / AI-pick / most-recent). Cheap; records into the incident group's rolling buffer.
        # "post" too: real concern reports are usually rich text (text + screenshots), not plain text.
        if is_detection and message_type in ("text", "post"):
            try:
                from features.overview import concern_context as _concern_ctx

                _concern_ctx.record_group_message(
                    chat_id,
                    message_id=message_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    sender_open_id=user_id,
                    text=text_raw,
                    ts=float(message_create_time) / 1000.0 if message_create_time.isdigit() else None,
                )
            except Exception as _cc_err:  # noqa: BLE001
                log.warning("concern_context: record failed chat_id=%s err=%s", chat_id, _cc_err)

        # Bot replies from typed commands: same destination as meeting cards for this incident row.
        notify_chat = get_session_meeting_card_post_chat_id(session_source) or chat_id

        if HELP_RE.match(text_raw.strip()):
            if token:
                st, body, _ = post_card_to_chat(notify_chat, token, build_help_commands_card())
                if st != 200:
                    log.warning("incident group help card failed HTTP=%s body=%s", st, (body or "")[:300])
            return

        # /p0 and /p1 — the ONLY commands that create a meeting. Checked before every keyword
        # heuristic so a declaration is a deliberate, unambiguous act instead of a phrase the AI
        # had to judge. Everything else (prose "p0", Issue Watch, card confirms) now only notifies.
        if is_detection and get_p0_command_declare_enabled():
            _cmd_pri = parse_p0_declare_command(
                _strip_leading_mentions(text_raw, mention_names).strip()
            )
            if _cmd_pri:
                _handle_p0_declare_command(
                    priority=_cmd_pri,
                    chat_id=chat_id,
                    notify_chat=notify_chat,
                    token=token,
                    user_id=user_id,
                    sender_lark_user_id=sender_lark_user_id,
                    source_chat_name=source_chat_name,
                    text_raw=text_raw,
                    message_id=message_id,
                    message_create_time=message_create_time,
                )
                return

        # Ring commands page duty/escalation into the already-active meeting. They REQUIRE a leading
        # slash (/m /e /fe /fpms /pms /scpms /sfpms …); @mentioning the bot alone is NOT enough, so a
        # bare "sfpms" / stray letter in normal chat (even with @bot) can never page. Checked before
        # the screenshot handler so these short commands aren't swallowed.
        _ring_raw = _strip_leading_mentions(text_raw, mention_names).strip()
        _ring_is_slash = _ring_raw.startswith("/")
        _ring_cmd = _ring_raw.lstrip("/").strip().lower()
        _ring_first = _ring_cmd.split()[0] if _ring_cmd.split() else ""

        # SRE Game escalation. First: a no/yes REPLY inside an active escalation thread advances/stops
        # it (scoped — a non-yes/no reply is ignored here and falls through to normal routing, so this
        # never swallows unrelated chatter). Then: a /sre<game> command starts a new escalation.
        _esc_keys = [k for k in (root_id, parent_id, thread_id) if k]
        if _esc_keys:
            from features.recording.sre_game import maybe_handle_sre_game_reply

            # Pass ALL thread identifiers (root/parent/thread) + the mention-stripped text so "@bot /n"
            # and a bare "/n" both parse to "n" and match whichever key the escalation was stored under.
            if maybe_handle_sre_game_reply(
                _esc_keys,
                _ring_raw,
                token,
                tenant_token=tenant_token or token,
                operator_open_id=user_id,
                tagged_open_ids=[x for x in (kwargs.get("mention_open_ids") or []) if x],
            ):
                return
        # /segame <game> — EGAME escalation: ring the 1st contact handling that EXACT e-game (e.g.
        # "/segame Bakunawa", "/segame Bakunawa 2", "/segame Color Land"). Takes the REST of the line as
        # the game name (may be multi-word), so it's handled here before the whitespace-split mixed
        # parser. Requires the leading slash like every ring command.
        if _ring_is_slash and _ring_first == "segame":
            _after = _ring_raw.lstrip("/").strip().split(None, 1)
            _game = _after[1].strip() if len(_after) > 1 else ""
            from features.recording.sre_game import start_egame_escalation

            start_egame_escalation(
                _game,
                session_source,
                notify_chat,
                token,
                command_message_id=message_id,
                thread_root=root_id,
                operator_open_id=user_id,
                tenant_token=tenant_token or token,
            )
            return

        # /po <game> — PO product-manager escalation for ANY game by name (free-text; covers games with
        # no fixed /po<game> token — Hantak, OSM, EGS, 'Baccarat Tournament', 'Marble Race: Las Vegas', …).
        # Handled here (takes the rest of the line as the game name) before the whitespace-split parser.
        if _ring_is_slash and _ring_first == "po":
            _after = _ring_raw.lstrip("/").strip().split(None, 1)
            _game = _after[1].strip() if len(_after) > 1 else ""
            from features.recording.sre_game import start_po_game_escalation_by_name

            start_po_game_escalation_by_name(
                _game,
                session_source,
                notify_chat,
                token,
                command_message_id=message_id,
                thread_root=root_id,
                operator_open_id=user_id,
                tenant_token=tenant_token or token,
            )
            return

        # /sre <game> — SRE Game escalation by free-text game name (e.g. /sre Baccarat), same as the fixed
        # /srebac tokens but for any game. First=="sre" only (a fixed /srebac has first=="srebac").
        if _ring_is_slash and _ring_first == "sre":
            _after = _ring_raw.lstrip("/").strip().split(None, 1)
            _game = _after[1].strip() if len(_after) > 1 else ""
            from features.recording.sre_game import start_sre_game_escalation_by_name

            start_sre_game_escalation_by_name(
                _game,
                session_source,
                notify_chat,
                token,
                command_message_id=message_id,
                thread_root=root_id,
                operator_open_id=user_id,
                tenant_token=tenant_token or token,
            )
            return

        # Mixed commands in ONE message: "/srebac sfpms cpms" or "/scpms /fpms /c @Juan @Maria" — fire
        # EACH into the active meeting. SRE-game commands (srebac …) start an escalation; duty/direct
        # commands (scpms, fpms, dba, /c …) page their people. A LEADING SLASH is REQUIRED to trigger
        # (bare tokens AFTER the leading slash are honored, e.g. "/srebac sfpms cpms"); @mentioning the
        # bot alone is NOT enough, so casual chat / a stray @bot never pages. /c also needs @bot for its
        # @mention targets. A single command ("/fpms", "/m") also flows through here.
        _game_cmds, _ring_cmds = _parse_mixed_commands(_ring_raw)
        if (_game_cmds or _ring_cmds) and _ring_is_slash:
            _bot_mentioned = _mentions_our_bot(mention_names)
            _direct = [x for x in (kwargs.get("mention_open_ids") or []) if x]
            log.info(
                "mixed cmds detected game=%s ring=%s slash=%s bot_mentioned=%s tagged=%s chat_tail=%s",
                _game_cmds,
                _ring_cmds,
                _ring_is_slash,
                _bot_mentioned,
                len(_direct),
                chat_id[-8:] if chat_id else "",
            )
            from features.recording.sre_game import (
                is_po_game_command as _is_po,
                start_po_game_escalation,
                start_sre_game_escalation,
            )
            from features.recording.vc_ring import handle_ring_commands_batch

            for _gc in _game_cmds:
                _start_game = start_po_game_escalation if _is_po(_gc) else start_sre_game_escalation
                _start_game(
                    _gc,
                    session_source,
                    notify_chat,
                    token,
                    command_message_id=message_id,
                    thread_root=root_id,
                    operator_open_id=user_id,
                    tenant_token=tenant_token or token,
                )
            # /c needs @bot so the @mentions are unambiguously the targets; drop it otherwise.
            _batch_cmds = [x for x in _ring_cmds if x != "c" or _bot_mentioned]
            if _batch_cmds:
                # ONE consolidated "Calling selected duty persons …" reply for all duty/direct rings,
                # threaded under the command message (same thread as any /srebac escalation card).
                handle_ring_commands_batch(
                    _batch_cmds,
                    session_source,
                    notify_chat,
                    token,
                    operator_open_id=user_id,
                    tenant_token=tenant_token or token,
                    direct_open_ids=_direct,
                    reply_to_message_id=message_id,
                )
            return

        # /c @person… — ad-hoc DIRECT call (Model A): ring the TAGGED people straight from the message
        # @mentions. REQUIRES a leading slash AND @mentioning the bot (so the @mentions are unambiguously
        # the targets, and the bot is dropped). "c @a" without the slash / without @bot does NOT trigger.
        if _ring_first == "c" and _ring_is_slash and _mentions_our_bot(mention_names):
            _direct = [x for x in (kwargs.get("mention_open_ids") or []) if x]
            log.info(
                "direct ring /c slash=%s tagged=%s chat_tail=%s",
                _ring_is_slash,
                len(_direct),
                chat_id[-8:] if chat_id else "",
            )
            from features.recording.vc_ring import handle_ring_command

            handle_ring_command(
                "c",
                session_source,
                notify_chat,
                token,
                operator_open_id=user_id,
                tenant_token=tenant_token or token,
                direct_open_ids=_direct,
            )
            return

        if RING_CMD_RE.match(_ring_cmd):
            _bot_mentioned = _mentions_our_bot(mention_names)
            log.info(
                "ring cmd detected cmd=%r slash=%s bot_mentioned=%s mentions=%s chat_tail=%s session_source_tail=%s",
                _ring_cmd,
                _ring_is_slash,
                _bot_mentioned,
                mention_names,
                chat_id[-8:] if chat_id else "",
                session_source[-8:] if session_source else "",
            )
            if _ring_is_slash:
                from features.recording.vc_ring import handle_ring_command

                handle_ring_command(
                    _ring_cmd,
                    session_source,
                    notify_chat,
                    token,
                    operator_open_id=user_id,
                    tenant_token=tenant_token or token,
                )
                return

        if try_handle_graph_screenshot_request(
            text_raw,
            chat_id,
            tenant_token or token,
            source_chat_name,
            mention_names=mention_names,
            groq_key=groq_key,
            message_id=message_id,
        ):
            return

        if p0_group_typed_meeting_commands_enabled() and _matches_typed_end_meeting_command(text_raw):
            if chat_has_active_session(session_source):
                sess = P0_SESSIONS.get(session_source) or {}
                priority = str(sess.get("priority") or "P0").strip().upper()
                log.info(
                    "Incident UX: end meeting requested message_chat=%s session_source=%s priority=%s",
                    chat_id,
                    session_source,
                    priority,
                )
                end_p0_session(session_source, token)
            else:
                log.info(
                    "Incident UX: end requested but no active session message_chat=%s session_source=%s",
                    chat_id,
                    session_source,
                )
                if token:
                    post_text_to_chat(
                        notify_chat,
                        token,
                        "ℹ️ No active P0/P1 meeting in this chat to end.",
                    )
            return

        # Cancel command (optional reason after the keyword phrase).
        # Gated by P0_GROUP_TYPED_MEETING_COMMANDS so typed cancel can be turned off in the group.
        cancel_m = (
            CANCEL_WITH_OPTIONAL_REASON_RE.match(text_raw)
            if p0_group_typed_meeting_commands_enabled()
            else None
        )
        if cancel_m:
            tail = (cancel_m.group(2) or "").strip()
            cancel_reason = tail if tail else "Unspecified"
            if chat_has_active_session(session_source):
                sess = P0_SESSIONS.get(session_source) or {}
                priority = str(sess.get("priority") or "P0").strip().upper()
                log.info(
                    "Incident UX: cancel requested message_chat=%s session_source=%s priority=%s reason=%r",
                    chat_id,
                    session_source,
                    priority,
                    cancel_reason,
                )
                cancel_p0_session(session_source, token, reason=cancel_reason)
            else:
                log.info(
                    "Incident UX: cancel requested but no active session message_chat=%s session_source=%s",
                    chat_id,
                    session_source,
                )
                if token:
                    post_text_to_chat(
                        notify_chat,
                        token,
                        "ℹ️ No active P0/P1 meeting in this chat to cancel.",
                    )
            return

        # Cooldown reset only — no new VC. / 只清冷却，不创建会议
        if COOLDOWN_RESET_RE.match(text_raw.strip()):
            clear_p0_cooldown(session_source)
            if token:
                post_text_to_chat(
                    notify_chat,
                    token,
                    "ℹ️ Cooldown cleared for this group. The next **p0** or **p1** declaration in this chat will no longer be blocked by cooldown.",
                )
            return

        if is_detection:
            kw_text = _text_for_priority_keyword_trigger(text_raw, chat_label=source_chat_name)
            if P0_KEYWORD_RE.search(text_raw or "") and not P0_KEYWORD_RE.search(kw_text or ""):
                log.info(
                    "Incident group: P0 keyword ignored (Lark composer footer only) chat_id=%s footer_tail=%r",
                    chat_id,
                    (text_raw or "")[-60:],
                )
            # "p0" / "priority 0" anywhere in the message (unless pasted invite footer) — notify duty.
            _p0_kw_hit = (not _is_pasted_meeting_invite_footer(text_raw)) and P0_KEYWORD_RE.search(kw_text)
            if _p0_kw_hit and _text.is_manual_p0_incident_overview_template(text_raw):
                log.info(
                    "Incident group: P0 keyword ignored (manual P0 Incident Overview template) text_head=%r",
                    text_raw[:200],
                )
                return
            # Every non-command "p0" mention just notifies + buzzes duty — /p0 (the command) is the
            # only thing that can ever create a meeting (start_p0(..., via_command=True) refuses
            # everything else), so there is nothing to classify declare-vs-question about anymore.
            _p0_skip_for_issue_watch = _p0_kw_hit and get_p0_issue_watch_enabled()
            if _p0_skip_for_issue_watch:
                log.info(
                    "Incident group: P0 keyword — notifying duty, Issue Watch also analyzes this "
                    "message chat_id=%s text_tail=%r",
                    chat_id,
                    (text_raw or "")[-80:],
                )
                # No return — Issue Watch's own richer analysis (player counts, categories) still
                # runs on this same message below.
                _maybe_p0_keyword_confirm_dm(
                    chat_id=chat_id,
                    token=token,
                    user_id=user_id,
                    sender_lark_user_id=sender_lark_user_id,
                    source_chat_name=source_chat_name,
                    text_raw=text_raw,
                    message_id=message_id,
                )
            if _p0_kw_hit and not _p0_skip_for_issue_watch:
                log.info("Incident group: P0 keyword — notifying duty chat_id=%s user_id=%s", chat_id, user_id)
                _maybe_p0_keyword_confirm_dm(
                    chat_id=chat_id,
                    token=token,
                    user_id=user_id,
                    sender_lark_user_id=sender_lark_user_id,
                    source_chat_name=source_chat_name,
                    text_raw=text_raw,
                    message_id=message_id,
                )
                return

            # P1 mentioned — notify + buzz duty only. /p1 (the command) is the only thing that can
            # ever create a meeting (start_p0(..., via_command=True) refuses everything else), so
            # there is nothing here to confirm and no state to track — see start_p0's refusal gate.
            if (not _is_pasted_meeting_invite_footer(text_raw)) and P1_KEYWORD_RE.search(kw_text):
                if _is_explicit_p0_negation(text_raw):
                    log.info(
                        "Incident group: P1 trigger ignored (explicit negation) text_head=%r",
                        text_raw[:200],
                    )
                    return
                if (user_id or "").strip() in get_p0_trigger_ignore_open_ids():
                    log.info("Incident group: P1 trigger ignored (P0_TRIGGER_IGNORE_OPEN_IDS) user_id=%s", user_id)
                    return
                if chat_has_active_session(chat_id):
                    log.info("Incident group: session already active chat_id=%s", chat_id)
                    return

                ai = _priority_keyword_ai_triage(text_raw, groq_key)
                if ai is not None:
                    _p1_intent = str(ai.get("intent") or "").strip().lower()
                    # Same policy as P0: only an explicit negation is silent. Anything else — a
                    # question ("is this p1?"), a mention, a handoff — still notifies duty.
                    if _p1_intent in ("negation",) or _is_explicit_p0_negation(text_raw):
                        log.info(
                            "Incident group: P1 trigger ignored (explicit negation) intent=%s text_head=%r",
                            _p1_intent or "(none)",
                            text_raw[:200],
                        )
                        return
                    if _p1_intent != "declare_p1":
                        log.info(
                            "Incident group: P1 AI triage intent=%s — still notifying duty (ask, do not drop) "
                            "text_head=%r",
                            _p1_intent or "(none)",
                            text_raw[:200],
                        )

                kw_dedupe = _keyword_trigger_dedupe_key(
                    chat_id, user_id, message_id, message_create_time, text_raw
                )
                if not _try_consume_keyword_trigger_dedupe(kw_dedupe):
                    log.info(
                        "Incident group: P1 keyword skipped (duplicate Lark delivery, same create_time+text) chat_id=%s",
                        chat_id,
                    )
                    return

                log.info("Incident group: P1 keyword — notifying duty chat_id=%s user_id=%s", chat_id, user_id)
                notify_p1_mentioned(chat_id, token, user_id, phrase=text_raw, source_message_id=message_id)
                return

            if try_handle_issue_watch(
                _strip_lark_composer_message_footer(text_raw, chat_label=source_chat_name),
                chat_id,
                user_id,
                tenant_token or token,
                source_chat_name=source_chat_name,
                message_id=message_id,
                message_create_time=message_create_time,
            ):
                return

            # Ignore non P0/P1 chatter in the incident group to avoid noisy auto replies.
            log.info("Incident group: ignoring non P0/P1 message")
            return

        log.info(
            "Prompt/mirror session UX: ignoring message (use detection group to type p0/p1) message_chat=%s",
            chat_id,
        )
        return

    # ---------------------------------------------------------
    # On-demand Grafana screenshot (hub / allowed chats)
    # ---------------------------------------------------------
    if text_raw and try_handle_graph_screenshot_request(
        text_raw,
        chat_id,
        tenant_token or token,
        source_chat_name,
        mention_names=mention_names,
        groq_key=groq_key,
        message_id=message_id,
    ):
        return

    # ---------------------------------------------------------
    # WIKI GROUP
    # ---------------------------------------------------------
    if WIKI_GROUP_CHAT_ID and chat_id == WIKI_GROUP_CHAT_ID:
        if not text_raw:
            return

        log.info("Wiki group: routing to wiki")
        handle_wiki_ai(text_raw, chat_id, token, groq_key)
        return

    # ---------------------------------------------------------
    # DM / OTHER CHAT WHILE SESSION IS ACTIVE
    # ---------------------------------------------------------
    if has_active_session:
        if message_type == "image" and image_key:
            log.info(
                "Active session: handling image source_chat_id=%s active_session_chat_id=%s user_id=%s image_key=%s",
                chat_id,
                active_chat_id,
                user_id,
                image_key,
            )
            handle_dm_generate_overview(
                sender_open_id=user_id,
                tenant_token=tenant_token,
                image_key=image_key,
                mention_names=mention_names,
                message_id=message_id,
            )
            return

        if text_raw:
            log.info(
                "Active session: handling text source_chat_id=%s active_session_chat_id=%s user_id=%s text_head=%r",
                chat_id,
                active_chat_id,
                user_id,
                text_raw[:120],
            )
            handle_dm_generate_overview(
                sender_open_id=user_id,
                tenant_token=tenant_token,
                text=text_raw,
                mention_names=mention_names,
                message_id=message_id,
            )
            return

    log.info("Ignored message from non-allowed chat_id=%s", chat_id)
    return
