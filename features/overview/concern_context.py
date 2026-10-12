"""Rolling per-chat message buffer + concern resolution for the typed-"p0" auto-overview (Path B).

When a P0 is declared by TYPING "p0" (not an Issue Watch alert, which is tied to a specific detected
concern), the auto-overview needs to know WHICH concern it is about — especially in a busy group where
many issues were raised before the "p0". This resolves the concern text the overview is built from:

  1. reply / thread  -> the parent (replied-to) message's text  — EXACT (read back from Lark if not buffered)
  2. else, AI-pick    -> Claude chooses the concern among recent messages (P0_TYPED_DECLARE_CONCERN_AI_PICK)
  3. else, recent     -> the most recent substantive message before the "p0"
  4. else             -> the declaration text itself (unchanged)

Whichever message is picked, if it is a thread reply its thread root's text is prepended — a reply like
"can we consider this P0, more than 5 players" has no symptom; the thread's first message does.

The buffer records every group message the bot already receives (no new Lark scope), keeping each
message's ``parent_id``/``root_id`` so the reply anchor works without threading it through callers.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from p0_logic import config as _config

log = logging.getLogger("lark-ops-ai")

_BUF_LOCK = threading.Lock()
# chat_id -> deque[{message_id, parent_id, root_id, sender, text, ts}]
_BUF: Dict[str, "Deque[Dict[str, Any]]"] = {}
_BUF_MAX = 40          # messages kept per chat
_BUF_TTL = 7200.0      # 2h — older rows are ignored when resolving

_BARE_P0_RE = re.compile(r"(?is)^\W*(?:p0|priority\s*0|p1|priority\s*1)\W*$")
_ID_ONLY_RE = re.compile(r"(?is)^[\d\s,;:.\-]+$")

# Talk ABOUT the severity label rather than about the incident: "is this not P0 ?", "should we
# declare this as p0", "can we consider this as P1", "will declare this as P0". These carry no
# symptom, so they must never become the concern an overview is written from — duty replying to
# one of them is normal ("Will declare this as P0" is literally a reply to "Is this not P0 ?").
_PRIORITY_CHATTER_RE = re.compile(
    r"(?is)^\W*(?:@[\w_]+\s*)*"                       # leading @mentions
    r"(?:hi|hello|team|guys|po|ok|okay|yes|no|sure)?[\s,\.]*"
    r"(?:"
    r"(?:is|are|it'?s|this|that|so)\b[^.?!]{0,40}\bp[01]\b"          # "is this not P0 ?"
    r"|(?:can|should|shall|may|let'?s|will|we|i)\b[^.?!]{0,60}"
    r"\b(?:declare|consider|treat|tag|mark|raise|escalate)\b[^.?!]{0,40}\bp[01]\b"
    r"|(?:declar\w+|escalat\w+)\b[^.?!]{0,30}\bp[01]\b"
    r"|\bnot\s+(?:a\s+)?p[01]\b"
    r")"
    r"[\s\W]*$"
)


def _is_priority_chatter(text: str) -> bool:
    """True when the message is only about the P0/P1 label, not about a symptom."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    return bool(t) and bool(_PRIORITY_CHATTER_RE.match(t))


def record_group_message(
    chat_id: str,
    *,
    message_id: str = "",
    parent_id: str = "",
    root_id: str = "",
    sender_open_id: str = "",
    text: str = "",
    ts: Optional[float] = None,
) -> None:
    """Append a received group message to the rolling buffer. Cheap; called on every group message."""
    cid = (chat_id or "").strip()
    body = (text or "").strip()
    if not cid or not body:
        return
    row = {
        "message_id": (message_id or "").strip(),
        "parent_id": (parent_id or "").strip(),
        "root_id": (root_id or "").strip(),
        "sender": (sender_open_id or "").strip(),
        "text": body,
        "ts": float(ts) if ts else time.time(),
    }
    with _BUF_LOCK:
        dq = _BUF.get(cid)
        if dq is None:
            dq = deque(maxlen=_BUF_MAX)
            _BUF[cid] = dq
        dq.append(row)


def _rows(cid: str) -> List[Dict[str, Any]]:
    with _BUF_LOCK:
        return list(_BUF.get(cid) or [])


def _text_by_id(cid: str, message_id: str) -> str:
    mid = (message_id or "").strip()
    if not mid:
        return ""
    for r in reversed(_rows(cid)):
        if r["message_id"] == mid:
            return str(r.get("text") or "").strip()
    return ""


def _row_by_id(cid: str, message_id: str) -> Optional[Dict[str, Any]]:
    mid = (message_id or "").strip()
    if not mid:
        return None
    for r in reversed(_rows(cid)):
        if r["message_id"] == mid:
            return r
    return None


def _content_to_text(message_type: str, content: str) -> str:
    """Plain text of a fetched ``text`` / ``post`` message body ('' for anything else)."""
    try:
        obj = json.loads(content or "{}")
    except Exception:  # noqa: BLE001
        return ""
    if message_type == "text":
        return str((obj or {}).get("text") or "").strip()
    if message_type != "post":
        return ""
    parts: List[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            if str(v.get("tag") or "").lower() == "text" and isinstance(v.get("text"), str):
                parts.append(v["text"].strip())
            for child in v.values():
                walk(child)
        elif isinstance(v, list):
            for child in v:
                walk(child)

    # content_v2 duplicates content — walk only one of them.
    walk((obj or {}).get("content") or obj)
    return "\n".join(p for p in parts if p).strip()


def _message_text(cid: str, message_id: str) -> str:
    """Text of ``message_id``: rolling buffer first, else read it back from Lark (older than the
    buffer window, or sent before a restart). Only accepted if it belongs to ``cid``."""
    mid = (message_id or "").strip()
    if not mid:
        return ""
    hit = _text_by_id(cid, mid)
    if hit:
        return hit
    try:
        from p0_logic import lark_client as _lark

        msg = _lark.get_im_message(_lark.get_tenant_token_primary(), mid)
    except Exception as e:  # noqa: BLE001
        log.warning("concern_context: fetch message failed mid_tail=%s err=%s", mid[-12:], e)
        return ""
    if not msg or (msg.get("chat_id") and msg["chat_id"] != cid):
        return ""
    return _content_to_text(str(msg.get("message_type") or "").lower(), str(msg.get("content") or ""))


def _thread_root_id(cid: str, row: Dict[str, Any]) -> str:
    """Thread root of ``row``. A thread reply also shown in the main chat ("also send to chat") arrives
    as two messages and only the in-thread copy carries root_id — fall back to a same-text row's."""
    rid = str(row.get("root_id") or row.get("parent_id") or "").strip()
    if rid:
        return rid if rid != row.get("message_id") else ""
    text = str(row.get("text") or "").strip()
    for r in reversed(_rows(cid)):
        if r is not row and str(r.get("text") or "").strip() == text:
            rid = str(r.get("root_id") or r.get("parent_id") or "").strip()
            if rid and rid != r.get("message_id"):
                return rid
    return ""


def _with_thread_root(cid: str, row: Dict[str, Any]) -> str:
    """The row's text, preceded by its thread root's text when the row is a thread reply.

    A reply like "Can we consider this case as P0, more than 5 players" names no symptom — the
    symptom is in the thread's first message, which is often older than the concern window.
    """
    text = str(row.get("text") or "").strip()
    rid = _thread_root_id(cid, row)
    if not rid:
        return text
    root = _message_text(cid, rid)
    if not root or root == text or _is_bare_priority(root) or root in text:
        return text
    log.info("concern_context: prepended thread root to concern cid_tail=%s", cid[-8:])
    return f"{root}\n\n{text}"


def _is_bare_priority(text: str) -> bool:
    return bool(_BARE_P0_RE.match((text or "").strip()))


def _looks_substantive(text: str) -> bool:
    """A candidate concern: long enough, not priority talk, not just a list of IDs/numbers."""
    t = (text or "").strip()
    if len(t) < _config.get_p0_issue_watch_min_text_len():
        return False
    if _is_priority_chatter(t):
        return False
    if _is_bare_priority(t) or _ID_ONLY_RE.match(t):
        return False
    return True


def _recent_concerns(cid: str, *, exclude_message_id: str = "") -> List[Dict[str, Any]]:
    now = time.time()
    within = _config.get_p0_typed_declare_concern_window_min() * 60.0
    limit = _config.get_p0_typed_declare_concern_max_msgs()
    ex = (exclude_message_id or "").strip()
    out: List[Dict[str, Any]] = []
    for r in _rows(cid):
        if ex and r["message_id"] == ex:
            continue
        if (now - float(r.get("ts") or 0)) > within:
            continue
        if not _looks_substantive(str(r.get("text") or "")):
            continue
        out.append(r)
    return out[-limit:] if limit > 0 else out


def _ai_pick_concern(decl_text: str, concerns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ask Claude which recent message is the concern being declared P0. Returns its row or None."""
    if not concerns:
        return None
    from p0_logic import anthropic_client as _anthropic

    numbered = "\n".join(f"[{i}] {str(c.get('text') or '')[:400]}" for i, c in enumerate(concerns))
    system = (
        "You help an incident bot pick which recent chat message is the concern being declared as a "
        "P0 incident. You are given the declaration message and a numbered list of recent messages. "
        "Reply with ONLY the bracket number of the single message that best describes the concern "
        "being declared (e.g. 2). If none of them is a real concern, reply -1. No other text."
    )
    user = (
        f"Declaration message: {(decl_text or '(bare p0)').strip()[:300]!r}\n\n"
        f"Recent messages:\n{numbered}\n\n"
        "Which bracket number is the concern being declared as P0?"
    )
    try:
        ans = (_anthropic.anthropic_chat_once(system, user, max_tokens=8) or "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("concern_context: AI-pick failed err=%s", e)
        return None
    m = re.search(r"-?\d+", ans)
    if not m:
        return None
    try:
        idx = int(m.group(0))
    except ValueError:
        return None
    if 0 <= idx < len(concerns):
        log.info("concern_context: AI-pick chose [%s] of %s recent concern(s)", idx, len(concerns))
        return concerns[idx]
    return None


def _combine(concern: str, decl_text: str) -> str:
    """Concern is the source; append the declaration text only if it adds detail beyond a bare 'p0'."""
    concern = (concern or "").strip()
    decl = (decl_text or "").strip()
    if not concern:
        return decl
    if decl and not _is_bare_priority(decl) and decl not in concern:
        return f"{concern}\n\n{decl}"
    return concern


def resolve_declaration_concern(
    chat_id: str,
    *,
    decl_message_id: str = "",
    decl_text: str = "",
) -> str:
    """Resolve the concern text a typed-"p0" auto-overview should be built from (see module docstring).

    Falls back to ``decl_text`` unchanged when nothing better is found or the feature is off.
    """
    cid = (chat_id or "").strip()
    decl = (decl_text or "").strip()
    if not cid or not _config.get_p0_typed_declare_auto_overview_enabled():
        return decl

    # 1. Reply / thread anchor — the parent (or thread root) message's text. Parent id comes from the
    #    buffered declaration row, so no caller has to thread it through.
    row = _row_by_id(cid, decl_message_id)
    if row:
        for pid in (str(row.get("parent_id") or ""), str(row.get("root_id") or "")):
            ptext = _message_text(cid, pid)
            if not ptext or ptext == decl or _is_bare_priority(ptext):
                continue
            # Duty usually declares by REPLYING to "is this not P0?" — the parent is then priority
            # talk, not the concern. Fall through to AI-pick / most-recent instead of writing the
            # overview from a question.
            if _is_priority_chatter(ptext):
                log.info(
                    "concern_context: reply-parent is priority talk (%r) — falling back to AI-pick",
                    ptext[:60],
                )
                continue
            log.info("concern_context: resolved concern via reply-parent cid_tail=%s", cid[-8:])
            prow = _row_by_id(cid, pid) or {"message_id": pid, "text": ptext}
            return _combine(_with_thread_root(cid, prow), decl)

    concerns = _recent_concerns(cid, exclude_message_id=decl_message_id)
    if not concerns:
        return decl

    # 2. AI-pick among the recent concerns.
    if _config.get_p0_typed_declare_concern_ai_pick_enabled():
        picked = _ai_pick_concern(decl, concerns)
        if picked:
            return _combine(_with_thread_root(cid, picked), decl)

    # 3. Most recent substantive message.
    log.info("concern_context: resolved concern via most-recent (of %s) cid_tail=%s", len(concerns), cid[-8:])
    return _combine(_with_thread_root(cid, concerns[-1]), decl)
