"""
P0 session state: create, end, cancel, timers, and session lookup.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from p0_logic import cards as _cards
from p0_logic import config as _config
from p0_logic import lark_client as _lark
from . import session_disk as _session_disk
from p0_logic import support as _support

log = logging.getLogger("lark-ops-ai")

P0_SESSIONS: Dict[str, Dict[str, Any]] = {}
_LAST_P0_BY_CHAT: Dict[str, int] = {}
_LAST_P0_LOCK = threading.Lock()

# P1 keyword: waiting for Yes/No on "create meeting?" card (keyed by incident group chat_id)
P1_PROMPT_PENDING: Dict[str, Dict[str, Any]] = {}
_P1_PROMPT_LOCK = threading.Lock()

# Last successful ``end_p0_session`` fields per incident chat (in-memory) — replay ended card if user types "end" again.
_LAST_ENDED_SNAPSHOT_BY_CHAT: Dict[str, Dict[str, str]] = {}
_LAST_ENDED_SNAPSHOT_LOCK = threading.Lock()

P0_COOLDOWN_SEC = _config.P0_COOLDOWN_SEC

# Sentinel for DM overview queue items that are not tied to a live P0 session row.
STANDALONE_DM_SOURCE_CHAT_ID = "__standalone__"

# Per operator (open_id): one active DM instruction slot; further incidents queue until overview is sent.
_DM_INSTR_QUEUE: Dict[str, List[Dict[str, Any]]] = {}
_DM_ACTIVE_ITEM: Dict[str, Dict[str, Any]] = {}
_DM_INSTR_LOCK = threading.Lock()

# DM text when a second+ incident queues while the operator is still on the first overview.
_DM_CONCURRENT_MEETINGS_NOTICE = (
    "ℹ️ Multiple meetings were declared around the same time.\n"
    "Finish the first overview first then it will proceed to other one"
)


def _post_dm_concurrent_meetings_notice(operator_open_id: str, token: str) -> None:
    oid = (operator_open_id or "").strip()
    token = (token or "").strip()
    if not oid or not token:
        return
    try:
        st, body = _lark.post_text_to_open_id(oid, token, _DM_CONCURRENT_MEETINGS_NOTICE)
        if st != 200:
            log.warning(
                "concurrent meetings notice failed HTTP=%s open_id_tail=%s body=%s",
                st,
                oid[-8:] if len(oid) > 8 else oid,
                (body or "")[:400],
            )
    except Exception as e:
        log.warning("concurrent meetings notice exception open_id_tail=%s err=%s", oid[-8:] if len(oid) > 8 else oid, e)


def is_standalone_overview_active(operator_open_id: str) -> bool:
    """True when operator has an active ``coe`` / ``cog`` DM slot (green card or draft, no meeting)."""
    oid = (operator_open_id or "").strip()
    if not oid:
        return False
    with _DM_INSTR_LOCK:
        active = _DM_ACTIVE_ITEM.get(oid)
        if not active:
            return False
        return str(active.get("chat_id") or "").strip() == STANDALONE_DM_SOURCE_CHAT_ID


def note_if_standalone_create_overview_blocked(operator_open_id: str, tenant_token: str = "") -> str:
    """
    If non-empty, DM this text instead of enqueueing another standalone ``create overview``.
    Covers: active incident slot, duplicate standalone, draft tied to a live incident.

    When the incident meeting already ended but DM state was not released (or draft still
    points at the old ``oc_``), we heal that first so the operator is not told both
    \"use Build overview\" and \"no meeting — type create overview emergency\".
    """
    oid = (operator_open_id or "").strip()
    tok = (tenant_token or "").strip()
    if not oid:
        return ""

    stale_cid = ""
    with _DM_INSTR_LOCK:
        active = _DM_ACTIVE_ITEM.get(oid)
        if active:
            cid = str(active.get("chat_id") or "").strip()
            if cid and cid != STANDALONE_DM_SOURCE_CHAT_ID and not chat_has_active_session(cid):
                stale_cid = cid

    if stale_cid:
        if tok:
            release_dm_slots_for_incident_chat(stale_cid, tok)
        else:
            log.warning(
                "note_if_standalone: stale DM slot for ended incident, no token — dropping slot only open_id_tail=%s",
                oid[-8:] if len(oid) > 8 else oid,
            )
            with _DM_INSTR_LOCK:
                cur = _DM_ACTIVE_ITEM.get(oid)
                if cur and str(cur.get("chat_id") or "").strip() == stale_cid:
                    del _DM_ACTIVE_ITEM[oid]

    with _DM_INSTR_LOCK:
        active = _DM_ACTIVE_ITEM.get(oid)
        if active:
            cid = str(active.get("chat_id") or "").strip()
            if cid == STANDALONE_DM_SOURCE_CHAT_ID:
                return (
                    "ℹ️ Standalone overview is already active. Type c to abort, or finish this flow, "
                    "then type coe or cog again."
                )
            return "ℹ️ For this incident use the Build overview button on the DM card."

    from features.overview import drafts as _drafts

    _drafts.orphan_incident_draft_if_session_ended(oid)
    d = _drafts.get_draft(oid) or {}
    src = str(d.get("source_incident_chat_id") or "").strip()
    if src and src != STANDALONE_DM_SOURCE_CHAT_ID:
        return "ℹ️ For this incident use the Build overview button on the DM card."
    return ""


def get_dm_target_chat_for_operator(operator_open_id: str) -> str:
    """Target ``oc_`` for DM drafts while a queued slot is active (avoids wrong session when multiple P0 exist)."""
    oid = (operator_open_id or "").strip()
    if not oid:
        return ""
    with _DM_INSTR_LOCK:
        active = _DM_ACTIVE_ITEM.get(oid)
        if active:
            return str(active.get("target_chat") or "").strip()
    return get_active_target_chat() or _config.get_dm_overview_target_chat_id()


def enqueue_dm_instruction_if_needed(operator_open_id: str, token: str, item: Dict[str, Any]) -> None:
    """
    Post at most one DM instruction card per operator at a time. Additional incidents are queued FIFO
    until ``release_dm_after_overview_sent`` runs after a successful Send overview.
    """
    oid = (operator_open_id or "").strip()
    token = (token or "").strip()
    if not oid or not token:
        log.warning("enqueue_dm_instruction_if_needed: missing operator_open_id or token")
        return
    chat_id = (item.get("chat_id") or "").strip()
    target_chat = (item.get("target_chat") or "").strip()
    priority = (item.get("priority") or "P0").strip().upper()
    if priority not in ("P0", "P1"):
        priority = "P0"
    label = str(item.get("label") or "").strip()
    op_uid = str(item.get("operator_lark_user_id") or "").strip()
    norm = {
        "chat_id": chat_id,
        "target_chat": target_chat,
        "priority": priority,
        "label": label,
        "operator_lark_user_id": op_uid,
    }
    send_now = False
    with _DM_INSTR_LOCK:
        if oid not in _DM_ACTIVE_ITEM:
            _DM_ACTIVE_ITEM[oid] = norm
            send_now = True
            log.info(
                "DM instruction active (immediate) open_id_tail=%s incident=%s target=%s",
                oid[-8:] if len(oid) > 8 else oid,
                chat_id,
                target_chat,
            )
        else:
            _DM_INSTR_QUEUE.setdefault(oid, []).append(norm)
            log.info(
                "DM instruction queued open_id_tail=%s queue_len=%s incident=%s",
                oid[-8:] if len(oid) > 8 else oid,
                len(_DM_INSTR_QUEUE.get(oid) or []),
                chat_id,
            )
    if not send_now:
        _post_dm_concurrent_meetings_notice(oid, token)
        return
    _activate_dm_instruction_slot(
        oid,
        token,
        norm,
        context="DM instruction",
    )


def _activate_dm_instruction_slot(
    operator_open_id: str,
    token: str,
    item: Dict[str, Any],
    *,
    context: str,
    issue_watch_alert_key: str = "",
) -> bool:
    """Clear drafts, register DM slot, then green card or Issue Watch suggested preview."""
    oid = (operator_open_id or "").strip()
    tok = (token or "").strip()
    chat_id = str(item.get("chat_id") or "").strip()
    target_chat = str(item.get("target_chat") or "").strip()
    priority = str(item.get("priority") or "P0").strip().upper()
    if priority not in ("P0", "P1"):
        priority = "P0"
    label = str(item.get("label") or "").strip()
    op_uid = str(item.get("operator_lark_user_id") or "").strip()
    alert_key = (issue_watch_alert_key or "").strip()

    from features.overview import drafts as _drafts

    _drafts.clear_draft(oid)
    _drafts.clear_preview(oid)
    _drafts.cancel_preview_timer(oid)
    _drafts.seed_draft_for_incident(oid, target_chat, chat_id, draft_priority=priority)

    if alert_key and priority == "P0" and _config.get_p0_issue_watch_auto_overview_enabled():
        from features.issue_watch import issue_watch_overview as _iwo

        if _iwo.push_suggested_overview_on_p0_declare(
            oid,
            tok,
            alert_key,
            source_incident_chat_id=chat_id,
            target_chat=target_chat,
            source_chat_label=label,
        ):
            log.info(
                "%s: Issue Watch suggested overview open_id_tail=%s incident=%s alert_key=%s",
                context,
                oid[-8:] if len(oid) > 8 else oid,
                chat_id,
                alert_key[:12],
            )
            return True

    _send_dm_instruction_card_logged(
        oid,
        tok,
        priority,
        label,
        context=context,
        target_chat=target_chat,
        source_incident_chat_id=chat_id,
        operator_lark_user_id=op_uid,
    )
    return False


def enqueue_dm_issue_watch_overview_if_needed(
    operator_open_id: str,
    token: str,
    item: Dict[str, Any],
    alert_key: str,
) -> None:
    """On P0 declare with a recent Issue Watch alert: DM suggested preview instead of green card."""
    oid = (operator_open_id or "").strip()
    token = (token or "").strip()
    key = (alert_key or "").strip()
    if not oid or not token or not key:
        enqueue_dm_instruction_if_needed(operator_open_id, token, item)
        return
    chat_id = (item.get("chat_id") or "").strip()
    target_chat = (item.get("target_chat") or "").strip()
    priority = (item.get("priority") or "P0").strip().upper()
    if priority not in ("P0", "P1"):
        priority = "P0"
    label = str(item.get("label") or "").strip()
    op_uid = str(item.get("operator_lark_user_id") or "").strip()
    norm = {
        "chat_id": chat_id,
        "target_chat": target_chat,
        "priority": priority,
        "label": label,
        "operator_lark_user_id": op_uid,
    }
    send_now = False
    with _DM_INSTR_LOCK:
        if oid not in _DM_ACTIVE_ITEM:
            _DM_ACTIVE_ITEM[oid] = norm
            send_now = True
            log.info(
                "DM issue-watch overview active (immediate) open_id_tail=%s incident=%s alert_key=%s",
                oid[-8:] if len(oid) > 8 else oid,
                chat_id,
                key[:12],
            )
        else:
            _DM_INSTR_QUEUE.setdefault(oid, []).append(norm)
            log.info(
                "DM issue-watch overview queued open_id_tail=%s queue_len=%s incident=%s",
                oid[-8:] if len(oid) > 8 else oid,
                len(_DM_INSTR_QUEUE.get(oid) or []),
                chat_id,
            )
    if not send_now:
        _post_dm_concurrent_meetings_notice(oid, token)
        return
    _activate_dm_instruction_slot(
        oid,
        token,
        norm,
        context="Issue Watch overview on P0 declare",
        issue_watch_alert_key=key,
    )


def release_dm_slots_for_incident_chat(source_incident_chat_id: str, token: str) -> None:
    """
    When a P0/P1 session ends (end / cancel) without sending an overview, the operator's DM instruction
    slot must still advance — otherwise the next ``p1`` looks like a second concurrent incident and
    triggers the \"Multiple meetings\" notice.
    """
    src = (source_incident_chat_id or "").strip()
    tok = (token or "").strip()
    if not src or not tok:
        return
    with _DM_INSTR_LOCK:
        oids = [oid for oid, a in _DM_ACTIVE_ITEM.items() if str(a.get("chat_id") or "").strip() == src]
    for oid in oids:
        release_dm_after_overview_sent(oid, tok, src)


def release_dm_after_overview_sent(operator_open_id: str, token: str, sent_source_incident_chat_id: str) -> None:
    """After overview is posted to the group: advance the FIFO queue and post the next instruction card if any."""
    oid = (operator_open_id or "").strip()
    token = (token or "").strip()
    sent = (sent_source_incident_chat_id or "").strip()
    next_item: Optional[Dict[str, Any]] = None
    with _DM_INSTR_LOCK:
        active = _DM_ACTIVE_ITEM.get(oid)
        if not active:
            return
        exp = str(active.get("chat_id") or "").strip()
        if sent and exp and sent != exp:
            log.warning(
                "release_dm_after_overview_sent: source mismatch expected=%s got=%s open_id_tail=%s",
                exp,
                sent,
                oid[-8:] if len(oid) > 8 else oid,
            )
            return
        del _DM_ACTIVE_ITEM[oid]
        q = list(_DM_INSTR_QUEUE.get(oid) or [])
        if q:
            next_item = q.pop(0)
            _DM_INSTR_QUEUE[oid] = q
            _DM_ACTIVE_ITEM[oid] = next_item
    from features.overview import drafts as _drafts

    _drafts.clear_draft(oid)
    _drafts.clear_preview(oid)
    _drafts.cancel_preview_timer(oid)
    if next_item:
        _activate_dm_instruction_slot(oid, token, next_item, context="queued DM instruction")


def release_standalone_overview_cancel(operator_open_id: str, token: str) -> None:
    """
    Standalone ``create overview`` preview was cancelled without sending: remove the active
    slot so the operator can trigger ``create overview emergency|game`` again.
    Does **not** repost the green instruction card for the cancelled flow (caller sends text only).
    If another incident was queued, advance FIFO and post that instruction card.
    """
    oid = (operator_open_id or "").strip()
    token = (token or "").strip()
    if not oid:
        return
    next_item: Optional[Dict[str, Any]] = None
    with _DM_INSTR_LOCK:
        active = _DM_ACTIVE_ITEM.get(oid)
        if not active:
            return
        if str(active.get("chat_id") or "").strip() != STANDALONE_DM_SOURCE_CHAT_ID:
            return
        del _DM_ACTIVE_ITEM[oid]
        q = list(_DM_INSTR_QUEUE.get(oid) or [])
        if q:
            next_item = q.pop(0)
            _DM_INSTR_QUEUE[oid] = q
            _DM_ACTIVE_ITEM[oid] = next_item
    from features.overview import drafts as _drafts

    if next_item and token:
        tc = str(next_item.get("target_chat") or "").strip()
        cid = str(next_item.get("chat_id") or "").strip()
        pr = str(next_item.get("priority") or "P0").strip().upper()
        if pr not in ("P0", "P1"):
            pr = "P0"
        lab = str(next_item.get("label") or "").strip()
        q_op = str(next_item.get("operator_lark_user_id") or "").strip()
        _drafts.seed_draft_for_incident(oid, tc, cid, draft_priority=pr)
        _send_dm_instruction_card_logged(
            oid,
            token,
            pr,
            lab,
            context="queued DM after standalone cancel",
            target_chat=tc,
            source_incident_chat_id=cid,
            operator_lark_user_id=q_op,
        )


def _safe_match_ref(val: Any, meeting_ref: str) -> bool:
    s = str(val or "").strip()
    return bool(s and meeting_ref and s == meeting_ref)


def find_session_by_meeting_ref(meeting_ref: str) -> Tuple[str, Dict[str, Any]]:
    meeting_ref = (meeting_ref or "").strip()
    if not meeting_ref:
        return "", {}
    for chat_id, sess in P0_SESSIONS.items():
        if _safe_match_ref((sess or {}).get("meeting_id"), meeting_ref) or _safe_match_ref((sess or {}).get("meeting_no"), meeting_ref):
            return chat_id, (sess or {})
    return "", {}


def find_session_by_meeting_no(meeting_no: str) -> Tuple[str, Dict[str, Any]]:
    meeting_no = (meeting_no or "").strip()
    if not meeting_no:
        return "", {}
    for chat_id, sess in P0_SESSIONS.items():
        cur = str((sess or {}).get("meeting_no") or "").strip()
        if cur and cur == meeting_no:
            return chat_id, (sess or {})
    if _session_disk.enabled():
        cid, sd = _session_disk.find_session_by_meeting_no_disk(meeting_no)
        if cid and sd:
            P0_SESSIONS[cid] = sd
            return cid, sd
    return "", {}


def get_source_chat_label_for_target_chat(target_chat: str) -> str:
    """Human-readable source incident group name stored on the active session (if any)."""
    target_chat = (target_chat or "").strip()
    if not target_chat:
        return ""
    _cid, sess = find_session_by_target_chat(target_chat)
    return str((sess or {}).get("source_chat_name") or "").strip()


def find_session_by_target_chat(target_chat: str) -> Tuple[str, Dict[str, Any]]:
    target_chat = (target_chat or "").strip()
    if not target_chat:
        return "", {}
    for chat_id, sess in P0_SESSIONS.items():
        cur_target = str((sess or {}).get("target_chat") or "").strip()
        if cur_target == target_chat:
            return chat_id, (sess or {})
    return "", {}


def resolve_source_incident_chat_for_session_command(message_chat_id: str) -> str:
    """
    Map a **message** ``oc_`` (detection or prompt / mirror) to the session **source** incident ``oc_``.

    Used so **cancel** / **end** typed in the prompt group still find the row in ``P0_SESSIONS`` and
    release DM slots (``release_dm_slots_for_incident_chat``).
    """
    cid = (message_chat_id or "").strip()
    if not cid.startswith("oc_"):
        return ""
    if chat_has_active_session(cid):
        return cid
    det = _config.get_source_incident_chat_id_for_mirror_target(cid)
    if det and chat_has_active_session(det):
        return det
    src, _sess = find_session_by_target_chat(cid)
    if src:
        return src
    if _session_disk.enabled():
        src2, data = _session_disk.find_session_source_by_target_chat_disk(cid)
        if src2 and isinstance(data, dict):
            if src2 not in P0_SESSIONS:
                P0_SESSIONS[src2] = data
            return src2
    return ""


def get_active_session_key() -> str:
    if not P0_SESSIONS:
        return ""
    return list(P0_SESSIONS.keys())[-1]


def get_active_session() -> Optional[Dict[str, Any]]:
    key = get_active_session_key()
    return P0_SESSIONS.get(key) if key else None


def get_active_target_chat() -> str:
    if not P0_SESSIONS:
        return ""
    last_key = list(P0_SESSIONS.keys())[-1]
    sess = P0_SESSIONS.get(last_key) or {}
    target_chat = str(sess.get("target_chat") or "").strip()
    return target_chat or last_key


def _session_prompt_chat_id(sess: Dict[str, Any], source_incident_chat_id: str) -> str:
    """Group where meeting / P1 cards were posted: ``target_chat`` when split from detection group."""
    t = str((sess or {}).get("target_chat") or "").strip()
    return t or (source_incident_chat_id or "").strip()


def _patch_meeting_invite_to_terminal(
    sess: Dict[str, Any],
    token: str,
    *,
    kind: str,
    duration_text: str,
    cancel_reason: str = "",
) -> bool:
    """
    Replace the original red invite card in-place (same message_id) so chat is not spammed.
    ``kind`` = ``ended`` | ``cancelled``. Returns True if PATCH returned HTTP 200.
    """
    mid = str((sess or {}).get("meeting_invite_message_id") or "").strip()
    if not mid or not token:
        return False
    if str((sess or {}).get("meeting_invite_notice_kind") or "") == "text_unfurl":
        log.info(
            "patch meeting invite skipped kind=text_unfurl — Lark VC link preview updates in place mid=%s",
            mid[:24],
        )
        return True
    try:
        meeting_no = str(sess.get("meeting_no") or "").strip()
        priority = str(sess.get("priority") or "P0").strip().upper()
        em_topic = str(sess.get("emergency_topic") or "").strip()
        if kind == "ended":
            card = _cards.build_meeting_link_ended_card(
                priority=priority,
                duration_text=duration_text,
                meeting_no=meeting_no,
                emergency_topic=em_topic,
                update_multi=True,
            )
        elif kind == "cancelled":
            card = _cards.build_meeting_link_cancelled_card(
                priority=priority,
                duration_text=duration_text,
                meeting_no=meeting_no,
                reason=cancel_reason or "Unspecified",
                emergency_topic=em_topic,
                update_multi=True,
            )
        else:
            log.warning("patch meeting invite: unknown kind=%r", kind)
            return False
        st, body = _lark.patch_interactive_card(token, mid, card)
        if st != 200:
            log.warning(
                "patch meeting invite terminal failed HTTP=%s kind=%s message_id=%s body=%s",
                st,
                kind,
                mid,
                (body or "")[:500],
            )
            return False
        log.info("Patched meeting invite → %s message_id=%s", kind, mid)
        return True
    except Exception as e:
        log.warning("patch meeting invite terminal exception: %s", e)
        return False


def _store_last_ended_snapshot(
    chat_id: str,
    meeting_no: str,
    duration_text: str,
    priority: str,
    emergency_topic: str,
) -> None:
    chat_id = (chat_id or "").strip()
    if not chat_id:
        return
    prio = (priority or "P0").strip().upper()
    if prio not in ("P0", "P1"):
        prio = "P0"
    with _LAST_ENDED_SNAPSHOT_LOCK:
        _LAST_ENDED_SNAPSHOT_BY_CHAT[chat_id] = {
            "meeting_no": meeting_no or "",
            "duration_text": (duration_text or "Not available").strip() or "Not available",
            "priority": prio,
            "emergency_topic": (emergency_topic or "").strip(),
        }


def _clear_last_ended_snapshot(chat_id: str) -> None:
    chat_id = (chat_id or "").strip()
    with _LAST_ENDED_SNAPSHOT_LOCK:
        _LAST_ENDED_SNAPSHOT_BY_CHAT.pop(chat_id, None)


def get_last_ended_snapshot(chat_id: str) -> Optional[Dict[str, str]]:
    """Copy of last ``end_p0_session`` card fields for this chat, or None."""
    chat_id = (chat_id or "").strip()
    with _LAST_ENDED_SNAPSHOT_LOCK:
        d = _LAST_ENDED_SNAPSHOT_BY_CHAT.get(chat_id)
        return dict(d) if d else None


def bind_live_meeting_id(meeting_ref: str) -> None:
    """
    Store the live VC ``meeting.id`` from webhook ``vc.meeting.join_meeting_v1`` on the correct session.

    Prefer resolution by ``meeting_no`` / existing ``meeting_id``; if still unknown, bind to the only
    session that has no ``meeting_id`` yet, else the newest such session, else legacy fallback to the
    last in-memory key.
    """
    meeting_ref = (meeting_ref or "").strip()
    if not meeting_ref or not P0_SESSIONS:
        return
    cid, _ = find_session_by_meeting_ref(meeting_ref)
    if cid:
        sess = P0_SESSIONS.get(cid) or {}
        cur = str(sess.get("meeting_id") or "").strip()
        if cur != meeting_ref:
            sess["meeting_id"] = meeting_ref
            P0_SESSIONS[cid] = sess
            if _session_disk.enabled():
                _session_disk.save_session(cid, sess)
            log.info("Bound live meeting_id=%s to chat_id=%s (matched ref)", meeting_ref, cid)
        return
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for k, s in P0_SESSIONS.items():
        mid = str((s or {}).get("meeting_id") or "").strip()
        if not mid:
            candidates.append((k, s or {}))
    bind_key = ""
    if len(candidates) == 1:
        bind_key = candidates[0][0]
    elif len(candidates) > 1:
        candidates.sort(key=lambda t: int((t[1] or {}).get("start_epoch") or 0), reverse=True)
        bind_key = candidates[0][0]
        log.warning(
            "bind_live_meeting_id: multiple sessions missing meeting_id; bound meeting_ref=%s to newest chat_id=%s",
            meeting_ref,
            bind_key,
        )
    if bind_key:
        sess = P0_SESSIONS.get(bind_key) or {}
        sess["meeting_id"] = meeting_ref
        P0_SESSIONS[bind_key] = sess
        if _session_disk.enabled():
            _session_disk.save_session(bind_key, sess)
        log.info("Bound live meeting_id=%s to chat_id=%s (unbound session)", meeting_ref, bind_key)
        return
    last_key = list(P0_SESSIONS.keys())[-1]
    sess = P0_SESSIONS.get(last_key) or {}
    sess["meeting_id"] = meeting_ref
    P0_SESSIONS[last_key] = sess
    if _session_disk.enabled():
        _session_disk.save_session(last_key, sess)
    log.warning(
        "bind_live_meeting_id: fallback last chat_id=%s for meeting_ref=%s (all sessions had meeting_id)",
        last_key,
        meeting_ref,
    )


def record_vc_external_join_for_meeting_ref(meeting_ref: str, joiner_open_id: str) -> None:
    """
    Count a VC join as "external" (not the incident trigger) for auto-cancel-if-empty semantics.
    Persisted on the session row when disk is enabled.

    Only increments when ``joiner_open_id`` is **non-empty** and differs from the session
    ``trigger_open_id``. Join events without ``open_id`` are ignored so Lark noise does not
    block auto-cancel (empty room).
    """
    meeting_ref = (meeting_ref or "").strip()
    if not meeting_ref:
        return
    joiner_open_id = (joiner_open_id or "").strip()
    if not joiner_open_id:
        log.debug(
            "record_vc_external_join skipped (no joiner open_id) meeting_ref=%s",
            meeting_ref,
        )
        return
    cid, sess = find_session_by_meeting_ref(meeting_ref)
    if not cid or not sess:
        return
    trigger = str((sess or {}).get("trigger_open_id") or "").strip()
    if trigger and joiner_open_id == trigger:
        return
    sess2 = P0_SESSIONS.get(cid) or {}
    n = int(sess2.get("vc_external_join_count") or 0)
    sess2["vc_external_join_count"] = n + 1
    P0_SESSIONS[cid] = sess2
    if _session_disk.enabled():
        _session_disk.save_session(cid, sess2)
    log.info(
        "record_vc_external_join meeting_ref=%s chat_id=%s count=%s joiner_open_id=%s",
        meeting_ref,
        cid,
        sess2["vc_external_join_count"],
        joiner_open_id,
    )


def schedule_vc_auto_cancel_if_no_external_joins(chat_id: str) -> None:
    """Schedule auto-cancel when no external join was recorded (see config per-source chat)."""
    chat_id = (chat_id or "").strip()
    if not chat_id:
        return
    _config.reload_env_runtime()
    delay = float(_config.get_p0_vc_auto_cancel_sec_for_source_chat(chat_id))
    if delay <= 0:
        scoped = _config.get_p0_vc_auto_cancel_if_no_joins_chat_ids()
        if scoped and chat_id not in scoped:
            log.info(
                "vc auto-cancel: not scheduled chat_id=%s (not in P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_IDS)",
                chat_id,
            )
        elif scoped:
            log.info(
                "vc auto-cancel: not scheduled chat_id=%s (P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_SEC is 0)",
                chat_id,
            )
        else:
            log.info(
                "vc auto-cancel: not scheduled chat_id=%s (P0_VC_AUTO_CANCEL_IF_NO_JOINS_SEC unset or 0)",
                chat_id,
            )
        return
    sess0 = P0_SESSIONS.get(chat_id)
    if not sess0:
        log.warning("vc auto-cancel: not scheduled chat_id=%s (no session in memory)", chat_id)
        return
    run_id = secrets.token_hex(8)
    sess0["vc_auto_cancel_run_id"] = run_id
    P0_SESSIONS[chat_id] = sess0
    if _session_disk.enabled():
        _session_disk.save_session(chat_id, sess0)
    log.info(
        "vc auto-cancel: scheduled chat_id=%s delay_sec=%s run_id=%s",
        chat_id,
        int(delay),
        run_id,
    )

    def worker() -> None:
        time.sleep(delay)
        try:
            _config.reload_env_runtime()
            if _config.get_p0_vc_auto_cancel_sec_for_source_chat(chat_id) <= 0:
                log.info(
                    "vc auto-cancel: worker exit (config now disabled or chat not eligible) chat_id=%s",
                    chat_id,
                )
                return
            sess = P0_SESSIONS.get(chat_id)
            if not sess:
                log.info("vc auto-cancel: worker exit (session gone) chat_id=%s", chat_id)
                return
            if str(sess.get("vc_auto_cancel_run_id") or "") != run_id:
                log.info(
                    "vc auto-cancel: worker exit (stale run_id, new session?) chat_id=%s",
                    chat_id,
                )
                return
            ext = int(sess.get("vc_external_join_count") or 0)
            if ext > 0:
                log.info(
                    "vc auto-cancel: skipped chat_id=%s external_join_count=%s (need ou_ on join events to count)",
                    chat_id,
                    ext,
                )
                return
            tok = _lark.get_tenant_token_primary()
            if not tok:
                log.warning("vc auto-cancel: no primary tenant token chat_id=%s", chat_id)
                return
            log.info(
                "vc auto-cancel: no external joins after %s s — cancelling chat_id=%s",
                int(delay),
                chat_id,
            )
            cancel_p0_session(
                chat_id,
                tok,
                reason="No participants joined (auto-cancel).",
            )
        except Exception as e:
            log.warning("vc auto-cancel worker failed chat_id=%s err=%s", chat_id, e)

    threading.Thread(
        target=worker,
        name=f"vc-auto-cancel-{chat_id[-12:]}",
        daemon=True,
    ).start()


def _p0_ongoing_buzz_sent_key(tier: str) -> str:
    t = (tier or "").strip().lower()
    return "p0_ongoing_buzz_major_sent" if t == "major" else "p0_ongoing_buzz_minor_sent"


def _p0_ongoing_buzz_tier_allowed(sess: Dict[str, Any], tier: str) -> bool:
    t = (tier or "").strip().lower()
    return t in ("major", "minor")


def _send_p0_ongoing_dm_buzz(
    chat_id: str,
    sess: Dict[str, Any],
    token: str,
    trigger_open_id: str,
    *,
    tier: str,
    delay_sec: int,
) -> None:
    """DM duty operators — Major at 5 min, Minor at 10 min (configurable)."""
    if not token:
        log.warning("p0 ongoing buzz: no tenant token chat_id=%s tier=%s", chat_id, tier)
        return
    buzz_min = max(1, int(delay_sec) // 60)
    duration_text = f"{buzz_min} minute" if buzz_min == 1 else f"{buzz_min} minutes"
    label = str(sess.get("source_chat_name") or "").strip()
    if not label:
        label = _lark.get_group_chat_name(chat_id, token)
    meeting_no = str(sess.get("meeting_no") or "").strip()
    contacts = _config.get_p0_ongoing_contact_names()
    card = _cards.build_p0_ongoing_dm_buzz_card(
        source_chat_label=label,
        meeting_no=meeting_no,
        duration_text=duration_text,
        contact_names=contacts,
        severity_tier=tier,
    )
    targets = _dm_instruction_targets(trigger_open_id)
    urgent_mode = _config.get_p0_ongoing_lark_urgent_mode()
    sent = 0
    urgent_ok = 0
    for oid in targets:
        if not oid:
            continue
        st, body, mid = _lark.post_card_to_open_id(oid, token, card)
        if st != 200:
            log.warning(
                "p0 ongoing buzz: DM failed HTTP=%s open_id_tail=%s tier=%s body=%s",
                st,
                oid[-12:] if len(oid) > 12 else oid,
                tier,
                (body or "")[:300],
            )
            continue
        sent += 1
        if urgent_mode != "off" and mid:
            uok, udetail = _lark.urgent_message_for_users(token, mid, [oid], mode=urgent_mode)
            if uok:
                urgent_ok += 1
            else:
                log.warning(
                    "p0 ongoing buzz: Lark urgent_%s failed open_id_tail=%s tier=%s detail=%s "
                    "(enable im:message.urgent on the bot app?)",
                    urgent_mode,
                    oid[-12:] if len(oid) > 12 else oid,
                    tier,
                    (udetail or "")[:300],
                )
    log.info(
        "p0 ongoing buzz: sent=%s/%s urgent_%s=%s chat_id=%s tier=%s delay_sec=%s contacts=%r",
        sent,
        len(targets),
        urgent_mode,
        urgent_ok,
        chat_id,
        tier,
        delay_sec,
        contacts,
    )


def _mark_p0_ongoing_buzz_sent(chat_id: str, run_id: str, tier: str) -> None:
    sent_key = _p0_ongoing_buzz_sent_key(tier)
    sess2 = P0_SESSIONS.get(chat_id)
    if not sess2 or str(sess2.get("p0_ongoing_buzz_run_id") or "") != run_id:
        return
    sess2[sent_key] = True
    P0_SESSIONS[chat_id] = sess2
    if _session_disk.enabled():
        _session_disk.save_session(chat_id, sess2)


def _try_send_p0_ongoing_dm_buzz_now(
    chat_id: str,
    *,
    tier: str,
    run_id: str = "",
) -> bool:
    """Send buzz if session active and not already sent."""
    cid = (chat_id or "").strip()
    t = (tier or "").strip().lower()
    if not cid or t not in ("major", "minor"):
        return False
    sess = P0_SESSIONS.get(cid)
    if not sess:
        return False
    rid = (run_id or str(sess.get("p0_ongoing_buzz_run_id") or "")).strip()
    if rid and str(sess.get("p0_ongoing_buzz_run_id") or "") != rid:
        return False
    sent_key = _p0_ongoing_buzz_sent_key(t)
    if sess.get(sent_key):
        return False
    if str(sess.get("priority") or "").strip().upper() != "P0":
        return False
    if not _p0_ongoing_buzz_tier_allowed(sess, t):
        return False
    tok = _lark.get_tenant_token_primary()
    if not tok:
        log.warning("p0 ongoing buzz: no primary tenant token chat_id=%s tier=%s", cid, t)
        return False
    trigger = str(sess.get("trigger_open_id") or "").strip()
    delay_sec = (
        _config.get_p0_ongoing_dm_buzz_major_delay_sec()
        if t == "major"
        else _config.get_p0_ongoing_dm_buzz_minor_delay_sec()
    )
    _send_p0_ongoing_dm_buzz(cid, sess, tok, trigger, tier=t, delay_sec=delay_sec)
    _mark_p0_ongoing_buzz_sent(cid, rid, t)
    return True


def _start_p0_ongoing_buzz_worker(
    chat_id: str,
    trigger_open_id: str,
    run_id: str,
    *,
    tier: str,
    delay_sec: float,
) -> None:
    t = (tier or "").strip().lower()

    def worker() -> None:
        time.sleep(max(0.0, float(delay_sec)))
        try:
            _config.reload_env_runtime()
            if not _config.get_p0_ongoing_dm_buzz_enabled():
                return
            sess = P0_SESSIONS.get(chat_id)
            if not sess:
                log.info("p0 ongoing buzz: worker exit (session ended) chat_id=%s tier=%s", chat_id, t)
                return
            if str(sess.get("p0_ongoing_buzz_run_id") or "") != run_id:
                log.info("p0 ongoing buzz: worker exit (stale run_id) chat_id=%s tier=%s", chat_id, t)
                return
            if not _p0_ongoing_buzz_tier_allowed(sess, t):
                log.info(
                    "p0 ongoing buzz: worker skip (tier=%s) chat_id=%s",
                    t,
                    chat_id,
                )
                return
            _try_send_p0_ongoing_dm_buzz_now(chat_id, tier=t, run_id=run_id)
        except Exception as e:
            log.warning("p0 ongoing buzz worker failed chat_id=%s tier=%s err=%s", chat_id, t, e)

    threading.Thread(
        target=worker,
        name=f"p0-ongoing-buzz-{t}-{chat_id[-12:]}",
        daemon=True,
    ).start()


def schedule_p0_ongoing_dm_buzz(chat_id: str, trigger_open_id: str) -> None:
    """
    After P0 start, DM operators when the meeting is still active:
    **Major** → 5 min (default); **Minor** → 10 min (default).
    """
    chat_id = (chat_id or "").strip()
    if not chat_id or not _config.get_p0_ongoing_dm_buzz_enabled():
        return
    sess0 = P0_SESSIONS.get(chat_id)
    if not sess0:
        log.warning("p0 ongoing buzz: not scheduled chat_id=%s (no session)", chat_id)
        return
    if str(sess0.get("priority") or "").strip().upper() != "P0":
        log.info("p0 ongoing buzz: not scheduled chat_id=%s (priority is not P0)", chat_id)
        return
    major_delay = float(_config.get_p0_ongoing_dm_buzz_major_delay_sec())
    minor_delay = float(_config.get_p0_ongoing_dm_buzz_minor_delay_sec())
    if major_delay <= 0 and minor_delay <= 0:
        return
    run_id = secrets.token_hex(8)
    sess0["p0_ongoing_buzz_run_id"] = run_id
    sess0["p0_ongoing_buzz_major_sent"] = False
    sess0["p0_ongoing_buzz_minor_sent"] = False
    sess0.pop("p0_ongoing_buzz_sent", None)
    P0_SESSIONS[chat_id] = sess0
    if _session_disk.enabled():
        _session_disk.save_session(chat_id, sess0)
    log.info(
        "p0 ongoing buzz: scheduled chat_id=%s run_id=%s major_sec=%s minor_sec=%s",
        chat_id,
        run_id,
        int(major_delay),
        int(minor_delay),
    )
    if major_delay > 0:
        _start_p0_ongoing_buzz_worker(
            chat_id,
            trigger_open_id,
            run_id,
            tier="major",
            delay_sec=major_delay,
        )
    if minor_delay > 0:
        _start_p0_ongoing_buzz_worker(
            chat_id,
            trigger_open_id,
            run_id,
            tier="minor",
            delay_sec=minor_delay,
        )


def end_p0_session(
    chat_id: str,
    token: Optional[str] = None,
    *,
    vc_end_meeting_id: str = "",
    skip_vc_end: bool = False,
) -> None:
    chat_id = (chat_id or "").strip()
    sess = P0_SESSIONS.get(chat_id) or {}
    if not sess and _session_disk.enabled():
        sess = _session_disk.load_session(chat_id) or {}
    if sess and chat_id and chat_id not in P0_SESSIONS:
        P0_SESSIONS[chat_id] = sess
    if sess:
        meeting_no_snap = str(sess.get("meeting_no") or "").strip()
        start_epoch_snap = int(sess.get("start_epoch") or 0)
        duration_snap = _cards.format_duration(start_epoch_snap)
        priority_snap = str(sess.get("priority") or "P0").strip().upper()
        em_snap = str(sess.get("emergency_topic") or "").strip()
        _store_last_ended_snapshot(chat_id, meeting_no_snap, duration_snap, priority_snap, em_snap)
    if token and sess:
        # End the live VC on Lark so recording / Video Meeting Assistant can finalize (same as End in the client).
        # ``vc.meeting.meeting_ended_v1`` fires *after* the meeting is already over — POST .../meetings/{id}/end then
        # returns 404; skip those calls and only clean up reserve + cards.
        preferred = (vc_end_meeting_id or "").strip()
        meeting_id = preferred or str(sess.get("meeting_id") or "").strip()
        meeting_no = str(sess.get("meeting_no") or "").strip()
        reserve_id = str(sess.get("reserve_id") or "").strip()
        vc_ended = False
        if not skip_vc_end:
            if meeting_id:
                vc_ended = _lark.end_vc_meeting(token, meeting_id)
                if not vc_ended:
                    log.warning("end_p0_session: end_vc_meeting failed meeting_id=%s", meeting_id)
            if not vc_ended and meeting_no and meeting_no != meeting_id:
                vc_ended = _lark.end_vc_meeting(token, meeting_no)
                if vc_ended:
                    log.info("end_p0_session: ended VC via meeting_no=%s", meeting_no)
                else:
                    log.warning("end_p0_session: end_vc_meeting failed meeting_no=%s", meeting_no)
        else:
            log.info("end_p0_session: skip_vc_end=1 (meeting already ended on Lark)")
        if not vc_ended and reserve_id:
            _lark.delete_vc_reserve(token, reserve_id)
        start_epoch = int(sess.get("start_epoch") or 0)
        priority = str(sess.get("priority") or "P0").strip().upper()
        duration_text = _cards.format_duration(start_epoch)
        em_topic = str(sess.get("emergency_topic") or "").strip()
        patched = _patch_meeting_invite_to_terminal(sess, token, kind="ended", duration_text=duration_text)
        if not patched:
            log.warning(
                "end_p0_session: could not patch invite card to ended state chat_id=%s message_id=%s",
                chat_id,
                str(sess.get("meeting_invite_message_id") or "")[:24],
            )
    if token and chat_id:
        s_end = P0_SESSIONS.get(chat_id)
        if s_end and s_end.get("dm_instruction_deferred"):
            _flush_deferred_dm_instruction_for_incident(chat_id)
    P0_SESSIONS.pop(chat_id, None)
    _session_disk.delete_session(chat_id)
    if token:
        release_dm_slots_for_incident_chat(chat_id, token)
    try:
        from features.screenshot.graph_screenshot import on_p0_session_ended_for_graph_screenshot

        on_p0_session_ended_for_graph_screenshot()
    except Exception as e:
        log.warning("end_p0_session: graph screenshot interval stop failed: %s", e)
    if token and sess:
        try:
            from features.recording.vc_recording_fanout import schedule_recording_fanout_from_p0_session

            schedule_recording_fanout_from_p0_session(
                token,
                chat_id=chat_id,
                meeting_id=str(sess.get("meeting_id") or "").strip(),
                meeting_no=str(sess.get("meeting_no") or "").strip(),
                emergency_topic=str(sess.get("emergency_topic") or "").strip(),
                start_epoch=int(sess.get("start_epoch") or 0),
            )
        except Exception as e:
            log.warning("end_p0_session: recording fanout schedule failed: %s", e)


def cancel_p0_session(
    chat_id: str,
    token: Optional[str] = None,
    reason: str = "Unspecified",
) -> None:
    chat_id = (chat_id or "").strip()
    sess = P0_SESSIONS.get(chat_id) or {}
    if not sess and _session_disk.enabled():
        sess = _session_disk.load_session(chat_id) or {}
    if sess and chat_id and chat_id not in P0_SESSIONS:
        P0_SESSIONS[chat_id] = sess
    if sess:
        _clear_last_ended_snapshot(chat_id)
    reserve_id = str(sess.get("reserve_id") or "").strip()
    meeting_id = str(sess.get("meeting_id") or "").strip()
    meeting_no = str(sess.get("meeting_no") or "").strip()
    if token and sess:
        meeting_ended = False
        if meeting_id:
            meeting_ended = _lark.end_vc_meeting(token, meeting_id)
        if (not meeting_ended) and reserve_id:
            _lark.delete_vc_reserve(token, reserve_id)
        log.info("cancel_p0_session VC action chat_id=%s reserve_id=%s meeting_id=%s meeting_no=%s", chat_id, reserve_id, meeting_id, meeting_no)
        start_epoch = int(sess.get("start_epoch") or 0)
        priority = str(sess.get("priority") or "P0").strip().upper()
        duration_text = _cards.format_duration(start_epoch)
        em_topic = str(sess.get("emergency_topic") or "").strip()
        patched = _patch_meeting_invite_to_terminal(
            sess, token, kind="cancelled", duration_text=duration_text, cancel_reason=reason
        )
        prompt_cid = _session_prompt_chat_id(sess, chat_id)
        if not patched:
            try:
                _lark.post_card_to_chat(
                    prompt_cid,
                    token,
                    _cards.build_meeting_link_cancelled_card(
                        priority=priority,
                        duration_text=duration_text,
                        meeting_no=meeting_no,
                        reason=reason,
                        emergency_topic=em_topic,
                        update_multi=False,
                    ),
                )
            except Exception as e:
                log.error("Failed to post meeting cancelled card (fallback): %s", e)
        _fanout_p0_meeting_cancelled(
            token,
            source_incident_chat_id=chat_id,
            prompt_chat_id=prompt_cid,
            meeting_no=meeting_no,
            duration_text=duration_text,
            priority=priority,
            reason=reason,
            emergency_topic=em_topic,
        )
    if token and chat_id:
        s_can = P0_SESSIONS.get(chat_id)
        if s_can and s_can.get("dm_instruction_deferred"):
            _flush_deferred_dm_instruction_for_incident(chat_id)
    P0_SESSIONS.pop(chat_id, None)
    _session_disk.delete_session(chat_id)
    if token:
        release_dm_slots_for_incident_chat(chat_id, token)
    try:
        from features.screenshot.graph_screenshot import on_p0_session_ended_for_graph_screenshot

        on_p0_session_ended_for_graph_screenshot()
    except Exception as e:
        log.warning("cancel_p0_session: graph screenshot interval stop failed: %s", e)


def end_p0_session_by_meeting_no(meeting_no: str, token: Optional[str] = None) -> None:
    chat_id, _ = find_session_by_meeting_no(meeting_no)
    if not chat_id:
        log.warning("No active p0 session found for meeting_no=%s", meeting_no)
        return
    end_p0_session(chat_id, token)


def end_p0_session_by_meeting_ref(
    meeting_ref: str,
    token: Optional[str] = None,
    *,
    meeting_no_fallback: str = "",
) -> None:
    """Resolve session by long ``meeting.id`` or stored ref; optional ``meeting_no`` if join never bound."""
    meeting_ref = (meeting_ref or "").strip()
    chat_id, _ = find_session_by_meeting_ref(meeting_ref)
    if not chat_id and meeting_no_fallback:
        chat_id, _ = find_session_by_meeting_no(meeting_no_fallback.strip())
    if not chat_id:
        log.warning(
            "No active p0 session found for meeting_ref=%s meeting_no_fallback=%s",
            meeting_ref,
            meeting_no_fallback,
        )
        return
    end_p0_session(chat_id, token, vc_end_meeting_id=meeting_ref, skip_vc_end=True)


def cancel_p0_session_by_meeting_no(
    meeting_no: str,
    token: Optional[str] = None,
    reason: str = "Unspecified",
) -> None:
    chat_id, _ = find_session_by_meeting_no(meeting_no)
    if not chat_id:
        log.warning("No active p0 session found for meeting_no=%s", meeting_no)
        return
    cancel_p0_session(chat_id, token, reason=reason)


def _dm_instruction_targets(trigger_open_id: str) -> List[str]:
    """open_ids that receive the DM instruction card: env list if set, else [trigger] if any."""
    fixed = _config.get_dm_instruction_open_ids()
    if fixed:
        return fixed
    t = (trigger_open_id or "").strip()
    return [t] if t else []


def _parse_lark_api_code(raw: Any) -> int:
    """Lark responses use numeric ``code`` (often int, sometimes str). Never raises."""
    if raw is None:
        return -1
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return -1
        try:
            return int(s)
        except ValueError:
            return -1
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return -1


def _dm_instruction_lark_response_ok(st: int, resp_body: str) -> bool:
    """True when HTTP 200 and Lark ``code`` is 0 (or missing)."""
    if st != 200:
        return False
    try:
        j = json.loads(resp_body) if resp_body else {}
    except Exception:
        return False
    if not isinstance(j, dict):
        return False
    if "code" not in j:
        return True
    return _parse_lark_api_code(j.get("code")) == 0


def _send_dm_instruction_card_logged(
    open_id: str,
    tenant_token: str,
    priority: str,
    source_chat_label: str,
    context: str = "",
    *,
    target_chat: str = "",
    source_incident_chat_id: str = "",
    operator_lark_user_id: str = "",
) -> None:
    """
    Send the green **Build overview** DM instruction card.

    Prefers ``receive_id_type=user_id`` when ``operator_lark_user_id`` is set (same tenant id as group
    messages) — some tenants return HTTP 400 / 230099 for interactive cards via ``open_id`` only.
    Falls back to ``open_id`` if the user_id path fails.
    """
    oid = (open_id or "").strip()
    if not oid:
        return
    label = (context or "DM instruction").strip()
    card = _cards.build_dm_instruction_card(
        priority,
        source_chat_label=source_chat_label,
        target_chat=target_chat,
        source_incident_chat_id=source_incident_chat_id,
    )
    uid = (operator_lark_user_id or "").strip()
    attempts: List[Tuple[str, Any]] = []
    if uid:
        attempts.append(
            (
                "user_id",
                lambda: _lark.post_card_to_user_cross_app(oid, uid, tenant_token, card, use_user_id=True),
            )
        )
    attempts.append(("open_id", lambda: _lark.post_card_to_open_id(oid, tenant_token, card)))
    try:
        st, resp_body, mid = 0, "", ""
        mode_used = "open_id"
        for i, (name, fn) in enumerate(attempts):
            st, resp_body, mid = fn()
            if _dm_instruction_lark_response_ok(st, resp_body):
                mode_used = name
                break
            if i + 1 < len(attempts):
                log.warning(
                    "%s: %s path failed HTTP=%s open_id_tail=%s body=%s — retrying",
                    label,
                    name,
                    st,
                    oid[-8:] if len(oid) > 8 else oid,
                    (resp_body or "")[:400],
                )
        else:
            body_head = (resp_body or "")[:800]
            log.error(
                "%s failed after %s priority=%s open_id=%s body=%s",
                label,
                [a[0] for a in attempts],
                priority,
                oid,
                body_head,
            )
            return
        log.info(
            "%s sent OK via %s priority=%s open_id=%s message_id=%s",
            label,
            mode_used,
            priority,
            oid,
            (mid or "").strip() or "(none)",
        )
    except Exception as e:
        log.error("%s exception priority=%s open_id=%s err=%s", label, priority, oid, e)


def get_p1_prompt_pending(chat_id: str) -> Optional[Dict[str, Any]]:
    chat_id = (chat_id or "").strip()
    if not chat_id:
        return None
    with _P1_PROMPT_LOCK:
        p = P1_PROMPT_PENDING.get(chat_id)
        return dict(p) if p else None


def set_p1_prompt_pending(chat_id: str, trigger_open_id: str) -> str:
    """Store pending P1 meeting confirmation; returns nonce embedded in the Yes/No card buttons."""
    chat_id = (chat_id or "").strip()
    trigger_open_id = (trigger_open_id or "").strip()
    if not chat_id:
        return ""
    nonce = secrets.token_hex(8)
    with _P1_PROMPT_LOCK:
        P1_PROMPT_PENDING[chat_id] = {
            "trigger_open_id": trigger_open_id,
            "ts": int(time.time()),
            "nonce": nonce,
        }
    return nonce


def pop_p1_prompt_pending(chat_id: str) -> Optional[Dict[str, Any]]:
    chat_id = (chat_id or "").strip()
    if not chat_id:
        return None
    with _P1_PROMPT_LOCK:
        p = P1_PROMPT_PENDING.pop(chat_id, None)
        return dict(p) if p else None


def consume_p1_prompt_for_confirm(chat_id: str, nonce_from_button: str = "") -> Optional[Dict[str, Any]]:
    """
    Remove P1 meeting-confirm pending only if button nonce matches (stops stale card clicks).
    If payload has no nonce (legacy card), only consume when stored pending has no nonce.
    """
    chat_id = (chat_id or "").strip()
    want = (nonce_from_button or "").strip()
    if not chat_id:
        return None
    with _P1_PROMPT_LOCK:
        p = P1_PROMPT_PENDING.get(chat_id)
        if not p:
            return None
        stored = str(p.get("nonce") or "").strip()
        if want:
            if stored != want:
                log.warning("P1 confirm ignored: nonce mismatch chat_id=%s", chat_id)
                return None
        else:
            if stored:
                log.warning("P1 confirm ignored: card missing nonce but server expects one chat_id=%s", chat_id)
                return None
        P1_PROMPT_PENDING.pop(chat_id, None)
        return dict(p)


def request_p1_meeting_confirmation(chat_id: str, token: str, trigger_open_id: str) -> bool:
    """Post Yes/No card in the same chat as meeting cards (``get_session_meeting_card_post_chat_id``)."""
    chat_id = (chat_id or "").strip()
    token = (token or "").strip()
    if not chat_id or not token:
        return False
    pend = get_p1_prompt_pending(chat_id)
    nonce = str((pend or {}).get("nonce") or "").strip()
    if not nonce:
        log.error("request_p1_meeting_confirmation: no pending nonce for chat_id=%s", chat_id)
        return False
    prompt_chat = _config.get_session_meeting_card_post_chat_id(chat_id)
    st, body, _ = _lark.post_card_to_chat(prompt_chat, token, _cards.build_p1_meeting_confirm_card(nonce))
    if st != 200:
        log.error("request_p1_meeting_confirmation failed HTTP=%s body=%s", st, (body or "")[:500])
        return False
    return True


def chat_has_active_session(chat_id: str) -> bool:
    cid = (chat_id or "").strip()
    if not cid:
        return False
    if cid in P0_SESSIONS:
        return True
    if _session_disk.enabled():
        d = _session_disk.load_session(cid)
        if d:
            P0_SESSIONS[cid] = d
            return True
    return False


def dm_preview_allowed_for_incident(source_incident_chat_id: str, target_chat: str) -> bool:
    """
    DM overview (Build overview / Send to group) is tied to a live P0/P1 session for the incident group.
    Standalone ``create overview`` flows (no VC) stay allowed without a session.
    """
    src = (source_incident_chat_id or "").strip()
    tc = (target_chat or "").strip()
    if src == STANDALONE_DM_SOURCE_CHAT_ID:
        return True
    if src and src != STANDALONE_DM_SOURCE_CHAT_ID:
        return chat_has_active_session(src)
    if tc:
        _cid, sess = find_session_by_target_chat(tc)
        return bool(sess)
    return False


def handle_p1_meeting_confirm_yes(
    chat_id: str, token: str, fallback_trigger_open_id: str, nonce: str
) -> str:
    """
    Consume P1 "create meeting?" pending and start a P1 VC. Used by card **create** action and typed **create meeting**.

    Returns ``""`` on success, or ``"session_active"`` / ``"stale"``.
    """
    chat_id = (chat_id or "").strip()
    token = (token or "").strip()
    if not chat_id or not token:
        return "stale"
    if chat_has_active_session(chat_id):
        return "session_active"
    pending = consume_p1_prompt_for_confirm(chat_id, nonce)
    if not pending:
        return "stale"
    trigger = str(pending.get("trigger_open_id") or "").strip() or (fallback_trigger_open_id or "").strip()
    start_p0(chat_id, token, trigger, priority="P1")
    return ""


def handle_p1_meeting_confirm_no(chat_id: str, token: str, nonce: str) -> str:
    """
    Consume P1 prompt and skip VC. Returns ``""``, ``"session_active"``, or ``"stale"``.
    """
    chat_id = (chat_id or "").strip()
    token = (token or "").strip()
    if not chat_id or not token:
        return "stale"
    if chat_has_active_session(chat_id):
        return "session_active"
    pending = consume_p1_prompt_for_confirm(chat_id, nonce)
    if not pending:
        return "stale"
    _lark.post_text_to_chat(
        chat_id,
        token,
        "ℹ️ No P1 meeting will be created. Type **p1** in this group again when you need a new meeting.",
    )
    return ""


def p0_cooldown(chat_id: str) -> bool:
    chat_id = (chat_id or "").strip()
    if not chat_id:
        return False
    now = int(time.time())
    with _LAST_P0_LOCK:
        last = _LAST_P0_BY_CHAT.get(chat_id, 0)
        if now - last < P0_COOLDOWN_SEC:
            return True
        _LAST_P0_BY_CHAT[chat_id] = now
        return False


def p0_cooldown_remaining_sec(chat_id: str) -> int:
    """Seconds left before this chat can trigger p0/p1 again (0 if cooldown clear). Read-only."""
    chat_id = (chat_id or "").strip()
    if not chat_id:
        return 0
    now = int(time.time())
    with _LAST_P0_LOCK:
        last = _LAST_P0_BY_CHAT.get(chat_id, 0)
        elapsed = now - last
        if elapsed >= P0_COOLDOWN_SEC:
            return 0
        return int(P0_COOLDOWN_SEC - elapsed)


def _resolve_lark_user_id_for_dm(
    operator_open_id: str,
    sess: Dict[str, Any],
    explicit_lark_user_id: str,
) -> str:
    """Tenant user_id for cross-app DM; prefer event payload, then session trigger, then contact lookup."""
    ex = (explicit_lark_user_id or "").strip()
    if ex:
        return ex
    oid = (operator_open_id or "").strip()
    tr_open = str(sess.get("trigger_open_id") or "").strip()
    if oid and tr_open and oid == tr_open:
        u = str(sess.get("trigger_lark_user_id") or "").strip()
        if u:
            return u
    tok_p = _lark.get_tenant_token_primary()
    return _lark.get_tenant_user_id_by_open_id(tok_p, oid)


def clear_p0_cooldown(chat_id: str) -> None:
    """
    Drop the per-chat cooldown timestamp so **p0** / **p1** keywords can fire again
    without waiting — does **not** start a meeting.
    """
    chat_id = (chat_id or "").strip()
    if not chat_id:
        return
    with _LAST_P0_LOCK:
        _LAST_P0_BY_CHAT.pop(chat_id, None)
    log.info("clear_p0_cooldown chat_id=%s", chat_id)


def _post_meeting_link_unfurl_notice(
    chat_id: str,
    token: str,
    *,
    link: str,
    priority: str,
    emergency_topic: str = "",
) -> str:
    """
    Option A: one plain-text message — topic, P0 created, join label, raw URL (Lark unfurl below).
    No card, no markdown asterisks.
    """
    cid = (chat_id or "").strip()
    url = (link or "").strip()
    if not cid or not token:
        return ""
    text = _cards.build_p0_meeting_created_text(
        url, priority=priority, emergency_topic=emergency_topic
    )
    try:
        st, body = _lark.post_text_to_chat(cid, token, text)
        ok, code, msg = _lark.lark_im_message_create_ok(body)
        mid = _lark.parse_im_message_id_from_response(body)
        if st == 200 and ok and mid:
            log.info(
                "meeting plain-text notice ok chat_id_tail=%s priority=%s mid_tail=%s",
                cid[-12:] if len(cid) > 12 else cid,
                priority,
                mid[-12:] if len(mid) > 12 else mid,
            )
            return mid
        log.warning(
            "meeting plain-text notice failed chat_id_tail=%s HTTP=%s lark_code=%s msg=%r",
            cid[-12:] if len(cid) > 12 else cid,
            st,
            code,
            msg,
        )
    except Exception as e:
        log.warning("meeting plain-text notice exception chat_id_tail=%s err=%s", cid[-12:], e)
    return ""


def _fanout_p0_meeting_created_link_notice(
    token: str,
    source_incident_chat_id: str,
    prompt_chat_id: str,
    *,
    link: str,
    priority: str = "P0",
    emergency_topic: str = "",
) -> None:
    """Same unfurl text to boss / hub — native Lark VC preview below the link."""
    targets = _config.get_p0_meeting_created_text_fanout_chat_ids(source_incident_chat_id)
    if not targets:
        return
    prompt = (prompt_chat_id or "").strip()
    for oc in targets:
        if oc == prompt:
            continue
        _post_meeting_link_unfurl_notice(
            oc,
            token,
            link=link,
            priority=priority,
            emergency_topic=emergency_topic,
        )


def _fanout_p0_meeting_cancelled(
    token: str,
    *,
    source_incident_chat_id: str,
    prompt_chat_id: str,
    meeting_no: str,
    duration_text: str,
    priority: str,
    reason: str,
    emergency_topic: str,
) -> None:
    """Post the grey cancelled card to boss / hub groups (prompt group already got the primary notice)."""
    targets = _config.get_p0_meeting_cancelled_fanout_chat_ids(source_incident_chat_id)
    if not targets:
        return
    card = _cards.build_meeting_link_cancelled_card(
        priority=priority,
        duration_text=duration_text,
        meeting_no=meeting_no,
        reason=reason,
        emergency_topic=emergency_topic,
        update_multi=False,
    )
    prompt = (prompt_chat_id or "").strip()
    for oc in targets:
        if oc == prompt:
            continue
        try:
            st, body, _ = _lark.post_card_to_chat(oc, token, card)
            ok, code, msg = _lark.lark_im_message_create_ok(body)
            if st == 200 and ok:
                log.info(
                    "cancel_p0: meeting cancelled fan-out ok chat_id_tail=%s source=%s",
                    oc[-12:] if len(oc) > 12 else oc,
                    source_incident_chat_id[:24],
                )
            else:
                log.warning(
                    "cancel_p0: meeting cancelled fan-out HTTP=%s lark_code=%s chat=%s msg=%r",
                    st,
                    code,
                    oc[:24],
                    msg,
                )
        except Exception as e:
            log.warning(
                "cancel_p0: meeting cancelled fan-out exception chat=%s err=%s",
                oc[:24],
                e,
            )


def start_p0(
    chat_id: str,
    token: str,
    trigger_open_id: str,
    priority: str = "P0",
    source_chat_name: str = "",
    trigger_lark_user_id: str = "",
    silent_when_blocked: bool = False,
    vc_ring_target_open_ids: Optional[List[str]] = None,
    issue_watch_alert_key: str = "",
) -> None:
    """
    Create a new P0/P1 VC meeting session.

    ``silent_when_blocked`` — when True, do NOT post visible warnings to the source
    chat if blocked by an active session or cooldown. Just log and return. Use this
    from heuristic / keyword auto-trigger paths to avoid noisy false-positives in
    production incident groups (e.g. when someone re-pastes an overview template).
    Explicit user actions (P0 thread confirm, P1 confirm Yes) should leave this
    False so users get a clear reason why nothing happened.
    """
    from . import participants as _participants

    _config.reload_env_runtime()
    chat_id = (chat_id or "").strip()
    trigger_open_id = (trigger_open_id or "").strip()
    trigger_lark_user_id = (trigger_lark_user_id or "").strip()
    priority = (priority or "P0").strip().upper()
    if not chat_id:
        return
    pop_p1_prompt_pending(chat_id)
    _clear_last_ended_snapshot(chat_id)
    # Bot warnings during start: same chat as meeting cards (incident vs mirror — see config).
    notify_chat = _config.get_session_meeting_card_post_chat_id(chat_id)
    with _session_disk.exclusive_lock(chat_id) as _lock_ok:
        if not _lock_ok:
            # Could not acquire within the bounded deadline — another declare is already
            # in progress for this chat. Skip rather than proceed unlocked or double-declare.
            log.info("start_p0: skipped — another declare already in progress for chat_id=%s", chat_id)
            return
        if P0_SESSIONS.get(chat_id):
            if silent_when_blocked:
                log.info("start_p0: blocked (session already active) silent chat_id=%s", chat_id)
                return
            _lark.post_text_to_chat(
                notify_chat,
                token,
                "ℹ️ A P0/P1 meeting session is already active in this group. Use **end meeting** or **cancel meeting** before declaring again.",
            )
            return
        sd = _session_disk.load_session(chat_id)
        if sd:
            P0_SESSIONS[chat_id] = sd
            if silent_when_blocked:
                log.info("start_p0: blocked (session loaded from disk) silent chat_id=%s", chat_id)
                return
            _lark.post_text_to_chat(
                notify_chat,
                token,
                "ℹ️ A P0/P1 meeting session is already active in this group. Use **end meeting** or **cancel meeting** before declaring again.",
            )
            return
        if p0_cooldown(chat_id):
            if silent_when_blocked:
                log.info("start_p0: blocked (cooldown active) silent chat_id=%s", chat_id)
                return
            total_min = max(1, (P0_COOLDOWN_SEC + 59) // 60)
            mins_label = "minute" if total_min == 1 else "minutes"
            msg = f"⚠️ Meeting was just created earlier — try again after {total_min} {mins_label}."
            _lark.post_text_to_chat(notify_chat, token, msg)
            return
        now = int(time.time())
        emergency_topic = _config.get_emergency_topic_for_source_chat(chat_id)
        vc_meeting_topic = _config.get_vc_meeting_topic_for_source_chat(chat_id)
        vc = _lark.create_vc_reserve(token, meeting_topic=vc_meeting_topic)
        link = (vc.get("link") or "").strip()
        if not link:
            _lark.post_text_to_chat(notify_chat, token, "❌ Failed to create Lark VC meeting (reserve/apply).")
            return
        target_chat = _config.get_session_meeting_card_post_chat_id(chat_id)
        chat_label = (source_chat_name or "").strip()
        if not chat_label:
            chat_label = _lark.get_group_chat_name(chat_id, token)
        affected_players = ""
        P0_SESSIONS[chat_id] = {
            "priority": priority,
            "start_epoch": now,
            "link": link,
            "reserve_id": vc.get("reserve_id", ""),
            "meeting_no": vc.get("meeting_no", ""),
            "meeting_id": vc.get("meeting_id", ""),
            "trigger_open_id": trigger_open_id,
            "trigger_lark_user_id": trigger_lark_user_id,
            "source_chat": chat_id,
            "target_chat": target_chat,
            "source_chat_name": chat_label,
            "emergency_topic": emergency_topic,
            "participants": [],
            "affected_players": affected_players,
            "vc_external_join_count": 0,
            "vc_ring_target_open_ids": list(vc_ring_target_open_ids or []),
            "vc_ring_invited_open_ids": [],
            "vc_ring_done": False,
        }
        if trigger_open_id:
            try:
                host_label = _lark.lookup_user_name_by_open_id(token, trigger_open_id)
                if not host_label:
                    host_label = f"Host ({trigger_open_id[-6:]})"
                _participants.add_meeting_participant(host_label)
                log.info("Seeded host participant=%s for chat_id=%s", host_label, chat_id)
            except Exception as e:
                log.warning("Failed seeding fallback host participant open_id=%s err=%s", trigger_open_id, e)
        log.info("start session created priority=%s source_chat=%s target_chat=%s trigger_open_id=%s", priority, chat_id, target_chat, trigger_open_id)
        meeting_no = str(vc.get("meeting_no", "")).strip()
        invite_mid = _post_meeting_link_unfurl_notice(
            target_chat,
            token,
            link=link,
            priority=priority,
            emergency_topic=emergency_topic,
        )
        if not invite_mid:
            log.error(
                "start_p0: meeting unfurl notice failed target_tail=%s",
                target_chat[-12:] if len(target_chat) > 12 else target_chat,
            )
            fallback_text = _cards.build_p0_meeting_created_text(
                link, priority=priority, emergency_topic=emergency_topic
            )
            st_fb, body_fb = _lark.post_text_to_chat(target_chat, token, fallback_text)
            ok_fb, _, lark_msg = _lark.lark_im_message_create_ok(body_fb)
            invite_mid = _lark.parse_im_message_id_from_response(body_fb) if ok_fb else ""
            if st_fb != 200 or not ok_fb:
                st_fb2, body_fb2 = _lark.post_text_to_chat(chat_id, token, fallback_text)
                ok_fb2, _, lark_msg2 = _lark.lark_im_message_create_ok(body_fb2)
                if st_fb2 != 200 or not ok_fb2:
                    _lark.post_text_to_chat(
                        notify_chat,
                        token,
                        "❌ Failed to post meeting link. Check bot is in the group "
                        f"(target_tail={target_chat[-12:] if len(target_chat) > 12 else target_chat}). "
                        f"Lark: {lark_msg or lark_msg2 or 'unknown'}",
                    )
                    P0_SESSIONS.pop(chat_id, None)
                    return
                invite_mid = _lark.parse_im_message_id_from_response(body_fb2)
                log.warning(
                    "start_p0: meeting unfurl failed but text fallback ok in source chat_tail=%s",
                    chat_id[-12:] if len(chat_id) > 12 else chat_id,
                )
            else:
                log.warning(
                    "start_p0: meeting unfurl failed but text fallback ok target_tail=%s",
                    target_chat[-12:] if len(target_chat) > 12 else target_chat,
                )
        if invite_mid:
            P0_SESSIONS[chat_id]["meeting_invite_message_id"] = invite_mid
            P0_SESSIONS[chat_id]["meeting_invite_notice_kind"] = "text_unfurl"
        if _session_disk.enabled():
            _session_disk.save_session(chat_id, P0_SESSIONS[chat_id])
    # Fan out the meeting-created notice to mirror groups OUTSIDE the exclusive lock —
    # it posts to N chats (serial, bounded HTTP each) and does not gate the "session
    # already active" decision, so it must not extend the per-chat lock hold.
    if priority == "P0":
        try:
            _fanout_p0_meeting_created_link_notice(
                token,
                chat_id,
                target_chat,
                link=link,
                priority=priority,
                emergency_topic=emergency_topic,
            )
        except Exception as e:
            log.warning("start_p0: meeting-created fanout failed: %s", e)
    dm_targets = _dm_instruction_targets(trigger_open_id)
    log.info(
        "start_p0 DM targets count=%s open_ids=%s (API expects open_id ou_..., not user_id gceda344-style)",
        len([x for x in dm_targets if (x or "").strip()]),
        [x for x in dm_targets if (x or "").strip()],
    )
    dm_targets_list = [x for x in dm_targets if (x or "").strip()]
    issue_watch_key = (issue_watch_alert_key or "").strip()
    # The two remaining SYNCHRONOUS pieces — the Bitable ops/deploy cards and the DM
    # instruction/overview posts — only send messages, hold no lock, and their results
    # are not consumed here. Run them OFF the declare thread so a slow Lark/Bitable call
    # can never stall the declare. (Grafana + the schedulers below are already
    # non-blocking — each spawns its own daemon thread.)
    def _post_declare_finalize() -> None:
        # Bitable ops/deploy cards — own per-session dedupe (_session_bitable_already_posted),
        # no lock, result unused → safe to background.
        if priority == "P0":
            try:
                from features.overview import bitable_adjustments as _bitable_adj

                log.info("start_p0: running adjustment bitable on P0 declare chat_tail=%s", chat_id[-12:] if len(chat_id) > 12 else chat_id)
                _bitable_adj.maybe_post_adjustment_notice_on_p0_declare(
                    token,
                    source_chat_id=chat_id,
                    priority=priority,
                )
            except Exception as e:
                log.warning("start_p0: adjustment bitable on declare failed: %s", e)
        # Auto overview preview only when P0 is declared from Issue Watch DM (explicit alert_key).
        # Typed p0 / thread confirm always get the green Build overview card.
        if (
            issue_watch_key
            and priority == "P0"
            and _config.get_p0_issue_watch_auto_overview_enabled()
        ):
            log.info(
                "start_p0: Issue Watch suggested overview on declare chat_id=%s alert_key=%s",
                chat_id[:24],
                issue_watch_key[:12],
            )
        for oid in dm_targets:
            if not oid:
                continue
            dm_item: Dict[str, Any] = {
                "chat_id": chat_id,
                "target_chat": target_chat,
                "priority": priority,
                "label": chat_label,
            }
            if oid == trigger_open_id and trigger_lark_user_id:
                dm_item["operator_lark_user_id"] = trigger_lark_user_id
            if issue_watch_key:
                enqueue_dm_issue_watch_overview_if_needed(oid, token, dm_item, issue_watch_key)
            else:
                enqueue_dm_instruction_if_needed(oid, token, dm_item)

    threading.Thread(
        target=_post_declare_finalize, name="p0-declare-finalize", daemon=True
    ).start()
    try:
        from features.screenshot.graph_screenshot import schedule_p0_graph_screenshot

        schedule_p0_graph_screenshot(token, priority, chat_label)
    except Exception as e:
        log.warning("start_p0: graph screenshot hook failed: %s", e)
    try:
        schedule_vc_auto_cancel_if_no_external_joins(chat_id)
    except Exception as e:
        log.warning("start_p0: schedule vc auto-cancel failed: %s", e)
    if priority == "P0":
        try:
            schedule_p0_ongoing_dm_buzz(chat_id, trigger_open_id)
        except Exception as e:
            log.warning("start_p0: schedule p0 ongoing DM buzz failed: %s", e)


def _dm_instruction_item_from_session(chat_id: str, sess: Dict[str, Any]) -> Dict[str, Any]:
    pr = str(sess.get("priority") or "P0").strip().upper()
    if pr not in ("P0", "P1"):
        pr = "P0"
    return {
        "chat_id": chat_id,
        "target_chat": str(sess.get("target_chat") or "").strip(),
        "priority": pr,
        "label": str(sess.get("source_chat_name") or "").strip(),
    }


def _flush_deferred_dm_instruction_for_incident(chat_id: str) -> None:
    """Legacy: post the green DM if ``dm_instruction_deferred`` is still set (older sessions / cancel+end flush)."""
    cid = (chat_id or "").strip()
    if not cid.startswith("oc_"):
        return
    tok = _lark.get_tenant_token_primary()
    if not tok:
        log.warning("_flush_deferred_dm_instruction_for_incident: no primary tenant token chat_id=%s", cid)
        return
    sess = P0_SESSIONS.get(cid)
    if not sess or not sess.get("dm_instruction_deferred"):
        return
    sess["dm_instruction_deferred"] = False
    if _session_disk.enabled():
        _session_disk.save_session(cid, sess)
    item = _dm_instruction_item_from_session(cid, sess)
    trigger = str(sess.get("trigger_open_id") or "").strip()
    for oid in _dm_instruction_targets(trigger):
        if oid:
            enqueue_dm_instruction_if_needed(oid, tok, item)
