"""
P0 logic configuration: env reload, timeouts, API bases, regex patterns, timing constants.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import logging

log = logging.getLogger("lark-ops-ai")

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_env_file_path() -> str:
    """
    Primary env file for logging / legacy single-file mode.

    Dev overlay mode (``ENV_PROFILE=dev``) uses ``resolve_env_layer_paths()`` instead.
    """
    raw = (os.getenv("ENV_PATH") or "").strip()
    if raw:
        return raw
    p = _REPO_ROOT / ".env"
    if p.is_file():
        return str(p)
    return "/home/ubuntu/lark-ops-ai/.env"


def resolve_env_layer_paths() -> List[str]:
    """
    Env files to merge in order (later wins).

    **Dev:** ``ENV_PROFILE=dev`` → ``.env`` (secrets) + ``.env.dev`` (routing overrides).
    Optional ``ENV_OVERLAY=/path`` adds a second file on top of ``.env`` (any profile).

    **Prod / server:** single file from ``ENV_PATH`` or repo ``.env`` (unchanged).
    """
    overlay = (os.getenv("ENV_OVERLAY") or "").strip()
    profile = (os.getenv("ENV_PROFILE") or "").strip().lower()
    base = _REPO_ROOT / ".env"
    if profile == "dev" or overlay:
        paths: List[str] = []
        if base.is_file():
            paths.append(str(base))
        ov = overlay or str(_REPO_ROOT / ".env.dev")
        if Path(ov).is_file() and ov not in paths:
            paths.append(ov)
        if paths:
            return paths
    return [resolve_env_file_path()]


def _merge_dotenv_files_into_environ(paths: List[str]) -> None:
    if not dotenv_values:
        return
    merged: Dict[str, str] = {}
    for path in paths:
        try:
            values = dotenv_values(path)
            for k, v in (values or {}).items():
                if v is None:
                    continue
                sv = str(v).strip()
                if sv:
                    merged[k] = sv
        except Exception as e:
            log.error("Failed to read env file %s: %s", path, e)
    for k, v in merged.items():
        os.environ[k] = v


def apply_env_layers() -> List[str]:
    """Load env file(s) into ``os.environ``. Returns paths applied."""
    paths = resolve_env_layer_paths()
    _merge_dotenv_files_into_environ(paths)
    return paths


ENV_PATH = resolve_env_file_path()


def reload_env_runtime() -> None:
    try:
        apply_env_layers()
    except Exception as e:
        log.error("Failed to reload .env: %s", e)


# Default incident group if env unset (matches historical lark_logic default).
_DEFAULT_INCIDENT_GROUP_FALLBACK = "oc_f4e833c6744e55eb50dfcd8830fa913e"


def get_incident_group_chat_ids() -> FrozenSet[str]:
    """
    All group chat ids (oc_...) where P0/P1 keywords are handled.

    Set either:
    - ``INCIDENT_GROUP_IDS=oc_a,oc_b`` (preferred for multiple), or
    - ``INCIDENT_GROUP_ID=oc_a`` or ``INCIDENT_GROUP_ID=oc_a,oc_b`` (comma-separated).
    """
    reload_env_runtime()
    raw = (os.getenv("INCIDENT_GROUP_IDS") or "").strip()
    if not raw:
        raw = (os.getenv("INCIDENT_GROUP_ID") or "").strip()
    if not raw:
        return frozenset({_DEFAULT_INCIDENT_GROUP_FALLBACK})
    out = frozenset(x.strip() for x in raw.split(",") if x.strip())
    for x in out:
        if not x.startswith("oc_"):
            log.warning(
                "INCIDENT_GROUP_IDS has invalid entry %r — expect full Lark group ids (oc_...). "
                "Check for duplicate INCIDENT_GROUP_IDS / INCIDENT_GROUP_ID lines or a truncated value.",
                x,
            )
    return out


def get_overview_post_chat_id() -> str:
    """
    If set, \"Send overview\" posts to this oc_ chat; otherwise posts to the group
    where the session started (per-chat_id session).

    Per-detection-group routing takes precedence when ``INCIDENT_OVERVIEW_TARGET_MAP`` is set
    (see ``get_overview_target_chat_id_for_source_incident``).
    """
    reload_env_runtime()
    return (os.getenv("OVERVIEW_TARGET_GROUP_CHAT_ID") or os.getenv("P0_OVERVIEW_POST_CHAT_ID") or "").strip()


def _parse_incident_overview_target_map(raw: str) -> Dict[str, str]:
    """
    Comma-separated ``oc_detection=oc_prompt`` pairs (both sides must be ``oc_...`` group chat ids).
    """
    out: Dict[str, str] = {}
    if not (raw or "").strip():
        return out
    for segment in raw.split(","):
        segment = segment.strip()
        if "=" not in segment:
            continue
        key, _, val = segment.partition("=")
        key, val = key.strip(), val.strip()
        if key.startswith("oc_") and val.startswith("oc_"):
            # Reject "=oc_" placeholders (copy-paste cut off); full Lark group ids are longer.
            if len(key) < 12 or len(val) < 12:
                log.warning(
                    "INCIDENT_OVERVIEW_TARGET_MAP: skip incomplete pair %r "
                    "(each side must be a full oc_... group chat id, e.g. oc_8c1c...=oc_f4e8...)",
                    segment,
                )
                continue
            out[key] = val
    return out


def _parse_oc_chat_id_csv(raw: str) -> List[str]:
    """Comma-separated ``oc_...`` group chat ids (deduped, invalid entries skipped)."""
    out: List[str] = []
    seen: set[str] = set()
    for part in (raw or "").split(","):
        p = part.strip()
        if not p.startswith("oc_") or len(p) < 12 or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def get_p0_notification_hub_chat_ids() -> List[str]:
    """
    Shared hub group(s) for **both** plain P0 join text and recording-available text.

    ``P0_NOTIFICATION_HUB_CHAT_IDS`` (alias ``P0_HUB_CHAT_IDS``) — comma-separated ``oc_...``.
    Merged with ``P0_MEETING_CREATED_TEXT_CHAT_IDS`` and ``VC_RECORDING_FANOUT_CHAT_IDS``.
    """
    reload_env_runtime()
    raw = (
        os.getenv("P0_NOTIFICATION_HUB_CHAT_IDS") or os.getenv("P0_HUB_CHAT_IDS") or ""
    ).strip()
    return _parse_oc_chat_id_csv(raw)


def get_p0_monitoring_chat_ids() -> List[str]:
    """
    ``P0_MONITORING_CHAT_IDS`` — ops monitoring group(s) for duty-warning mirrors and log alerts.
    Comma-separated ``oc_...``. Bot must be in each group. Empty = monitoring off.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_MONITORING_CHAT_IDS") or "").strip()
    return _parse_oc_chat_id_csv(raw)


def _env_scalar(raw: str) -> str:
    """Strip whitespace and inline ``#`` comments (safe when ``os.getenv`` bypasses dotenv)."""
    s = (raw or "").strip()
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    return s


def _env_flag_on(name: str, default: str = "0") -> bool:
    v = _env_scalar(os.getenv(name) or default).lower()
    return v in ("1", "true", "yes", "on")


def p0_monitoring_duty_warnings_enabled() -> bool:
    """Mirror duty DM warnings (e.g. overview send blocked) to monitoring GC. Default on when chat IDs set."""
    reload_env_runtime()
    if not get_p0_monitoring_chat_ids():
        return False
    return _env_flag_on("P0_MONITORING_DUTY_WARNINGS", "1")


def p0_monitoring_log_alerts_enabled() -> bool:
    """Post ERROR+ log lines to monitoring GC. Default on when chat IDs set."""
    reload_env_runtime()
    if not get_p0_monitoring_chat_ids():
        return False
    return _env_flag_on("P0_MONITORING_LOG_ALERTS", "1")


def get_p0_monitoring_log_min_level() -> int:
    """``P0_MONITORING_LOG_MIN_LEVEL`` — ERROR (default), WARNING, or CRITICAL."""
    reload_env_runtime()
    raw = _env_scalar(os.getenv("P0_MONITORING_LOG_MIN_LEVEL") or "ERROR").upper()
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
    }.get(raw, logging.ERROR)


def get_p0_monitoring_alert_cooldown_sec() -> int:
    """Dedupe identical monitoring alerts (default 120s). ``0`` = no dedupe."""
    reload_env_runtime()
    raw = (os.getenv("P0_MONITORING_ALERT_COOLDOWN_SEC") or "120").strip()
    try:
        n = int(raw)
        return max(0, min(n, 3600))
    except ValueError:
        return 120


def get_incident_overview_target_map() -> Dict[str, str]:
    """Parsed ``INCIDENT_OVERVIEW_TARGET_MAP`` env (detection ``oc_`` -> mirror ``oc_`` for meeting cards when split)."""
    reload_env_runtime()
    if p0_single_incident_group_mode():
        return {}
    return _parse_incident_overview_target_map(os.getenv("INCIDENT_OVERVIEW_TARGET_MAP") or "")


def p0_single_incident_group_mode() -> bool:
    """
    ``P0_SINGLE_INCIDENT_GROUP=1`` — one Lark group for P0 (no detection/prompt split).

    Meeting cards, overview, ended/cancelled, and typed commands all stay in ``INCIDENT_GROUP_IDS``.
    ``INCIDENT_OVERVIEW_TARGET_MAP`` and global ``P0_OVERVIEW_POST_CHAT_ID`` are ignored.
    """
    reload_env_runtime()
    v = (os.getenv("P0_SINGLE_INCIDENT_GROUP") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_incident_overview_send_map() -> Dict[str, str]:
    """
    Parsed ``INCIDENT_OVERVIEW_SEND_MAP``: ``source_incident_oc=overview_destination_oc``.

    Used **only** for the final **Send overview** Lark post (see ``handlers.send_preview``).
    Meeting invite / P1 cards follow ``get_session_meeting_card_post_chat_id`` instead.
    """
    reload_env_runtime()
    return _parse_incident_overview_target_map(os.getenv("INCIDENT_OVERVIEW_SEND_MAP") or "")


def get_incident_overview_send_chat_id(source_incident_chat_id: str) -> str:
    """Resolved overview **post** group for ``source_incident`` from ``INCIDENT_OVERVIEW_SEND_MAP``; empty if unmapped."""
    sid = (source_incident_chat_id or "").strip()
    if not sid:
        return ""
    m = get_incident_overview_send_map()
    if sid in m:
        return m[sid]
    return ""


def get_source_incident_chat_id_for_mirror_target(mirror_chat_id: str) -> str:
    """
    Reverse of ``INCIDENT_OVERVIEW_TARGET_MAP``: given prompt / mirror ``oc_``, return the detection ``oc_``
    key when that mapping is **unique**. Used when DM preview lost ``source_incident_chat_id`` but
    ``target_chat`` (session prompt) is known — so **Send overview** can still post to detection when
    ``P0_OVERVIEW_POST_TO_INCIDENT_SOURCE_CHAT=1``.
    """
    mc = (mirror_chat_id or "").strip()
    if not mc.startswith("oc_"):
        return ""
    m = get_incident_overview_target_map()
    hits = [det for det, pr in m.items() if pr == mc]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        log.warning(
            "INCIDENT_OVERVIEW_TARGET_MAP: multiple detection chats map to the same prompt %s — "
            "cannot infer unique overview source",
            mc[:28],
        )
    return ""


def get_p0_overview_post_to_source_incident_chat() -> bool:
    """
    ``P0_OVERVIEW_POST_TO_INCIDENT_SOURCE_CHAT`` — if ``1``, **Send overview** posts to the **detection**
    group (where P0 was declared: ``source_incident_chat_id``), not the prompt / mirror session ``target_chat``.

    Use with ``P0_MEETING_CARDS_IN_SOURCE_INCIDENT_CHAT=0`` and ``INCIDENT_OVERVIEW_TARGET_MAP`` so meeting
    cards, ended/cancelled, and cooldown bot text stay in the **prompt** group.

    If ``INCIDENT_OVERVIEW_SEND_MAP`` has an entry for this incident ``oc_``, that map still **wins** (broadcast override).
    """
    reload_env_runtime()
    if p0_single_incident_group_mode():
        return True
    v = (os.getenv("P0_OVERVIEW_POST_TO_INCIDENT_SOURCE_CHAT") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_overview_detection_fanout_chat_ids() -> List[str]:
    """
    Extra Lark group chat ids (``oc_...``) that receive a **duplicate** overview card when the primary
    post lands in a **detection** group (see ``is_overview_post_destination_detection``).

    Comma-separated ``OVERVIEW_DETECTION_FANOUT_CHAT_IDS`` (alias ``P0_OVERVIEW_DETECTION_FANOUT_CHAT_IDS``).
    Bot must be a member of each group. Empty = disabled.
    """
    reload_env_runtime()
    raw = (
        os.getenv("OVERVIEW_DETECTION_FANOUT_CHAT_IDS") or os.getenv("P0_OVERVIEW_DETECTION_FANOUT_CHAT_IDS") or ""
    ).strip()
    if not raw:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        p = part.strip()
        if not p.startswith("oc_") or len(p) < 12:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def get_p0_meeting_created_text_map() -> Dict[str, str]:
    """
    ``P0_MEETING_CREATED_TEXT_MAP`` — per detection ``oc_`` → one extra group that gets the plain
    ``🚨 P0 meeting created`` text (red meeting card still posts to the prompt group as before).
    """
    reload_env_runtime()
    return _parse_incident_overview_target_map(os.getenv("P0_MEETING_CREATED_TEXT_MAP") or "")


def get_p0_meeting_created_text_fanout_chat_ids(source_incident_chat_id: str) -> List[str]:
    """
    Lark groups that receive the plain P0 meeting-created text in **addition** to the red card in
    the prompt group. Per-detection ``P0_MEETING_CREATED_TEXT_MAP`` wins; else global
    ``P0_MEETING_CREATED_TEXT_CHAT_IDS`` (alias ``P0_MEETING_ALERT_TEXT_CHAT_IDS``).
    """
    reload_env_runtime()
    sid = (source_incident_chat_id or "").strip()
    out: List[str] = []
    seen: set[str] = set()

    def _add(cid: str) -> None:
        c = (cid or "").strip()
        if not c.startswith("oc_") or len(c) < 12 or c in seen:
            return
        seen.add(c)
        out.append(c)

    if sid:
        mapped = get_p0_meeting_created_text_map().get(sid, "")
        if mapped:
            _add(mapped)
    raw = (
        os.getenv("P0_MEETING_CREATED_TEXT_CHAT_IDS")
        or os.getenv("P0_MEETING_ALERT_TEXT_CHAT_IDS")
        or ""
    ).strip()
    for part in raw.split(","):
        _add(part.strip())
    for hub in get_p0_notification_hub_chat_ids():
        _add(hub)
    return out


def p0_meeting_cancelled_fanout_enabled() -> bool:
    """``P0_MEETING_CANCELLED_FANOUT_ENABLED`` — notify extra groups when a meeting is cancelled (default ``1``)."""
    reload_env_runtime()
    v = (os.getenv("P0_MEETING_CANCELLED_FANOUT_ENABLED") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_meeting_cancelled_fanout_chat_ids(source_incident_chat_id: str) -> List[str]:
    """
    Lark groups that receive the **meeting cancelled** card when a session is cancelled
    (manual cancel or auto-cancel). Falls back to ``get_p0_meeting_created_text_fanout_chat_ids``
    when ``P0_MEETING_CANCELLED_FANOUT_CHAT_IDS`` is unset (same boss / hub groups as join alert).
    """
    reload_env_runtime()
    if not p0_meeting_cancelled_fanout_enabled():
        return []
    sid = (source_incident_chat_id or "").strip()
    out: List[str] = []
    seen: set[str] = set()

    def _add(cid: str) -> None:
        c = (cid or "").strip()
        if not c.startswith("oc_") or len(c) < 12 or c in seen:
            return
        seen.add(c)
        out.append(c)

    raw = (
        os.getenv("P0_MEETING_CANCELLED_FANOUT_CHAT_IDS")
        or os.getenv("P0_MEETING_CANCELLED_TEXT_CHAT_IDS")
        or ""
    ).strip()
    if raw:
        for part in raw.split(","):
            _add(part.strip())
        return out
    return get_p0_meeting_created_text_fanout_chat_ids(sid)


def is_overview_post_destination_detection(
    dest_chat_id: str, source_incident_chat_id: str, session_target_chat: str
) -> bool:
    """
    True when ``dest_chat_id`` is a **detection** room (source-side incident group), i.e. the same
    overview routing users would call “post to detection”.

    - If ``INCIDENT_OVERVIEW_TARGET_MAP`` is set: ``dest`` is a **key** (detection ``oc_``).
    - Else: ``dest`` equals the resolved source incident ``oc_`` and that id is in ``INCIDENT_GROUP_IDS``.
    """
    d = (dest_chat_id or "").strip()
    if not d.startswith("oc_"):
        return False
    sid = (source_incident_chat_id or "").strip()
    tc = (session_target_chat or "").strip()
    if not sid.startswith("oc_") and tc.startswith("oc_"):
        sid = get_source_incident_chat_id_for_mirror_target(tc) or sid
    m = get_incident_overview_target_map()
    if m:
        return d in m
    if sid.startswith("oc_") and d == sid and sid in get_incident_group_chat_ids():
        return True
    # DM ``create overview …`` uses a non-oc_ placeholder source; final dest equals ``target_chat``.
    if (
        (not sid.startswith("oc_"))
        and d == tc
        and tc.startswith("oc_")
        and tc in get_incident_group_chat_ids()
    ):
        return True
    return False


def get_vc_recording_fanout_chat_ids() -> List[str]:
    """
    When Lark emits **vc.meeting.recording_ready_v1** (Open API reserves only), post the recording link
    to these ``oc_`` group chats. Comma-separated ``VC_RECORDING_FANOUT_CHAT_IDS`` (alias
    ``P0_VC_RECORDING_FANOUT_CHAT_IDS``). Bot must be in each group. Empty = disabled.
    """
    reload_env_runtime()
    raw = (
        os.getenv("VC_RECORDING_FANOUT_CHAT_IDS") or os.getenv("P0_VC_RECORDING_FANOUT_CHAT_IDS") or ""
    ).strip()
    out = _parse_oc_chat_id_csv(raw)
    seen = set(out)
    for hub in get_p0_notification_hub_chat_ids():
        if hub not in seen:
            seen.add(hub)
            out.append(hub)
    return out


def get_vc_recording_fanout_user_open_ids() -> List[str]:
    """
    Users (``ou_...``) who get **view** on the cloud recording via
    ``PATCH .../recording/set_permission`` (**Feishu type = 1**, user authorization).

    Comma-separated ``VC_RECORDING_FANOUT_USER_OPEN_IDS`` (alias ``P0_VC_RECORDING_FANOUT_USER_OPEN_IDS``).
    Same **vc:record** / token caveats as group fan-out. Bot also DMs them the recording message when
    configured (including **users-only** fan-out with no ``VC_RECORDING_FANOUT_CHAT_IDS``).
    """
    reload_env_runtime()
    raw = (
        os.getenv("VC_RECORDING_FANOUT_USER_OPEN_IDS")
        or os.getenv("P0_VC_RECORDING_FANOUT_USER_OPEN_IDS")
        or ""
    ).strip()
    if not raw:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        p = part.strip()
        if not is_open_id(p) or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def get_vc_recording_fanout_topic_substring_filter() -> str:
    """
    If non-empty, only forward **recording_ready** when the meeting topic contains this substring
    (case-insensitive). Reduces noise if the same app creates other API meetings.

    Env: ``VC_RECORDING_FANOUT_TOPIC_SUBSTRING`` (e.g. ``Video meeting`` matching ``VIDEO_MEETING_TOPIC_PREFIX``).
    """
    reload_env_runtime()
    return (os.getenv("VC_RECORDING_FANOUT_TOPIC_SUBSTRING") or "").strip()


def get_vc_recording_fanout_set_permission_enabled() -> bool:
    """
    When true (default), after a recording URL exists, call **set_permission** so each fan-out **group**
    (Feishu type=2) and each ``VC_RECORDING_FANOUT_USER_OPEN_IDS`` user (type=1) gets **view** on the file.

    Set ``VC_RECORDING_FANOUT_SET_PERMISSION=0`` to skip if your token cannot call that API.

    Requires **vc:record** (authorize/update recording), not only ``vc:record:readonly``.
    """
    reload_env_runtime()
    v = (os.getenv("VC_RECORDING_FANOUT_SET_PERMISSION") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def get_vc_recording_fanout_tenant_wide_view_enabled() -> bool:
    """
    If true, **set_permission** also adds Feishu **type=3** (tenant-wide view): anyone in the **same tenant**
    can open the Minutes/recording link — useful when **vc:record** exists only for **user** token and tenant
    token still succeeds for broad authorize, or when group-level type=2 is rejected.

    **Security:** whole org gets view access to that file. Env: ``VC_RECORDING_FANOUT_TENANT_WIDE_VIEW=1``.
    """
    reload_env_runtime()
    v = (os.getenv("VC_RECORDING_FANOUT_TENANT_WIDE_VIEW") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_vc_recording_fanout_drive_perm() -> str:
    """
    Optional Drive API collaborator role for Minutes after VC ``set_permission`` (view-only).

    ``VC_RECORDING_FANOUT_DRIVE_PERM``: ``edit`` or ``view`` to grant via Drive API; empty / ``0`` = skip.

    Requires duty **user_access_token** with ``docs:permission.member:create`` (+ ``update`` for edit).
    """
    reload_env_runtime()
    v = (os.getenv("VC_RECORDING_FANOUT_DRIVE_PERM") or "").strip().lower()
    if not v or v in ("0", "false", "no", "off"):
        return ""
    if v in ("view", "edit"):
        return v
    return ""


def get_vc_recording_fanout_plain_meta_enabled() -> bool:
    """
    After the recording **card**, also post a compact ``RECORDING_READY`` text line for downstream
    Minutes bots. Default **off** (card only). Set ``VC_RECORDING_FANOUT_PLAIN_META=1`` to enable.
    """
    v = (os.getenv("VC_RECORDING_FANOUT_PLAIN_META") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_lark_overview_post_chat_id_for_send(source_incident_chat_id: str, session_target_chat: str) -> str:
    """
    Lark chat_id for the final **Send overview** card.

    1. ``INCIDENT_OVERVIEW_SEND_MAP[source]`` if set (explicit broadcast destination).
    2. Else **detection** ``source`` if ``P0_OVERVIEW_POST_TO_INCIDENT_SOURCE_CHAT=1``.
    3. Else ``session_target_chat`` (prompt / mirror from session).

    If ``source_incident_chat_id`` is empty but ``session_target_chat`` is the prompt side of
    ``INCIDENT_OVERVIEW_TARGET_MAP``, reverse-resolve detection so overview can still land in the
    incident group.
    """
    sid = (source_incident_chat_id or "").strip()
    tc = (session_target_chat or "").strip()
    if not sid.startswith("oc_") and tc.startswith("oc_"):
        sid = get_source_incident_chat_id_for_mirror_target(tc) or sid
    mapped = get_incident_overview_send_chat_id(sid) if sid.startswith("oc_") else ""
    if mapped:
        return mapped
    if get_p0_overview_post_to_source_incident_chat() and sid.startswith("oc_"):
        return sid
    return tc


def lark_overview_forwarder_enabled() -> bool:
    """
    ``LARK_OVERVIEW_FORWARDER_ENABLED=1`` — broadcast overview (``INCIDENT_OVERVIEW_SEND_MAP`` destination)
    goes through ``lark-forwarder`` (overview-only bot webhook), not the primary bot.
    """
    reload_env_runtime()
    v = (os.getenv("LARK_OVERVIEW_FORWARDER_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_lark_overview_forwarder_url() -> str:
    """Base URL for ``lark-forwarder`` (e.g. ``http://127.0.0.1:8010``)."""
    reload_env_runtime()
    return (os.getenv("LARK_OVERVIEW_FORWARDER_URL") or "").strip().rstrip("/")


def get_lark_overview_forwarder_secret() -> str:
    """Optional shared secret sent as ``Authorization: Bearer …`` to ``lark-forwarder``."""
    reload_env_runtime()
    return (os.getenv("LARK_OVERVIEW_FORWARDER_SECRET") or "").strip()


def resolve_overview_send_routing(
    source_incident_chat_id: str, session_target_chat: str
) -> tuple[str, str, bool]:
    """
    Returns ``(primary_dest, broadcast_dest, use_forwarder)`` for **Send overview**.

    When forwarder is enabled and ``INCIDENT_OVERVIEW_SEND_MAP`` defines a broadcast ``oc_`` for this
    incident, ``primary_dest`` is the detection / prompt group (primary bot) and ``broadcast_dest`` is
    posted via ``lark-forwarder`` only.
    """
    sid = (source_incident_chat_id or "").strip()
    tc = (session_target_chat or "").strip()
    if not sid.startswith("oc_") and tc.startswith("oc_"):
        sid = get_source_incident_chat_id_for_mirror_target(tc) or sid
    broadcast = get_incident_overview_send_chat_id(sid) if sid.startswith("oc_") else ""
    use_forwarder = bool(
        lark_overview_forwarder_enabled()
        and get_lark_overview_forwarder_url()
        and broadcast.startswith("oc_")
    )
    if use_forwarder:
        if get_p0_overview_post_to_source_incident_chat() and sid.startswith("oc_"):
            primary = sid
        elif tc.startswith("oc_"):
            primary = tc
        else:
            primary = get_lark_overview_post_chat_id_for_send(sid, tc)
        return primary.strip(), broadcast.strip(), True
    primary = get_lark_overview_post_chat_id_for_send(sid, tc)
    return primary.strip(), "", False


def get_p0_meeting_cards_in_source_incident_chat() -> bool:
    """
    ``P0_MEETING_CARDS_IN_SOURCE_INCIDENT_CHAT`` — if ``1``, meeting invite / P1 confirm / end summaries
    post in the **incident** group (where P0 was declared), not the mirror ``INCIDENT_OVERVIEW_TARGET_MAP`` room.

    Use with ``INCIDENT_OVERVIEW_SEND_MAP`` so **Send overview** still lands in emergency / game rooms.
    """
    reload_env_runtime()
    v = (os.getenv("P0_MEETING_CARDS_IN_SOURCE_INCIDENT_CHAT") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_session_meeting_card_post_chat_id(source_incident_chat_id: str) -> str:
    """
    Lark group for **meeting** UX: invite card, P1 yes/no, ended/cancel notices, session ``target_chat``.

    Default: legacy mirror via ``get_overview_target_chat_id_for_source_incident`` (same as before).
    With ``P0_MEETING_CARDS_IN_SOURCE_INCIDENT_CHAT=1``: always ``source_incident_chat_id``.
    """
    sid = (source_incident_chat_id or "").strip()
    if not sid:
        return ""
    if p0_single_incident_group_mode():
        return sid
    if get_p0_meeting_cards_in_source_incident_chat():
        return sid
    return get_overview_target_chat_id_for_source_incident(sid) or sid


def get_overview_target_chat_id_for_source_incident(source_incident_chat_id: str) -> str:
    """
    Legacy **mirror** target for meeting cards + DM ``target_chat`` when
    ``P0_MEETING_CARDS_IN_SOURCE_INCIDENT_CHAT`` is off (default).

    Resolution order:

    1. ``INCIDENT_OVERVIEW_TARGET_MAP[source_incident_chat_id]`` if that detection ``oc_`` is listed.
    2. Else ``OVERVIEW_TARGET_GROUP_CHAT_ID`` / ``P0_OVERVIEW_POST_CHAT_ID`` if set (single global prompt group).
    3. Else ``source_incident_chat_id`` (cards and overview stay in the detection group).

    For **Send overview** only, see ``get_incident_overview_send_chat_id`` / ``INCIDENT_OVERVIEW_SEND_MAP``.
    """
    sid = (source_incident_chat_id or "").strip()
    if not sid:
        return ""
    if p0_single_incident_group_mode():
        return sid
    m = get_incident_overview_target_map()
    if sid in m:
        return m[sid]
    g = get_overview_post_chat_id()
    if g:
        return g
    return sid


def get_target_group_chat_id() -> str:
    """Backward-compatible alias: optional fixed overview destination (not incident routing)."""
    return get_overview_post_chat_id()


def get_dm_overview_target_chat_id() -> str:
    """
    Where DM drafts / \"Send overview\" attach when **no** active P0 session.

    Order: ``OVERVIEW_TARGET_GROUP_CHAT_ID`` / ``P0_OVERVIEW_POST_CHAT_ID`` if set;
    else if ``INCIDENT_OVERVIEW_TARGET_MAP`` has exactly one ``oc_=oc_`` pair, use the mirror-side ``oc_``;
    else if ``INCIDENT_OVERVIEW_SEND_MAP`` has exactly one pair, use the **send** destination ``oc_``;
    else if exactly **one** incident group is configured, use that ``oc_`` id (common single-group deploys);
    else empty (multiple groups — need env or a live session).
    """
    reload_env_runtime()
    env_id = get_overview_post_chat_id()
    if env_id:
        return env_id
    m = get_incident_overview_target_map()
    if len(m) == 1:
        return next(iter(m.values()))
    m_send = get_incident_overview_send_map()
    if len(m_send) == 1:
        return next(iter(m_send.values()))
    ids = list(get_incident_group_chat_ids())
    if len(ids) == 1:
        return ids[0]
    return ""


def _parse_standalone_overview_tags_env() -> Dict[str, str]:
    """
    ``P0_STANDALONE_OVERVIEW_TAGS`` / ``STANDALONE_OVERVIEW_TAGS``:
    ``emergency=oc_aaa,game=oc_bbb`` (comma-separated ``tag=oc_...``).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_STANDALONE_OVERVIEW_TAGS") or os.getenv("STANDALONE_OVERVIEW_TAGS") or "").strip()
    out: Dict[str, str] = {}
    if not raw:
        return out
    for seg in raw.split(","):
        seg = seg.strip()
        if "=" not in seg:
            continue
        k, _, v = seg.partition("=")
        k, v = k.strip().lower(), v.strip()
        if k in ("emergency", "game") and v.startswith("oc_"):
            out[k] = v
    return out


def get_standalone_overview_target_chat_id_for_tag(tag: str) -> str:
    """
    Resolve ``oc_`` for DM command ``create overview emergency|game`` (no live meeting).

    1. Explicit ``P0_STANDALONE_OVERVIEW_TAGS=emergency=oc_...,game=oc_...`` if set.
    2. Else match ``INCIDENT_GROUP_EMERGENCY_TOPICS`` labels: ``emergency`` → label contains
       ``emergency``; ``game`` → label contains ``game`` or ``游戏``.
    """
    t = (tag or "").strip().lower()
    if t not in ("emergency", "game"):
        return ""
    explicit = _parse_standalone_overview_tags_env()
    if t in explicit:
        return explicit[t]
    for oc_id in sorted(get_incident_group_chat_ids()):
        label = get_emergency_topic_for_source_chat(oc_id)
        lo = label.lower()
        if t == "emergency" and "emergency" in lo:
            return oc_id
        if t == "game" and ("game" in lo or "游戏" in label):
            return oc_id
    return ""


REQ_TIMEOUT_ENV = (os.getenv("REQ_TIMEOUT", "15") or "15").strip()
try:
    REQ_TIMEOUT = float(REQ_TIMEOUT_ENV)
except Exception:
    REQ_TIMEOUT = 15.0


def timeout_kw() -> Dict[str, Any]:
    return {} if REQ_TIMEOUT <= 0 else {"timeout": REQ_TIMEOUT}


# Timezone and meeting (zoneinfo is stdlib in 3.9+; use backport on 3.8)
from datetime import datetime  # noqa: E402

try:
    from zoneinfo import ZoneInfo  # type: ignore
except ImportError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore

PHT = ZoneInfo("Asia/Manila")
MEETING_TOPIC = (os.getenv("MEETING_TOPIC") or "CP-Emergency feedback紧急问题反馈群").strip()


def get_emergency_topic_for_source_chat(chat_id: str) -> str:
    """
    Bilingual suffix for ``🚨 P0 — …`` (meeting cards + VC topic), per incident group.

    ``INCIDENT_GROUP_EMERGENCY_TOPICS=oc_aaa=CP-Emergency feedback紧急问题反馈群,oc_bbb=Game urgent-游戏紧急群``

    Comma-separated; each segment is ``oc_...=topic text``. If no match, uses ``MEETING_TOPIC``.
    """
    reload_env_runtime()
    cid = (chat_id or "").strip()
    default = (os.getenv("MEETING_TOPIC") or "CP-Emergency feedback紧急问题反馈群").strip()
    if not cid:
        return default
    raw = (os.getenv("INCIDENT_GROUP_EMERGENCY_TOPICS") or "").strip()
    if not raw:
        return default
    for segment in raw.split(","):
        segment = segment.strip()
        if "=" not in segment:
            continue
        key, _, val = segment.partition("=")
        key, val = key.strip(), val.strip()
        if key == cid and val:
            return val
    return default


# Prefix for Lark VC / recorded meeting title (same string sent to ``create_vc_reserve``).
VIDEO_MEETING_TOPIC_PREFIX = (os.getenv("VIDEO_MEETING_TOPIC_PREFIX") or "Video meeting—").strip()


def get_vc_meeting_topic_for_source_chat(chat_id: str) -> str:
    """
    Topic string for Lark video conference reserve (shows on recorded / meeting UI).

    Format: ``{VIDEO_MEETING_TOPIC_PREFIX}{emergency label}``, e.g.
    ``Video meeting—CP-Emergency feedback紧急问题反馈群`` or
    ``Video meeting—Game urgent-游戏紧急群`` (from ``INCIDENT_GROUP_EMERGENCY_TOPICS`` / ``MEETING_TOPIC``).

    If the stored label already starts with ``video meeting`` (case-insensitive), it is returned unchanged.
    """
    reload_env_runtime()
    tail = get_emergency_topic_for_source_chat(chat_id).strip()
    if not tail:
        tail = (os.getenv("MEETING_TOPIC") or "CP-Emergency feedback紧急问题反馈群").strip()
    low = tail.lower()
    if low.startswith("video meeting"):
        return tail
    prefix = (os.getenv("VIDEO_MEETING_TOPIC_PREFIX") or VIDEO_MEETING_TOPIC_PREFIX).strip() or "Video meeting—"
    if not prefix.endswith("—") and not prefix.endswith("-"):
        prefix = prefix + "—"
    return f"{prefix}{tail}"


# Open Platform API root (…/open-apis). Default Singapore; override with LARK_OPEN_API_BASE if needed.
LARK_BASE = (os.getenv("LARK_OPEN_API_BASE") or "https://open-sg.larksuite.com/open-apis").strip().rstrip("/")
_LARK_GLOBAL_FALLBACK = "https://open.larksuite.com/open-apis"

SHEETS_BASES = [LARK_BASE, _LARK_GLOBAL_FALLBACK]
SHEETS_V2_BASES = SHEETS_BASES[:]

# VC / IM: primary = same host as tenant token (SG by default); fallback = global endpoint.
VC_BASES = [LARK_BASE, _LARK_GLOBAL_FALLBACK]
IM_BASES = [LARK_BASE, _LARK_GLOBAL_FALLBACK]

# Regex patterns
OPEN_ID_RE = re.compile(r"^ou_[A-Za-z0-9]+$")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ID_RE = re.compile(r"\b\d{6,}\b")
NOT_SPECIFIED_RE = re.compile(r"^\s*(not specified|n/?a|none|unknown|-)?\s*$", re.IGNORECASE)

CLEAR_RE = re.compile(r"^\s*(clear|reset|discard|cancel|cl)\s*$", re.IGNORECASE)
# Abort standalone ``coe`` / ``cog`` on the green DM card (before preview **Cancel**).
STANDALONE_OVERVIEW_ABORT_RE = re.compile(r"^\s*c\s*$", re.IGNORECASE)
STATUS_RE = re.compile(r"^\s*(status|draft|check|st)\s*$", re.IGNORECASE)
HELP_RE = re.compile(r"^\s*(help|commands|command\s+list|h)\s*$", re.IGNORECASE)

# DM whole line: ``create overview emergency|game`` or shortcuts ``coe`` / ``cog``.
_STANDALONE_OVERVIEW_LONG_RE = re.compile(
    r"^\s*create\s+overview\s+(emergency|game)\s*$",
    re.IGNORECASE,
)
_STANDALONE_OVERVIEW_SHORT_RE = re.compile(r"^\s*(coe|cog)\s*$", re.IGNORECASE)
# Back-compat alias for callers that still use ``.match()``.
STANDALONE_OVERVIEW_DM_RE = _STANDALONE_OVERVIEW_LONG_RE


def parse_standalone_overview_dm_command(cmd: str) -> Optional[str]:
    """Return ``emergency`` or ``game`` from long or short standalone-overview DM command."""
    s = (cmd or "").strip()
    if not s:
        return None
    m = _STANDALONE_OVERVIEW_LONG_RE.match(s)
    if m:
        return (m.group(1) or "").strip().lower()
    m = _STANDALONE_OVERVIEW_SHORT_RE.match(s)
    if m:
        return {"coe": "emergency", "cog": "game"}[(m.group(1) or "").strip().lower()]
    return None


WHO_IN_MEETING_RE = re.compile(
    r"^\s*(who\s+(is|are)\s+in\s+the\s+meeting|who\s+is\s+in\s+meeting|participants|list\s+participants|sino\s+nasa\s+meeting|pt|parts)\s*$",
    re.IGNORECASE,
)
IS_IN_MEETING_RE = re.compile(
    r"^\s*is\s+(.+?)\s+in\s+the\s+meeting\s*\??\s*$",
    re.IGNORECASE,
)

CountBuilder = Callable[[int], Tuple[Optional[int], str]]

PLAYER_COUNT_PATTERNS: List[Tuple[CountBuilder, re.Pattern[str]]] = [
    (lambda n: (n, f"Less than {n} affected players"), re.compile(r"\bless than\s+(\d+)\s+players?\b", re.IGNORECASE)),
    (lambda n: (n, f"more than {n} affected players"), re.compile(r"\bmore than\s+(\d+)\s+players?\b", re.IGNORECASE)),
    (lambda n: (n, f"more than {n} affected players"), re.compile(r"\bover\s+(\d+)\s+players?\b", re.IGNORECASE)),
    (lambda n: (n, f"at least {n} affected players"), re.compile(r"\bat least\s+(\d+)\s+players?\b", re.IGNORECASE)),
    (lambda n: (n, f"at least {n} affected players"), re.compile(r"\b(\d+)\s*\+\s*players?\b", re.IGNORECASE)),
    (lambda n: (n, f"{n} affected players"), re.compile(r"\b(\d+)\s+players?\b", re.IGNORECASE)),
    (lambda n: (n, f"{n} affected players"), re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{3,})\s+users?\b", re.IGNORECASE)),
]

PLAYER_VAGUE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("multiple", re.compile(r"\bmultiple players\b", re.IGNORECASE)),
    ("many", re.compile(r"\bmany players\b", re.IGNORECASE)),
    ("several", re.compile(r"\bseveral players\b", re.IGNORECASE)),
    ("many", re.compile(r"\blarge volume of chats from players\b", re.IGNORECASE)),
    ("many", re.compile(r"\bhigh volume of chats from players\b", re.IGNORECASE)),
    ("many", re.compile(r"\blarge volume of player reports\b", re.IGNORECASE)),
    ("multiple", re.compile(r"\bmultiple affected players\b", re.IGNORECASE)),
]

PLAYER_VAGUE_LABELS: Dict[str, str] = {
    "multiple": "Multiple affected players",
    "many": "Many affected players",
    "several": "Several affected players",
}

# Groq
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = (os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
GROQ_VISION_MODEL = (os.getenv("GROQ_VISION_MODEL") or "llama-3.2-11b-vision-preview").strip()
# One Groq call for issue EN + zh_issue + zh_impact (faster than summarize + 2 translates). Set 0 to use legacy path.
GROQ_OVERVIEW_ONE_SHOT = (os.getenv("GROQ_OVERVIEW_ONE_SHOT", "1") or "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

# Timing
AUTO_PREVIEW_DELAY_SEC = float((os.getenv("AUTO_PREVIEW_DELAY_SEC", "6") or "6").strip())
ONGOING_CARD_DELAY_SEC = int((os.getenv("ONGOING_CARD_DELAY_SEC", "600") or "600").strip())
P1_TO_P0_ESCALATION_SEC = int((os.getenv("P1_TO_P0_ESCALATION_SEC", "900") or "900").strip())


def get_p0_ongoing_dm_buzz_enabled() -> bool:
    """``P0_ONGOING_DM_BUZZ_ENABLED=1`` — DM operators after a P0 meeting runs past the delay (default on)."""
    reload_env_runtime()
    v = (os.getenv("P0_ONGOING_DM_BUZZ_ENABLED") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def get_p0_ongoing_dm_buzz_delay_sec() -> int:
    """
    Legacy alias for **minor** tier delay. Prefer ``get_p0_ongoing_dm_buzz_minor_delay_sec``.
    """
    return get_p0_ongoing_dm_buzz_minor_delay_sec()


def get_p0_ongoing_dm_buzz_major_delay_sec() -> int:
    """Seconds after P0 start before the **Major** DM buzz (default 300 = 5 minutes)."""
    reload_env_runtime()
    raw = (os.getenv("P0_ONGOING_DM_BUZZ_MAJOR_DELAY_SEC") or "300").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 300


def get_p0_ongoing_dm_buzz_minor_delay_sec() -> int:
    """
    Seconds after P0 start before the **Minor** DM buzz (default 600 = 10 minutes).
    ``P0_ONGOING_DM_BUZZ_DELAY_SEC`` overrides when ``P0_ONGOING_DM_BUZZ_MINOR_DELAY_SEC`` is unset.
    """
    reload_env_runtime()
    raw_minor = (os.getenv("P0_ONGOING_DM_BUZZ_MINOR_DELAY_SEC") or "").strip()
    if raw_minor:
        try:
            return max(60, int(raw_minor))
        except ValueError:
            pass
    raw_legacy = (os.getenv("P0_ONGOING_DM_BUZZ_DELAY_SEC") or "").strip()
    if raw_legacy:
        try:
            return max(60, int(raw_legacy))
        except ValueError:
            pass
    return max(60, ONGOING_CARD_DELAY_SEC)


def get_p0_ongoing_contact_names() -> str:
    """Who operators should contact (``P0_ONGOING_CONTACT_NAMES``, default Greg, Eason and Rock)."""
    reload_env_runtime()
    return (os.getenv("P0_ONGOING_CONTACT_NAMES") or "Greg, Eason and Rock").strip() or "Greg, Eason and Rock"


def get_p0_ongoing_lark_urgent_mode() -> str:
    """
    Lark native 加急 (buzz) on the DM card after it is sent.

    ``P0_ONGOING_LARK_URGENT_MODE``: ``app`` (in-app urgent, default), ``phone``, ``sms``, or ``off``.
    Requires the matching ``im:message.urgent*`` scopes on the primary Lark app.
    """
    reload_env_runtime()
    v = (os.getenv("P0_ONGOING_LARK_URGENT_MODE") or "app").strip().lower()
    if v in ("0", "false", "no", "off", "none", "disabled"):
        return "off"
    if v in ("app", "phone", "sms"):
        return v
    return "app"
P0_COOLDOWN_SEC = int((os.getenv("P0_COOLDOWN_SEC", "300") or "300").strip())
SUPPORT_MAP_TTL_SEC = int((os.getenv("SUPPORT_MAP_TTL_SEC", "600") or "600").strip())

# Lark VC ``reserves/apply``: ``end_time`` must be set for multi-person meetings; official cap ~30 days.
_VC_RESERVE_MAX_OFFSET_SEC = 30 * 24 * 60 * 60
_VC_RESERVE_MIN_OFFSET_SEC = 60 * 60


def get_vc_reserve_end_offset_sec() -> int:
    """
    Seconds from **now** until the reserve ``end_time`` sent to Lark (not the same as “call must hang up”).

    Default **30 days** — longest window Feishu documents for ``/vc/v1/reserves/apply`` (no fixed 2h cap).

    Env: ``P0_VC_RESERVE_END_OFFSET_SEC`` (integer seconds), clamped between 1 hour and 30 days.
    """
    reload_env_runtime()
    default = _VC_RESERVE_MAX_OFFSET_SEC
    raw = (os.getenv("P0_VC_RESERVE_END_OFFSET_SEC") or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        log.warning("Invalid P0_VC_RESERVE_END_OFFSET_SEC=%r — using default %s", raw, default)
        return default
    if v < _VC_RESERVE_MIN_OFFSET_SEC or v > _VC_RESERVE_MAX_OFFSET_SEC:
        log.warning(
            "P0_VC_RESERVE_END_OFFSET_SEC=%s clamped to [%s, %s]",
            v,
            _VC_RESERVE_MIN_OFFSET_SEC,
            _VC_RESERVE_MAX_OFFSET_SEC,
        )
    return max(_VC_RESERVE_MIN_OFFSET_SEC, min(v, _VC_RESERVE_MAX_OFFSET_SEC))


_VC_AUTO_CANCEL_NO_JOIN_MAX_SEC = 7 * 24 * 60 * 60


def get_p0_vc_auto_cancel_if_no_joins_sec() -> int:
    """
    After ``start_p0``, if no *external* VC join is recorded within this many seconds, call
    ``cancel_p0_session`` (ends VC + cancelled card). **0** = disabled.

    "External" = join event with a **non-empty** ``open_id`` that differs from the session
    ``trigger_open_id``. Events without ``open_id`` do not count (avoids blocking auto-cancel on noise).

    ``P0_VC_AUTO_CANCEL_IF_NO_JOINS_SEC`` (see also ``P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_IDS``
    for an allowlist-only mode).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_VC_AUTO_CANCEL_IF_NO_JOINS_SEC") or "").strip()
    if not raw:
        return 0
    try:
        v = int(raw)
    except ValueError:
        log.warning("Invalid P0_VC_AUTO_CANCEL_IF_NO_JOINS_SEC=%r — using 0 (disabled)", raw)
        return 0
    if v < 0:
        return 0
    if v > _VC_AUTO_CANCEL_NO_JOIN_MAX_SEC:
        log.warning(
            "P0_VC_AUTO_CANCEL_IF_NO_JOINS_SEC=%s clamped to max %s",
            v,
            _VC_AUTO_CANCEL_NO_JOIN_MAX_SEC,
        )
        return _VC_AUTO_CANCEL_NO_JOIN_MAX_SEC
    return v


def _parse_oc_chat_id_list(
    raw_env: str,
    *,
    warn_prefix: str,
) -> FrozenSet[str]:
    if not raw_env.strip():
        return frozenset()
    out: List[str] = []
    for seg in raw_env.split(","):
        x = seg.strip()
        if not x:
            continue
        if x.startswith("oc_"):
            out.append(x)
        else:
            log.warning("%s: skip invalid %r — expected oc_...", warn_prefix, x)
    return frozenset(out)


def get_p0_vc_auto_cancel_if_no_joins_chat_ids() -> FrozenSet[str]:
    """
    Optional **allowlist** of incident ``oc_`` chats that get VC auto-cancel when no external joins.

    If **empty**: ``get_p0_vc_auto_cancel_if_no_joins_sec()`` applies to **all** incident-source sessions
    (same as a global timeout).

    If **non-empty**: only listed chats are scheduled; delay is
    ``get_p0_vc_auto_cancel_scoped_delay_sec()`` (default **1800** = 30 minutes).

    Env (preferred): ``P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_IDS`` (comma-separated ``oc_``).

    Legacy aliases: ``P0_EMERGENCY_TEST_GROUP_CHAT_IDS``, ``P0_EMERGENCY_TEST_GROUP_CHAT_ID``.
    """
    reload_env_runtime()
    raw = (
        os.getenv("P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_IDS")
        or os.getenv("P0_EMERGENCY_TEST_GROUP_CHAT_IDS")
        or os.getenv("P0_EMERGENCY_TEST_GROUP_CHAT_ID")
        or ""
    ).strip()
    return _parse_oc_chat_id_list(raw, warn_prefix="P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_IDS")


def get_p0_vc_auto_cancel_scoped_delay_sec() -> int:
    """
    Auto-cancel delay for chats listed in ``get_p0_vc_auto_cancel_if_no_joins_chat_ids()``.

    Default **1800** (30 minutes) when unset.

    Env (preferred): ``P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_SEC``.

    Legacy alias: ``P0_VC_AUTO_CANCEL_EMERGENCY_TEST_GROUP_SEC``.
    """
    reload_env_runtime()
    raw = (
        os.getenv("P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_SEC")
        or os.getenv("P0_VC_AUTO_CANCEL_EMERGENCY_TEST_GROUP_SEC")
        or ""
    ).strip()
    if not raw:
        v = 1800
    else:
        try:
            v = int(raw)
        except ValueError:
            log.warning(
                "Invalid P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_SEC=%r — using 1800",
                raw,
            )
            v = 1800
    if v < 0:
        return 0
    if v > _VC_AUTO_CANCEL_NO_JOIN_MAX_SEC:
        log.warning(
            "P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_SEC=%s clamped to max %s",
            v,
            _VC_AUTO_CANCEL_NO_JOIN_MAX_SEC,
        )
        return _VC_AUTO_CANCEL_NO_JOIN_MAX_SEC
    return v


def get_p0_vc_auto_cancel_sec_for_source_chat(source_chat_id: str) -> int:
    """
    Auto-cancel delay (seconds) for a VC session whose **source** incident chat is ``source_chat_id``.

    - If ``P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_IDS`` (or legacy emergency-test aliases) is **non-empty**:
      listed chats use ``P0_VC_AUTO_CANCEL_IF_NO_JOINS_CHAT_SEC`` (default **1800**); unlisted chats get **0**.
    - If that allowlist is **empty**: use ``P0_VC_AUTO_CANCEL_IF_NO_JOINS_SEC`` for every chat (**0** = off).
    """
    reload_env_runtime()
    cid = (source_chat_id or "").strip()
    scoped = get_p0_vc_auto_cancel_if_no_joins_chat_ids()
    if scoped:
        if cid and cid in scoped:
            return get_p0_vc_auto_cancel_scoped_delay_sec()
        return 0
    return get_p0_vc_auto_cancel_if_no_joins_sec()


def is_open_id(x: str) -> bool:
    return bool(OPEN_ID_RE.match((x or "").strip()))


def get_host_and_dm_open_id() -> str:
    """
    One ``ou_`` for **both** VC organizer (primary owner) **and** DM instruction recipient.

    Set ``P0_HOST_AND_DM_OPEN_ID=ou_...`` when a single duty user should host the meeting
    and receive the bot DM. Used only as a **fallback** when the more specific vars below
    are unset.
    """
    reload_env_runtime()
    v = (os.getenv("P0_HOST_AND_DM_OPEN_ID") or "").strip()
    return v if v and is_open_id(v) else ""


def get_owner_ids() -> List[str]:
    """
    Lark VC reserve `owner_id` (organizer). Set in .env — required for creating meetings.

    P0_OWNER_OPEN_IDS — comma-separated open_ids (first id is primary owner).
    P0_INVITEE_OPEN_IDS — legacy alias for the same variable.
    If both empty: ``P0_HOST_AND_DM_OPEN_ID`` (single user) is used as the only owner.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_OWNER_OPEN_IDS") or os.getenv("P0_INVITEE_OPEN_IDS") or "").strip()
    if not raw:
        one = get_host_and_dm_open_id()
        return [one] if one else []
    ids = [x.strip() for x in raw.split(",") if x and x.strip()]
    out = [x for x in ids if is_open_id(x)]
    return out


def get_p0_trigger_ignore_open_ids() -> FrozenSet[str]:
    """
    Senders in this set cannot start P0/P1 from the incident group (silent ignore).

    P0_TRIGGER_IGNORE_OPEN_IDS — comma-separated Lark user open_ids (ou_...).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_TRIGGER_IGNORE_OPEN_IDS") or "").strip()
    if not raw:
        return frozenset()
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return frozenset(x for x in ids if is_open_id(x))


def get_p0_thread_confirm_asker_open_ids() -> FrozenSet[str]:
    """
    ``P0_THREAD_CONFIRM_ASKER_OPEN_IDS`` — comma-separated ``ou_...``.

    Only these users may post a question like **\"is this P0?\"** to **arm** a thread
    confirmation (someone else replies **yes** → ``start_p0``). If empty, this flow is off.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_THREAD_CONFIRM_ASKER_OPEN_IDS") or "").strip()
    if not raw:
        return frozenset()
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return frozenset(x for x in ids if is_open_id(x))


def get_p0_thread_confirm_target_open_ids() -> FrozenSet[str]:
    """
    ``P0_THREAD_CONFIRM_TARGET_OPEN_IDS`` — comma-separated ``ou_...`` (optional).

    If **non-empty**, a qualifying **\"is this P0?\"** message also **arms** when **at least one**
    of these users appears in Lark ``mentions`` (someone @'d them to confirm), even if the sender
    is **not** in ``P0_THREAD_CONFIRM_ASKER_OPEN_IDS``.

    Use with ``P0_THREAD_CONFIRM_ASKER_OPEN_IDS`` (OR): duty users can still arm without @'s;
    anyone can arm when @'ing a designated confirmer.

    If both this and ``P0_THREAD_CONFIRM_ASKER_OPEN_IDS`` are empty, thread confirm is off.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_THREAD_CONFIRM_TARGET_OPEN_IDS") or "").strip()
    if not raw:
        return frozenset()
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return frozenset(x for x in ids if is_open_id(x))


def get_p0_thread_confirm_responder_open_ids() -> FrozenSet[str]:
    """
    ``P0_THREAD_CONFIRM_RESPONDER_OPEN_IDS`` — optional comma-separated ``ou_...``.

    If **non-empty**, only these users may reply **yes** to confirm. If **empty**, any user
    except the asker may confirm.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_THREAD_CONFIRM_RESPONDER_OPEN_IDS") or "").strip()
    if not raw:
        return frozenset()
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return frozenset(x for x in ids if is_open_id(x))


def get_p0_thread_confirm_ttl_sec() -> int:
    """``P0_THREAD_CONFIRM_TTL_SEC`` — how long a question stays armed (default 3600)."""
    reload_env_runtime()
    raw = (os.getenv("P0_THREAD_CONFIRM_TTL_SEC") or "3600").strip()
    try:
        n = int(raw)
    except Exception:
        n = 3600
    return max(60, min(n, 86400 * 7))


def get_p0_thread_confirm_allow_toplevel_yes() -> bool:
    """
    ``P0_THREAD_CONFIRM_ALLOW_TOPLEVEL_YES`` — if ``1``, while a question is **armed**,
    a **top-level** message in the same group (no ``parent_id`` / ``root_id``) that starts
    with **yes** can confirm P0, not only a **Reply** to the question.

    Default ``0`` (stricter: must use Reply / thread so Lark ties the message to the question).

    When enabled, see also ``P0_THREAD_CONFIRM_TOPLEVEL_GRACE_SEC`` and @mention of the asker
    (from webhook ``mentions[].id``) to limit false positives.
    """
    reload_env_runtime()
    v = (os.getenv("P0_THREAD_CONFIRM_ALLOW_TOPLEVEL_YES") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_thread_confirm_allow_asker_self_yes() -> bool:
    """
    ``P0_THREAD_CONFIRM_ALLOW_ASKER_SELF_YES`` — if ``1``, the designated asker may reply **yes**
    to their own **\"is this P0?\"** thread to start the meeting (same person asks + confirms).

    Default ``0``: someone *else* must reply **yes** (reduces self-trigger abuse).
    """
    reload_env_runtime()
    v = (os.getenv("P0_THREAD_CONFIRM_ALLOW_ASKER_SELF_YES") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_thread_confirm_use_groq() -> bool:
    """
    ``P0_THREAD_CONFIRM_USE_GROQ`` — if ``1``, when thread-confirm is **armed** and the reply
    does **not** match the regex allowlist, call Groq once to classify whether the reply affirms P0.

    Regex hits still skip Groq (fast path). Default ``0`` (regex only).
    """
    reload_env_runtime()
    v = (os.getenv("P0_THREAD_CONFIRM_USE_GROQ") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_keyword_groq_gate() -> bool:
    """
    ``P0_KEYWORD_GROQ_GATE`` — if ``1``, after regex filters pass, call Groq once to decide whether
    the message is a **new** P0 bridge declaration vs a passing mention (e.g. status inside an
    existing P0 meeting). Default ``0``.

    This is the **semantic** alternative to growing a long list of hard-coded heuristics in code;
    enable it when informal chat phrasing keeps bypassing rule-based filters.
    """
    reload_env_runtime()
    v = (os.getenv("P0_KEYWORD_GROQ_GATE") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_keyword_ai_triage() -> bool:
    """
    ``P0_KEYWORD_AI_TRIAGE`` — when ``1`` (default) and an AI key is set (``ANTHROPIC_API_KEY``,
    ``GEMINI_API_KEY``, or ``GROQ_API_KEY``), one LLM call classifies each ``p0`` / ``p1`` keyword hit.
    """
    reload_env_runtime()
    v = (os.getenv("P0_KEYWORD_AI_TRIAGE") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def _anthropic_api_key() -> str:
    reload_env_runtime()
    return (os.getenv("ANTHROPIC_API_KEY") or "").strip()


def get_anthropic_api_key() -> str:
    """``ANTHROPIC_API_KEY`` for Claude triage. Never commit this value."""
    return _anthropic_api_key()


def anthropic_claude_configured() -> bool:
    """True when Claude is reachable via API key, OAuth file, or ``ANTHROPIC_AUTH_TOKEN``."""
    from . import anthropic_client as _anthropic

    return _anthropic.has_anthropic_auth()


def _gemini_api_key() -> str:
    reload_env_runtime()
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def get_gemini_api_key() -> str:
    """``GEMINI_API_KEY`` or alias ``GOOGLE_API_KEY`` (Google AI Studio). Never commit this value."""
    return _gemini_api_key()


def priority_keyword_ai_provider_chain() -> list:
    """
    Ordered providers for P0/P1 keyword AI triage + failover.

    ``auto`` (default): **claude → gemini → groq** (each step skipped if key missing).
    Force one: ``P0_KEYWORD_AI_PROVIDER=claude|gemini|groq``.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_KEYWORD_AI_PROVIDER") or "auto").strip().lower()
    has_claude = anthropic_claude_configured()
    has_gemini = bool(_gemini_api_key())
    has_groq = bool(GROQ_API_KEY)
    avail = {"claude": has_claude, "gemini": has_gemini, "groq": has_groq}
    if raw in avail:
        return [raw] if avail[raw] else []
    chain: list = []
    for name in ("claude", "gemini", "groq"):
        if avail[name]:
            chain.append(name)
    return chain


def resolve_priority_keyword_ai_provider() -> str:
    """First provider in ``priority_keyword_ai_provider_chain()`` (for startup / availability checks)."""
    chain = priority_keyword_ai_provider_chain()
    return chain[0] if chain else ""


def get_p0_keyword_use_builtin_context_filters() -> bool:
    """
    ``P0_KEYWORD_USE_BUILTIN_CONTEXT_FILTERS`` — if ``1`` (default), apply built-in heuristics
    (issue prose / in-meeting / informational or past-date phrasing) before opening a VC.

    Set ``0`` to **disable** those code-based checks and rely on ``P0_KEYWORD_GROQ_GATE``
    and/or ``P0_KEYWORD_SUPPLEMENTAL_SKIP_REGEX`` instead (recommended: enable Groq if you turn this off).
    """
    reload_env_runtime()
    v = (os.getenv("P0_KEYWORD_USE_BUILTIN_CONTEXT_FILTERS") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def get_p0_keyword_supplemental_skip_regex() -> Optional[re.Pattern[str]]:
    """
    ``P0_KEYWORD_SUPPLEMENTAL_SKIP_REGEX`` — optional Python regex (``re.IGNORECASE | re.DOTALL``).
    If it matches the message **anywhere** (after ``p0`` is already present), the keyword VC trigger
    is skipped. Lets you tune false positives via **config**, without a code deploy.

    Invalid patterns log a warning and are ignored.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_KEYWORD_SUPPLEMENTAL_SKIP_REGEX") or "").strip()
    if not raw:
        return None
    try:
        return re.compile(raw, re.IGNORECASE | re.DOTALL)
    except re.error as e:
        log.warning("P0_KEYWORD_SUPPLEMENTAL_SKIP_REGEX invalid (ignored): %s", e)
        return None


def get_p0_thread_confirm_toplevel_grace_sec() -> float:
    """
    ``P0_THREAD_CONFIRM_TOPLEVEL_GRACE_SEC`` — after the duty user arms **\"is this P0?\"**,
    for this many seconds a **plain** top-level **yes** (no ``@`` to the asker) still counts
    as in-conversation confirmation. After the grace window, a top-level yes must **@mention**
    the asker's ``ou_...`` (as sent in the webhook ``mentions`` list) or the confirmer must
    use **Reply** to the question message.

    Default ``180``. Set ``0`` to require @mention (or thread reply) for **every** top-level yes.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_THREAD_CONFIRM_TOPLEVEL_GRACE_SEC") or "180").strip()
    try:
        n = float(raw)
    except Exception:
        n = 180.0
    return max(0.0, min(n, float(86400 * 7)))


def get_incident_group_command_open_ids() -> FrozenSet[str]:
    """
    Parsed from ``P0_INCIDENT_GROUP_COMMAND_OPEN_IDS`` (comma-separated ``ou_...``).
    **No longer used for gating** — incident-group controls are available to all chat members who can message the bot.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_INCIDENT_GROUP_COMMAND_OPEN_IDS") or "").strip()
    if not raw:
        return frozenset()
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return frozenset(x for x in ids if is_open_id(x))


def can_use_incident_group_commands(user_open_id: str) -> bool:
    """Incident-group control commands are available to all members who can message the bot."""
    return True


def get_dm_instruction_open_ids() -> List[str]:
    """
    If non-empty, P0/P1 DM instruction cards are sent to these users instead of whoever typed p0/p1.

    P0_DM_INSTRUCTION_OPEN_IDS — comma-separated open_ids (ou_...), multiple recipients.
    P0_DM_INSTRUCTION_OPEN_ID — single open_id (legacy; use if OPEN_IDS is unset).
    If those are unset: ``P0_HOST_AND_DM_OPEN_ID`` (same as single host + DM user).
    """
    reload_env_runtime()
    raw_multi = (os.getenv("P0_DM_INSTRUCTION_OPEN_IDS") or "").strip()
    if raw_multi:
        parts = [x.strip() for x in raw_multi.split(",") if x.strip()]
        return [x for x in parts if is_open_id(x)]
    raw_single = (os.getenv("P0_DM_INSTRUCTION_OPEN_ID") or "").strip()
    if raw_single and is_open_id(raw_single):
        return [raw_single]
    one = get_host_and_dm_open_id()
    return [one] if one else []


def get_dm_instruction_open_id() -> str:
    """First DM instruction recipient, or empty (for simple callers)."""
    ids = get_dm_instruction_open_ids()
    return ids[0] if ids else ""


def get_dm_repost_instruction_after_reset() -> bool:
    """
    If True, after **Clear draft** (button or ``CLEAR_RE`` text) the bot reposts the DM
    instruction card. A **draft-cleared** text prompt (paste screenshots/text again) is
    **always** sent regardless of this flag. **Cancel preview** always recalls the
    preview message and posts a fresh instruction card (not gated by this flag). Default
    **False** for instruction-card repost on clear-draft; set
    ``P0_DM_REPOST_INSTRUCTION_AFTER_RESET=1`` to repost the card too. Older instruction
    cards in the thread usually still accept button clicks.
    """
    reload_env_runtime()
    v = (os.getenv("P0_DM_REPOST_INSTRUCTION_AFTER_RESET") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _parse_incident_keyed_url_map(raw: str) -> Dict[str, str]:
    """
    Comma-separated ``oc_...=value`` (value may contain ``=`` in URL — split on first ``=`` only).
    """
    out: Dict[str, str] = {}
    if not (raw or "").strip():
        return out
    for segment in raw.split(","):
        segment = segment.strip()
        if "=" not in segment:
            continue
        key, _, val = segment.partition("=")
        key, val = key.strip(), val.strip()
        if key.startswith("oc_") and val:
            out[key] = val
    return out


def p0_graph_screenshot_enabled() -> bool:
    """
    When True, after a **P0** session starts the bot captures a Playwright screenshot of
    ``P0_GRAPH_SCREENSHOT_URL`` and posts it to ``P0_GRAPH_SCREENSHOT_TARGET_CHAT_ID``.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_url() -> str:
    reload_env_runtime()
    return (os.getenv("P0_GRAPH_SCREENSHOT_URL") or "").strip()


_GRAFANA_RANGE_FROM_QUERY = {
    "30m": "now-30m",
    "1h": "now-1h",
    "2h": "now-2h",
    "3h": "now-3h",
    "6h": "now-6h",
}


def get_p0_graph_screenshot_range_display(range_key: str) -> str:
    """Human label for captions (``6h`` → ``6 hours``)."""
    rk = (range_key or "").strip().lower()
    return {
        "30m": "30 minutes",
        "1h": "1 hour",
        "2h": "2 hours",
        "3h": "3 hours",
        "6h": "6 hours",
    }.get(rk, rk or "dashboard")


def build_p0_graph_screenshot_url_for_range(range_key: str) -> str:
    """
    Build Grafana URL for a time window by setting ``from=`` on ``P0_GRAPH_SCREENSHOT_URL``.
    Keys: ``30m``, ``1h``, ``2h``, ``3h``, ``6h``.
    """
    reload_env_runtime()
    base = get_p0_graph_screenshot_url()
    if not base:
        return ""
    rk = (range_key or "").strip().lower()
    from_val = _GRAFANA_RANGE_FROM_QUERY.get(rk)
    if not from_val:
        return base
    parsed = urlparse(base)
    q = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)]
    replaced = False
    out_q: List[Tuple[str, str]] = []
    for k, v in q:
        if k.lower() == "from":
            out_q.append((k, from_val))
            replaced = True
        else:
            out_q.append((k, v))
    if not replaced:
        out_q.append(("from", from_val))
    if not any(k.lower() == "to" for k, _ in out_q):
        out_q.append(("to", "now"))
    return urlunparse(parsed._replace(query=urlencode(out_q)))


def get_p0_graph_screenshot_auto_range_keys() -> List[str]:
    """
    Ranges posted automatically on P0 start and on each interval repeat.
    Default ``6h`` only. ``3h``, ``1h``, and ``30m`` are on-demand only unless added here.

    Env: ``P0_GRAPH_SCREENSHOT_AUTO_RANGES=6h``
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_AUTO_RANGES") or "6h").strip()
    out: List[str] = []
    for seg in raw.split(","):
        rk = seg.strip().lower()
        if rk in _GRAFANA_RANGE_FROM_QUERY and rk not in out:
            out.append(rk)
    return out or ["6h"]


def p0_graph_screenshot_on_demand_enabled() -> bool:
    """Typed screenshot requests in allowed chats (includes ``30m``). Default on when screenshots enabled."""
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_ON_DEMAND") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_ai_enabled() -> bool:
    """
    ``P0_GRAPH_SCREENSHOT_AI`` — when ``1`` (default) and an AI key is set (``ANTHROPIC_API_KEY`` or
    ``GROQ_API_KEY``), classifies natural-language requests like ``please give 30 mins`` (OTE-AI style).
    Provider: ``P0_GRAPH_SCREENSHOT_AI_PROVIDER`` (``auto`` prefers Claude).
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_AI") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_ai_provider() -> str:
    """``claude`` | ``groq`` | ``auto`` — see ``graph_screenshot_ai.resolve_graph_screenshot_ai_provider``."""
    from features.screenshot.graph_screenshot_ai import resolve_graph_screenshot_ai_provider

    return resolve_graph_screenshot_ai_provider()


def get_p0_graph_screenshot_bot_mention_hints() -> Tuple[str, ...]:
    """
    Substrings matched against Lark @mention display names to detect a direct bot ping.
    Env: ``P0_GRAPH_SCREENSHOT_BOT_MENTION_HINTS`` (comma-separated).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_BOT_MENTION_HINTS") or "").strip()
    if raw:
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return ("automation-bot", "p0-automation", "p1/p0", "lark-ops", "p0 bot", "p1 bot")


def get_p0_graph_screenshot_on_demand_chat_ids() -> FrozenSet[str]:
    """
    Extra chats allowed for on-demand Grafana requests (comma ``oc_``).

    When empty, only ``P0_GRAPH_SCREENSHOT_TARGET_CHAT_ID`` (screenshot hub) is allowed —
    not all ``INCIDENT_GROUP_IDS``.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_ON_DEMAND_CHAT_IDS") or "").strip()
    return _parse_oc_chat_id_list(raw, warn_prefix="P0_GRAPH_SCREENSHOT_ON_DEMAND_CHAT_IDS")


def get_p0_graph_screenshot_append_kiosk() -> bool:
    """
    When True (default), append Grafana ``kiosk`` / ``kiosk=tv`` to the dashboard URL if not already
    present — hides the left nav and yields a cleaner capture (ops-style “panels only”).
    Set ``P0_GRAPH_SCREENSHOT_KIOSK=0`` to disable.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_KIOSK") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def get_p0_graph_screenshot_include_time_bar() -> bool:
    """
    When True, pic 1 scrolls to the dashboard top so **Last 6 hours** / refresh controls appear,
    and does not append ``_dash.hideTimePicker=true`` to the capture URL.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_INCLUDE_TIME_BAR") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_zoom_percent() -> int:
    """
    **Browser zoom** for capture (like Chromium Ctrl +/-): ``30`` = 30% page zoom at 1920×1080 viewport.
    Uses CDP ``Emulation.setPageScaleFactor`` + CSS ``zoom`` fallback. Clamped 5–100; default **50**.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_ZOOM_PERCENT") or "50").strip()
    try:
        n = int(raw)
    except ValueError:
        log.warning("P0_GRAPH_SCREENSHOT_ZOOM_PERCENT=%r invalid — using 50", raw)
        return 50
    clamped = max(5, min(n, 100))
    if clamped != n:
        log.warning(
            "P0_GRAPH_SCREENSHOT_ZOOM_PERCENT=%s clamped to %s (allowed 5–100)",
            raw,
            clamped,
        )
    return clamped


def get_p0_graph_screenshot_dashboard_clip() -> bool:
    """
    ``P0_GRAPH_SCREENSHOT_DASHBOARD_CLIP`` — when ``1``, crop to the dashboard body (fallback if nav
    still visible). Default **off** — full viewport after CSS hide gives correct sizing in Lark.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_DASHBOARD_CLIP") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_clip_selectors() -> List[str]:
    """
    Comma-separated CSS selectors (first **substantial** visible match) for the **dashboard body**.
    Narrow ``.scrollbar-view`` hits (side gutter ~400px wide) are skipped so screenshots are not blank gray.
    Playwright uses the element’s box + ``scrollHeight`` as the ``full_page`` clip — excluding
    most browser chrome; pair with ``P0_GRAPH_SCREENSHOT_KIOSK=1`` (default).

    Override with ``P0_GRAPH_SCREENSHOT_CLIP_SELECTOR=main`` (or several, comma-separated).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_CLIP_SELECTOR") or "").strip()
    if raw.lower() in ("-", "none", "off", "0"):
        return []
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [
        '[data-testid="canvas-body"] .scrollbar-view',
        "main .scrollbar-view",
        "main",
        ".react-grid-layout",
    ]


def get_p0_graph_screenshot_target_chat_id() -> str:
    """Lark group ``oc_...`` to receive the screenshot (can differ from incident group)."""
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_TARGET_CHAT_ID") or "").strip()
    return v if v.startswith("oc_") and len(v) > 12 else ""


def get_p0_graph_screenshot_viewport_width() -> int:
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_VIEWPORT_WIDTH") or "1920").strip()
    try:
        n = int(raw)
    except Exception:
        n = 1280
    return max(320, min(n, 3840))


def get_p0_graph_screenshot_viewport_height() -> int:
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_VIEWPORT_HEIGHT") or "1080").strip()
    try:
        n = int(raw)
    except Exception:
        n = 720
    return max(240, min(n, 2160))


def get_p0_graph_screenshot_device_scale_factor() -> float:
    """
    Playwright ``device_scale_factor`` (CSS pixel ratio for screenshots). ``2`` renders **2×** device
    pixels per CSS pixel — much sharper when Lark downscales wide dashboards; default ``1``.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_DEVICE_SCALE_FACTOR") or "1").strip()
    try:
        x = float(raw)
    except Exception:
        x = 1.0
    if x < 1.0:
        return 1.0
    if x > 3.0:
        return 3.0
    return x


def get_p0_graph_screenshot_wait_ms() -> int:
    """Extra wait after ``goto`` (and ``wait_until``) before ``screenshot`` — lets Grafana panels query/render."""
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_WAIT_MS") or "4000").strip()
    try:
        n = int(raw)
    except Exception:
        n = 4000
    # Ceiling lowered 120s→30s: this is a fixed post-nav sleep; larger values only add
    # dead wait per capture and were a runaway-latency source.
    return max(0, min(n, 30_000))


def get_p0_graph_screenshot_panel_ready_timeout_ms() -> int:
    """
    After navigation, wait up to this many ms for Grafana dashboard panel DOM (e.g. ``.react-grid-item``)
    before the fixed ``P0_GRAPH_SCREENSHOT_WAIT_MS`` sleep. Reduces **blank black** screenshots when
    ``load`` fires before React panels mount. Set **0** to skip (default). For heavy dashboards try
    **20000–35000**.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_PANEL_READY_TIMEOUT_MS") or "0").strip()
    try:
        n = int(raw)
    except Exception:
        n = 0
    # Ceiling lowered 120s→60s: recommended range is 20–35s; 60s is ample headroom.
    return max(0, min(n, 60_000))


def get_p0_graph_screenshot_panel_content_ready_timeout_ms() -> int:
    """
    After panel scaffolding exists (``.react-grid-item`` / ``[data-panel-id]``), Grafana still needs
    time to run queries and paint **canvas / SVG** charts. This timeout drives ``page.wait_for_function``
    until graphs look present; set **0** to skip.

    If ``P0_GRAPH_SCREENSHOT_PANEL_CONTENT_READY_TIMEOUT_MS`` is **unset** but
    ``P0_GRAPH_SCREENSHOT_PANEL_READY_TIMEOUT_MS`` **> 0**, defaults to
    ``max(45000, panel_ready + 25000)`` (capped at **180000**) so enabling grid wait also waits for
    chart paint unless you explicitly set ``PANEL_CONTENT_READY_TIMEOUT_MS=0``.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_PANEL_CONTENT_READY_TIMEOUT_MS") or "").strip()
    if raw:
        try:
            return max(0, min(int(raw), 180_000))
        except ValueError:
            log.warning("P0_GRAPH_SCREENSHOT_PANEL_CONTENT_READY_TIMEOUT_MS=%r invalid — using cascade", raw)
    pr_raw = (os.getenv("P0_GRAPH_SCREENSHOT_PANEL_READY_TIMEOUT_MS") or "0").strip()
    try:
        pr = int(pr_raw)
    except ValueError:
        pr = 0
    if pr <= 0:
        return 0
    inferred = max(45_000, pr + 25_000)
    return min(inferred, 180_000)


def get_p0_graph_screenshot_nav_timeout_ms() -> int:
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_NAV_TIMEOUT_MS") or "60000").strip()
    try:
        n = int(raw)
    except Exception:
        n = 60000
    return max(5000, min(n, 300_000))


def get_p0_graph_screenshot_full_page() -> bool:
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_FULL_PAGE") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_full_document() -> bool:
    """
    When ``1``: ``page.screenshot(full_page=True)`` over the **entire document** (no CSS clip).

    That can produce a **very tall** PNG (whole scroll), often with large empty bands if panels lazy-load —
    bad for Lark. For **two images that each show half of what you see in the browser window** (fixed
    viewport, like 1920×1080), use ``P0_GRAPH_SCREENSHOT_VIEWPORT_ONLY=1`` + ``SPLIT_VERTICAL_HALVES=1``
    and keep this off.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_FULL_DOCUMENT") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_split_vertical_halves() -> bool:
    """
    When True: post **multiple** PNGs along the dashboard scroll (see
    ``P0_GRAPH_SCREENSHOT_VIEWPORT_SCROLL_COUNT``) or legacy Pillow / clip splits.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_SPLIT_VERTICAL_HALVES") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_viewport_scroll_count() -> int:
    """
    With ``P0_GRAPH_SCREENSHOT_VIEWPORT_ONLY=1``: number of **full-viewport** screenshots taken in a row,
    scrolling the main dashboard down ~one screen between each (less clutter per image for long boards).

    If ``P0_GRAPH_SCREENSHOT_VIEWPORT_SCROLL_COUNT`` is **unset**: ``2`` when split-halves is on, else ``1``.
    Clamped to **1–8**.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_VIEWPORT_SCROLL_COUNT") or "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 8))
        except ValueError:
            log.warning(
                "P0_GRAPH_SCREENSHOT_VIEWPORT_SCROLL_COUNT=%r invalid — using split default",
                raw,
            )
    v = (os.getenv("P0_GRAPH_SCREENSHOT_SPLIT_VERTICAL_HALVES") or "0").strip().lower()
    split = v in ("1", "true", "yes", "on")
    return 2 if split else 1


def get_p0_graph_screenshot_top_and_bottom() -> bool:
    """
    ``P0_GRAPH_SCREENSHOT_TOP_AND_BOTTOM`` — when ``1`` (default) with ``VIEWPORT_ONLY=1``:
    capture cropped dashboard bands matching manual refs (top KPI/FPMS, then CPMS/IGO/Pulsar).
    Set ``0`` to use incremental viewport scroll (``VIEWPORT_SCROLL_COUNT`` steps).
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_TOP_AND_BOTTOM") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_include_login_panel() -> bool:
    """
    ``P0_GRAPH_SCREENSHOT_INCLUDE_LOGIN_PANEL`` — ``0`` (recommended): **two** PNGs like manual VNC —
    pic1 = KPI/FPMS block; pic2 = **CPMS/IGO/Pulsar** (starts at CPMS header, includes Pulsar row).
    ``1``: **three** PNGs with Login as its own 3rd image.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_INCLUDE_LOGIN_PANEL") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_username() -> str:
    """Grafana login user for Playwright auto-login when the session is logged out."""
    reload_env_runtime()
    return (os.getenv("P0_GRAPH_SCREENSHOT_USERNAME") or os.getenv("GRAFANA_USERNAME") or "").strip()


def get_p0_graph_screenshot_password() -> str:
    """Grafana login password for Playwright auto-login (never log this value)."""
    reload_env_runtime()
    return (os.getenv("P0_GRAPH_SCREENSHOT_PASSWORD") or os.getenv("GRAFANA_PASSWORD") or "").strip()


def get_p0_graph_screenshot_goto_wait_until() -> str:
    """
    Playwright ``page.goto(..., wait_until=...)``.

    - ``load`` (default): wait for load event — good when you want charts to start rendering; pair with a
      higher ``P0_GRAPH_SCREENSHOT_WAIT_MS`` for dense Grafana dashboards.
    - ``domcontentloaded``: earlier — page shell before many panel queries finish (lighter / “before graphs”).
    - ``networkidle``: NOT allowed — Grafana's live/streaming dashboards keep WebSockets/long-poll
      open, so networkidle never settles and every ``goto`` blocks to the nav timeout. Coerced to ``load``.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_GOTO_WAIT_UNTIL") or "load").strip().lower()
    if raw == "networkidle":
        log.warning(
            "P0_GRAPH_SCREENSHOT_GOTO_WAIT_UNTIL=networkidle is unsafe on Grafana "
            "(never settles → goto hangs to the nav timeout) — using load instead"
        )
        return "load"
    allowed = frozenset(("load", "domcontentloaded", "commit"))
    if raw in allowed:
        return raw
    log.warning("P0_GRAPH_SCREENSHOT_GOTO_WAIT_UNTIL=%r invalid — using load", raw)
    return "load"


def get_p0_graph_screenshot_caption() -> str:
    """
    Text posted before the image. Empty env uses code default ``As of: {captured_at} · Last {range}``.

    Placeholders: ``{captured_at}``, ``{range}`` (e.g. ``1 hour``), ``{label}``.
    """
    reload_env_runtime()
    return (os.getenv("P0_GRAPH_SCREENSHOT_CAPTION") or "").strip()


def get_p0_graph_screenshot_timezone_name() -> str:
    """
    IANA zone for ``{captured_at}`` timestamps. Default **Malaysia Time** (``Asia/Kuala_Lumpur``, MYT).

    Set to ``UTC``, ``Asia/Singapore``, etc. if you need a different zone.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_TIMEZONE") or "").strip()
    return raw if raw else "Asia/Kuala_Lumpur"


def get_p0_graph_screenshot_chromium_args() -> List[str]:
    """
    Comma-separated extra Chromium flags for Playwright on Linux/Docker, e.g.
    ``--no-sandbox,--disable-dev-shm-usage``. Empty = default ``--disable-dev-shm-usage`` only.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_CHROMIUM_ARGS") or "").strip()
    if not raw:
        return ["--disable-dev-shm-usage"]
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_p0_graph_screenshot_playwright_user_data_dir() -> str:
    """
    If set to an **existing** directory, Playwright uses ``launch_persistent_context`` so Chromium
    keeps cookies/local storage (e.g. after you log in to **Grafana** once in a headed browser using
    this same profile path). Same idea as Slack ``SESSION_DIR`` — without this, each run is a fresh
    session and Grafana will usually show the login page unless the dashboard is anonymous/public.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_PLAYWRIGHT_USER_DATA_DIR") or "").strip()
    if not raw:
        return ""
    p = Path(raw).expanduser().resolve()
    return str(p) if p.is_dir() else ""


def get_p0_graph_screenshot_playwright_headless() -> bool:
    """
    Default **headless** Chromium. Set ``P0_GRAPH_SCREENSHOT_HEADED=1`` for a visible window
    (e.g. manual debugging on VNC — same capture path as the bot).
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_HEADED") or "0").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return False
    return True


def get_p0_graph_screenshot_swiftshader() -> bool:
    """
    Append ANGLE + **SwiftShader** Chromium flags (software GL). Fixes **solid black** screenshots
    on many Linux/VPS headless setups where the GPU stack does not composite Web/canvas.

    * Explicit ``P0_GRAPH_SCREENSHOT_SWIFTSHADER=0`` → off.
    * Explicit ``1`` → on.
    * **Unset** → **on** when ``sys.platform`` is Linux (ose-bot style servers); off on macOS/Windows.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_SWIFTSHADER") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return sys.platform.startswith("linux")


def get_p0_graph_screenshot_viewport_only() -> bool:
    """If ``1``, skip CSS clip and capture **viewport** only (``full_page=False``). Debugging / GPU issues."""
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_VIEWPORT_ONLY") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_band_warm_scroll() -> bool:
    """
    ``P0_GRAPH_SCREENSHOT_BAND_WARM_SCROLL`` — scroll through band 2 before capture so lazy
    panels (Pulsar row) load. Set ``0`` if runs feel stuck (skips warm scroll).
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_BAND_WARM_SCROLL") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_band_max_wait_ms() -> int:
    """
    Max ms to wait for panels per screenshot band (avoids feeling stuck on band 2).
    ``P0_GRAPH_SCREENSHOT_BAND_MAX_WAIT_MS`` — default **30000**; capped at 120000.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_BAND_MAX_WAIT_MS") or "30000").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 30_000
    return max(5000, min(n, 120_000))


def get_p0_graph_screenshot_fast_capture() -> bool:
    """
    ``P0_GRAPH_SCREENSHOT_FAST_CAPTURE=1`` (default) — shorter scroll settles and post-band sleeps;
    use with ``TOP_AND_BOTTOM=1``. Set ``0`` for slower, stricter panel waits.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_FAST_CAPTURE") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_on_demand_fast() -> bool:
    """
    On-demand chat requests (``please give 30 mins``) use shorter waits. Default **on**.
    ``P0_GRAPH_SCREENSHOT_ON_DEMAND_FAST=0`` to use full P0-quality timing.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_ON_DEMAND_FAST") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_browser_pool_enabled() -> bool:
    """
    Keep Chromium open between captures (~30–60s faster on P0 declare and on-demand).
    Default **on** when ``P0_GRAPH_SCREENSHOT_PLAYWRIGHT_USER_DATA_DIR`` is set; else off.
    Force off: ``P0_GRAPH_SCREENSHOT_BROWSER_POOL=0``.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_BROWSER_POOL") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(get_p0_graph_screenshot_playwright_user_data_dir())


def get_p0_graph_screenshot_on_demand_band_max_wait_ms() -> int:
    """Per-band panel wait cap for on-demand chat captures. Default **28000** (28s)."""
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_ON_DEMAND_BAND_MAX_WAIT_MS") or "28000").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 28_000
    return max(3000, min(n, 120_000))


def get_p0_graph_screenshot_on_demand_band_stable_polls() -> int:
    """Stability polls for on-demand fast captures. Default **2** (P0 auto still uses ``BAND_STABLE_POLLS``)."""
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_ON_DEMAND_BAND_STABLE_POLLS") or "2").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, min(n, 6))


def get_p0_graph_screenshot_on_demand_max_sec() -> int:
    """Enforced hard deadline for one on-demand capture; the capture is abandoned + failure posted if exceeded. Default **240** (4 min)."""
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_ON_DEMAND_MAX_SEC") or "240").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 240
    # Ceiling lowered 900→480: past ~4 min a capture is almost certainly wedged, not slow.
    return max(60, min(n, 480))


def get_p0_graph_screenshot_auto_max_sec() -> int:
    """Enforced hard deadline for auto P0-start / interval capture; the capture is abandoned + failure posted if exceeded. Default **360** (6 min)."""
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_AUTO_MAX_SEC") or "360").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 360
    # Ceiling lowered 1200→600: bounds worst-case single capture to 10 min even if misconfigured.
    return max(120, min(n, 600))


def get_p0_graph_screenshot_band_panel_ready_ratio() -> float:
    """
    Fraction of viewport panels that must show chart / table / stable ``No data`` before capture.
    Default **0.88** (``0.55`` when ``FAST_CAPTURE`` or on-demand fast — see graph_screenshot).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_BAND_PANEL_READY_RATIO") or "0.88").strip()
    try:
        r = float(raw)
    except ValueError:
        r = 0.88
    if r > 1.0:
        r = r / 100.0
    return max(0.5, min(r, 1.0))


def get_p0_graph_screenshot_band_max_blank_panels() -> int:
    """Max unloaded (black) panels allowed in the viewport before capture. Default **0**."""
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_BAND_MAX_BLANK_PANELS") or "0").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 0
    return max(0, min(n, 5))


def get_p0_graph_screenshot_band_stable_polls() -> int:
    """Consecutive ready checks required before screenshot. Default **3**."""
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_BAND_STABLE_POLLS") or "3").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 3
    return max(1, min(n, 8))


def get_p0_graph_screenshot_band_stable_poll_ms() -> int:
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_BAND_STABLE_POLL_MS") or "900").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 900
    return max(300, min(n, 3000))


def get_p0_graph_screenshot_react_enabled() -> bool:
    """
    Emoji reactions on the user's on-demand request message (``OnIt`` → ``DONE``).
    Default **on**. Requires Lark scope ``im:message.reactions:write_only``.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_REACT") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_graph_screenshot_react_queued_emoji() -> str:
    """Reaction when capture starts. Default ``OnIt`` (Lark emoji_type)."""
    reload_env_runtime()
    return (os.getenv("P0_GRAPH_SCREENSHOT_REACT_QUEUED") or "OnIt").strip() or "OnIt"


def get_p0_graph_screenshot_react_done_emoji() -> str:
    """Reaction when screenshots post successfully. Default ``DONE``."""
    reload_env_runtime()
    return (os.getenv("P0_GRAPH_SCREENSHOT_REACT_DONE") or "DONE").strip() or "DONE"


def get_p0_graph_screenshot_react_failed_emoji() -> str:
    """Reaction when capture fails. Default ``ERROR``. Empty = skip."""
    reload_env_runtime()
    return (os.getenv("P0_GRAPH_SCREENSHOT_REACT_FAILED") or "ERROR").strip()


def get_p0_group_overview_edit_enabled() -> bool:
    """
    When True, group overview cards get an **Edit overview** button; Save in DM PATCHes that group message.
    ``P0_GROUP_OVERVIEW_EDIT_ENABLED=0`` to disable.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GROUP_OVERVIEW_EDIT_ENABLED") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def get_p0_graph_screenshot_interval_min() -> int:
    """
    While any **P0** session is active, repeat screenshot posts every N minutes.
    ``P0_GRAPH_SCREENSHOT_INTERVAL_MIN`` — **0** = only on P0 start (default). **20** = every 20 min until P0 ends.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_GRAPH_SCREENSHOT_INTERVAL_MIN") or "0").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 0
    return max(0, min(n, 24 * 60))


def get_p0_graph_screenshot_blank_fallback_viewport() -> bool:
    """
    If the PNGs look **uniformly blank** (near-black, no contrast), retry once with viewport-only
    capture. On by default; set ``P0_GRAPH_SCREENSHOT_BLANK_FALLBACK_VIEWPORT=0`` to disable.
    """
    reload_env_runtime()
    v = (os.getenv("P0_GRAPH_SCREENSHOT_BLANK_FALLBACK_VIEWPORT") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def get_lark_primary_app_credentials() -> Tuple[str, str]:
    """Main bot: P0 meeting, green overview DM, most IM (``LARK_APP_ID`` / ``LARK_APP_SECRET``)."""
    reload_env_runtime()
    return ((os.getenv("LARK_APP_ID") or "").strip(), (os.getenv("LARK_APP_SECRET") or "").strip())


def _strip_lark_env_val(raw: str) -> str:
    """Strip whitespace and UTF-8 BOM (bad editor / copy-paste) from .env values."""
    s = (raw or "").strip().strip("\ufeff")
    return s.strip()


def get_lark_severity_app_credentials() -> Tuple[str, str]:
    """
    Optional second bot: severity Major/Minor + minor follow-up cards only.

    Set ``LARK_SEVERITY_APP_ID`` and ``LARK_SEVERITY_APP_SECRET``. Aliases:

    - ``LARK_APP_ID_SEVERITY`` / ``LARK_APP_SECRET_SEVERITY``
    - ``LARK_APP_ID_2`` / ``LARK_APP_SECRET_2`` (common when you name the second app this way)

    If either id or secret is empty, severity DMs use the **primary** app (automation bot).
    """
    reload_env_runtime()
    sid = _strip_lark_env_val(
        os.getenv("LARK_SEVERITY_APP_ID")
        or os.getenv("LARK_APP_ID_SEVERITY")
        or os.getenv("LARK_APP_ID_2")
        or ""
    )
    sec = _strip_lark_env_val(
        os.getenv("LARK_SEVERITY_APP_SECRET")
        or os.getenv("LARK_APP_SECRET_SEVERITY")
        or os.getenv("LARK_APP_SECRET_2")
        or ""
    )
    return sid, sec


def get_p0_issue_watch_enabled() -> bool:
    """``P0_ISSUE_WATCH_ENABLED=1`` — Claude watches detection groups and DMs duty on player issues."""
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_issue_watch_min_confidence() -> float:
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_MIN_CONFIDENCE") or "0.88").strip()
    try:
        c = float(raw)
    except ValueError:
        c = 0.88
    if c > 1.0:
        c = c / 100.0
    return max(0.5, min(c, 0.99))


def get_p0_issue_watch_window_min() -> int:
    """Sliding window for widespread (#8) reporter counting. Default **60** minutes."""
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_WINDOW_MIN") or "60").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 60
    return max(15, min(n, 240))


def get_p0_issue_watch_min_reports() -> int:
    """Unique reporters for same ``issue_fingerprint`` to trigger widespread alert. Default **4**."""
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_MIN_REPORTS") or "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    return max(2, min(n, 20))


def get_p0_issue_watch_min_affected_players() -> int:
    """
    Minimum **affected player count** (from prose or Account IDs) before Issue Watch alerts
    on player impact alone. Default **3** — 1–2 affected players do not trigger via count
    (high-confidence solo path still applies when no player count is mentioned).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_MIN_AFFECTED_PLAYERS") or "3").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 3
    return max(1, min(n, 50))


def get_p0_issue_watch_min_solo_reporters() -> int:
    """
    Without widespread (``MIN_REPORTS``) impact, require this many unique reporters
    on the same ``issue_fingerprint`` before a high-confidence solo message triggers
    a Major alert. Default **2** (single OM/player report never pages alone).
    Set **1** to restore legacy behaviour (one high-confidence message can alert).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_MIN_SOLO_REPORTERS") or "2").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, min(n, 10))


def get_p0_issue_watch_cooldown_min() -> int:
    """Per chat + category/fingerprint DM cooldown. Default **20** minutes."""
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_COOLDOWN_MIN") or "20").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 20
    return max(1, min(n, 180))


def get_p0_issue_watch_id_wait_sec() -> int:
    """When a report mentions players but has no IDs yet, wait this long for an Account/ID follow-up. Default **40** sec."""
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_ID_WAIT_SEC") or "40").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 40
    return max(0, min(n, 120))


def get_p0_issue_watch_min_text_len() -> int:
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_MIN_TEXT_LEN") or "15").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 15
    return max(8, min(n, 200))


def get_p0_issue_watch_auto_overview_enabled() -> bool:
    """
    When true (default), Major detection alert DMs include **Use suggested overview** /
    **Build overview manually** buttons.
    """
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_AUTO_OVERVIEW") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_issue_watch_buzz_enabled() -> bool:
    """``P0_ISSUE_WATCH_BUZZ_ENABLED`` — Lark 加急 on Major detection alert DM (default on)."""
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_BUZZ_ENABLED") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_issue_watch_lark_urgent_mode() -> str:
    """
    Lark buzz mode for Issue Watch alert DMs: ``app`` | ``phone`` | ``sms`` | ``off``.
    Falls back to ``P0_ONGOING_LARK_URGENT_MODE`` when unset.
    """
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_LARK_URGENT_MODE") or "").strip().lower()
    if v:
        return v if v in ("app", "phone", "sms", "off") else "app"
    return get_p0_ongoing_lark_urgent_mode()


def get_p0_issue_watch_declare_p0_enabled() -> bool:
    """``P0_ISSUE_WATCH_DECLARE_P0_ENABLED`` — DM alert buttons to declare P0 from duty (default on)."""
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_DECLARE_P0_ENABLED") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_issue_watch_declare_reply_ai_enabled() -> bool:
    """``P0_ISSUE_WATCH_DECLARE_REPLY_AI`` — Groq contextual declare reply (default on)."""
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_DECLARE_REPLY_AI") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_issue_watch_declare_reply_text() -> str:
    """Fallback plain-text reply when Groq declare-reply is off or fails."""
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_DECLARE_REPLY_TEXT") or "").strip()
    return raw or "We will declare this issue as P0."


def get_p0_issue_watch_declare_reaction() -> str:
    """Lark emoji_type reaction on the source concern message (empty = skip). Default ``OnIt``."""
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_DECLARE_REACTION") or "OnIt").strip()
    return raw


def get_p0_issue_watch_declare_reply_in_thread() -> bool:
    """
    ``P0_ISSUE_WATCH_DECLARE_REPLY_IN_THREAD`` — Lark ``reply_in_thread`` on the concern message.
    Default ``1`` (topic thread on that exact message). Set ``0`` for flat in-feed reply.
    """
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_DECLARE_REPLY_IN_THREAD") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_issue_watch_declare_also_send_to_group() -> bool:
    """
    ``P0_ISSUE_WATCH_DECLARE_ALSO_SEND_TO_GROUP`` — after thread reply, also post the same text
    to the main group feed (Lark UI: "Also send to the group"). API has no single flag for this.
    Default ``1``.
    """
    reload_env_runtime()
    v = (os.getenv("P0_ISSUE_WATCH_DECLARE_ALSO_SEND_TO_GROUP") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def _normalize_major_check_person_token(raw: str) -> Tuple[str, str]:
    """
    Parse one entry from ``P0_MAJOR_CHECK_PERSON_IDS``.

    Returns ``(open_id, user_id)`` — one may be empty. Accepts ``ou_...``, ``u_...`` (→ ``ou_``),
    or tenant ``user_id`` (e.g. ``SNT0006`` or opaque hash without prefix).
    """
    s = (raw or "").strip()
    if not s:
        return "", ""
    if s.startswith("ou_") and is_open_id(s):
        return s, ""
    if s.startswith("u_") and len(s) > 2:
        candidate = f"ou_{s[2:]}"
        if is_open_id(candidate):
            return candidate, ""
    if is_open_id(s):
        return s, ""
    return "", s


def get_p0_major_check_person_recipients() -> List[Tuple[str, str]]:
    """
    ``P0_MAJOR_CHECK_PERSON_IDS`` — comma-separated ``ou_...`` / ``u_...`` / ``user_id``.

    When set, Issue Watch **Declare as P0** DMs these users (no @-mention thread confirm).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_MAJOR_CHECK_PERSON_IDS") or "").strip()
    if not raw:
        return []
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for part in raw.split(","):
        oid, uid = _normalize_major_check_person_token(part)
        key = oid or uid
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((oid, uid))
    return out


def get_p0_issue_watch_declare_check_person_reply_text() -> str:
    """Thread reply on the concern when duty declares P0 (check-person flow)."""
    reload_env_runtime()
    raw = (os.getenv("P0_ISSUE_WATCH_DECLARE_CHECK_PERSON_REPLY_TEXT") or "").strip()
    if raw:
        return raw
    return (
        "Calling and inviting the check persons for major P0 issues. Thank you."
    )


def get_p0_major_check_person_dm_enabled() -> bool:
    """
    ``P0_MAJOR_CHECK_PERSON_DM_ENABLED`` — DM check persons a meeting link after declare.

    Default **off**. Invite is via **VC ring** when duty joins the meeting
    (``P0_VC_RING_ENABLED=1`` + ``P0_MAJOR_CHECK_PERSON_IDS``).
    Set ``1`` only if you also want a separate DM with ``{link}``.
    """
    reload_env_runtime()
    v = (os.getenv("P0_MAJOR_CHECK_PERSON_DM_ENABLED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_major_check_person_dm_text() -> str:
    """DM body when ``P0_MAJOR_CHECK_PERSON_DM_ENABLED=1``. Use ``{link}`` for meeting URL."""
    reload_env_runtime()
    raw = (os.getenv("P0_MAJOR_CHECK_PERSON_DM_TEXT") or "").strip()
    if raw:
        return raw
    return (
        "Major P0 declared — please join the bridge meeting: {link}"
    )


def get_p0_major_check_person_join_thread_text() -> str:
    """Reply on the concern thread when a check person joins VC. Use ``{name}``."""
    reload_env_runtime()
    raw = (os.getenv("P0_MAJOR_CHECK_PERSON_JOIN_THREAD_TEXT") or "").strip()
    if raw:
        return raw
    return "{name} is already in the P0 meeting."


def p0_thread_confirm_target_mentions_enabled() -> bool:
    """
    When ``P0_MAJOR_CHECK_PERSON_IDS`` is set, @-mention thread confirm arming is off by default.

    Set ``P0_THREAD_CONFIRM_ALLOW_TARGET_MENTIONS=1`` to keep legacy @-tag arming anyway.
    """
    reload_env_runtime()
    if get_p0_major_check_person_recipients():
        v = (os.getenv("P0_THREAD_CONFIRM_ALLOW_TARGET_MENTIONS") or "0").strip().lower()
        return v in ("1", "true", "yes", "on")
    v = (os.getenv("P0_THREAD_CONFIRM_ALLOW_TARGET_MENTIONS") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_lark_bot_open_id() -> str:
    """``LARK_BOT_OPEN_ID`` — optional ``ou_...`` for this bot (excluded from VC ring @mentions)."""
    reload_env_runtime()
    return (os.getenv("LARK_BOT_OPEN_ID") or "").strip()


def get_p0_vc_ring_enabled() -> bool:
    """``P0_VC_RING_ENABLED`` — ring users into VC when duty joins (needs OAuth + ``vc:meeting``)."""
    reload_env_runtime()
    v = (os.getenv("P0_VC_RING_ENABLED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_vc_ring_fallback_open_ids() -> List[str]:
    """``P0_VC_RING_FALLBACK_OPEN_IDS`` — comma-separated ``ou_...`` when concern has no @mention."""
    reload_env_runtime()
    raw = (os.getenv("P0_VC_RING_FALLBACK_OPEN_IDS") or "").strip()
    if not raw:
        return []
    out: List[str] = []
    seen: set = set()
    for part in raw.split(","):
        oid = (part or "").strip()
        if oid.startswith("ou_") and oid not in seen:
            seen.add(oid)
            out.append(oid)
    return out


def get_p0_vc_oauth_public_base_url() -> str:
    reload_env_runtime()
    return (os.getenv("P0_VC_OAUTH_PUBLIC_BASE_URL") or "").strip().rstrip("/")


def get_p0_vc_oauth_redirect_uri() -> str:
    reload_env_runtime()
    return (os.getenv("P0_VC_OAUTH_REDIRECT_URI") or "").strip()


def get_p0_vc_oauth_scope() -> str:
    """
    ``P0_VC_OAUTH_SCOPE`` — Lark user OAuth scopes for VC ring + recording fan-out.

    Default includes ``vc:meeting``, ``vc:record``, and Drive permission scopes for Minutes edit.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_VC_OAUTH_SCOPE") or "").strip()
    if raw:
        return raw
    return (
        "vc:meeting offline_access vc:record "
        "docs:permission.member:create docs:permission.member:update"
    )


def p0_adjustment_bitable_enabled() -> bool:
    """``P0_ADJUSTMENT_BITABLE_ENABLED`` — post deployment notice after Send overview. Default ``1`` when app_token set."""
    reload_env_runtime()
    v = (os.getenv("P0_ADJUSTMENT_BITABLE_ENABLED") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return bool(get_p0_adjustment_bitable_app_token())


def get_p0_adjustment_bitable_app_token() -> str:
    reload_env_runtime()
    return (os.getenv("P0_ADJUSTMENT_BITABLE_APP_TOKEN") or "").strip()


def get_p0_adjustment_bitable_table_id() -> str:
    reload_env_runtime()
    return (os.getenv("P0_ADJUSTMENT_BITABLE_TABLE_ID") or "").strip()


def get_p0_adjustment_bitable_post_chat_id() -> str:
    """``P0_ADJUSTMENT_BITABLE_POST_CHAT_ID`` — fixed ``oc_...`` for 📦/🔴 cards; blank = follow session/overview routing."""
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_POST_CHAT_ID") or "").strip()
    return raw if raw.startswith("oc_") else ""


def resolve_p0_adjustment_bitable_post_chat_id(
    *,
    fallback_chat_id: str,
    source_incident_chat_id: str = "",
) -> str:
    """
    Destination for Bitable boss cards.

    1. ``P0_ADJUSTMENT_BITABLE_POST_CHAT_ID`` when set (fixed hub group).
    2. Else ``fallback_chat_id`` (meeting-card chat on P0 declare, overview dest on Send).
    3. Else ``source_incident_chat_id``.
    """
    fixed = get_p0_adjustment_bitable_post_chat_id()
    if fixed:
        return fixed
    fb = (fallback_chat_id or "").strip()
    if fb.startswith("oc_"):
        return fb
    sid = (source_incident_chat_id or "").strip()
    if sid.startswith("oc_"):
        return sid
    return ""


def get_p0_adjustment_bitable_hours() -> int:
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_HOURS") or "48").strip()
    try:
        h = int(raw)
        return max(1, min(h, 168))
    except ValueError:
        return 48


def get_p0_adjustment_bitable_max_rows() -> int:
    """
    ``P0_ADJUSTMENT_BITABLE_MAX_ROWS`` — legacy cap for **both** tables when > 0.
    Prefer ``P0_ADJUSTMENT_BITABLE_OPS_MAX_ROWS`` / ``DEPLOY_MAX_ROWS`` (defaults apply).
    ``0`` = use per-table defaults only.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_MAX_ROWS") or "0").strip()
    try:
        n = int(raw)
        return max(0, n)
    except ValueError:
        return 0


def get_p0_adjustment_bitable_ops_max_rows() -> int:
    """Cap ops rows on card. Default **8**. ``0`` = no limit."""
    reload_env_runtime()
    legacy = get_p0_adjustment_bitable_max_rows()
    if legacy > 0:
        return legacy
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_OPS_MAX_ROWS") or "8").strip()
    try:
        n = int(raw)
        return max(0, n)
    except ValueError:
        return 8


def get_p0_adjustment_bitable_deploy_max_rows() -> int:
    """Cap deployment rows (before pagination). Default **16** (~2 pages). ``0`` = no limit."""
    reload_env_runtime()
    legacy = get_p0_adjustment_bitable_max_rows()
    if legacy > 0:
        return legacy
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_DEPLOY_MAX_ROWS") or "16").strip()
    try:
        n = int(raw)
        return max(0, n)
    except ValueError:
        return 16


def get_p0_adjustment_bitable_timezone_name() -> str:
    """
    IANA zone for Bitable window + displayed timestamps. Default **Malaysia Time**
    (``Asia/Kuala_Lumpur``, MYT).
    """
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_TIMEZONE") or "").strip()
    return raw if raw else "Asia/Kuala_Lumpur"


def get_p0_adjustment_bitable_tz_label() -> str:
    """Short label on cards/logs (default ``MYT``)."""
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_TZ_LABEL") or "").strip()
    return raw if raw else "MYT"


def get_p0_adjustment_bitable_doc_url() -> str:
    reload_env_runtime()
    return (
        os.getenv("P0_ADJUSTMENT_BITABLE_DOC_URL")
        or "https://casinoplus.sg.larksuite.com/base/LVrubE8f8af1yTslQgqlIaWPgcg?table=tblHHa3NmHmWian6&view=vewRD952Gw"
    ).strip()


def get_p0_adjustment_bitable_all_fields_table_ids() -> Tuple[str, ...]:
    """
    Bitable table IDs that show **every column** on the deployment card.

    Comma-separated ``P0_ADJUSTMENT_BITABLE_ALL_FIELDS_TABLE_IDS``.
    Default: ``tblHHa3NmHmWian6`` (new Deployments base). Other tables (e.g. 线上操作)
    keep the fixed column set unless their table id is listed here too.
    """
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_ALL_FIELDS_TABLE_IDS") or "").strip()
    if raw:
        parts = tuple(x.strip() for x in raw.split(",") if x.strip())
        return parts if parts else ("tblHHa3NmHmWian6",)
    return ("tblHHa3NmHmWian6",)


def p0_adjustment_bitable_reply_in_thread() -> bool:
    """``P0_ADJUSTMENT_BITABLE_REPLY_IN_THREAD`` — reply under overview card (default ``1``)."""
    reload_env_runtime()
    v = (os.getenv("P0_ADJUSTMENT_BITABLE_REPLY_IN_THREAD") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def p0_adjustment_bitable_thread_followups() -> bool:
    """
    ``P0_ADJUSTMENT_BITABLE_THREAD_FOLLOWUPS`` — first ops/deploy card in main group feed;
    remaining pages reply in that message's thread only (default ``1``).
    """
    reload_env_runtime()
    v = (os.getenv("P0_ADJUSTMENT_BITABLE_THREAD_FOLLOWUPS") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def p0_adjustment_bitable_on_p0_declare() -> bool:
    """Post ops + deployment cards when P0 is declared (default on when bitable enabled)."""
    reload_env_runtime()
    v = (os.getenv("P0_ADJUSTMENT_BITABLE_ON_P0_DECLARE") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_p0_adjustment_bitable_deploy_page_size() -> int:
    """Rows per deployment card page (max 8). Default 8."""
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_DEPLOY_PAGE_SIZE") or "8").strip()
    try:
        n = int(raw)
        return max(1, min(n, 8))
    except ValueError:
        return 8


def get_p0_adjustment_bitable_ops_page_size() -> int:
    """Rows per ops card page (max 8). Default 8."""
    reload_env_runtime()
    raw = (os.getenv("P0_ADJUSTMENT_BITABLE_OPS_PAGE_SIZE") or "8").strip()
    try:
        n = int(raw)
        return max(1, min(n, 8))
    except ValueError:
        return 8


def p0_adjustment_bitable_also_send_to_group() -> bool:
    """
    ``P0_ADJUSTMENT_BITABLE_ALSO_SEND_TO_GROUP`` — legacy path only: after thread reply on
    the overview card, also duplicate the notice in the main group feed. Default ``0``.
    """
    reload_env_runtime()
    v = (os.getenv("P0_ADJUSTMENT_BITABLE_ALSO_SEND_TO_GROUP") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _split_field_aliases(raw: str, default: str) -> Tuple[str, ...]:
    text = (raw or default).strip()
    parts = tuple(x.strip() for x in text.split("|") if x.strip())
    return parts or (default,)


def get_p0_adjustment_bitable_field_names() -> Dict[str, Tuple[str, ...]]:
    """
    Bitable column names (pipe-separated aliases). Only **Blue Green Time** and
    **Full Release Time** are used for the lookback window check.
    """
    reload_env_runtime()
    return {
        "blue_green_time": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_BLUE_GREEN_TIME_FIELD") or "",
            "Blue Green Time|蓝绿发布时间",
        ),
        "full_release_time": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_FULL_RELEASE_TIME_FIELD") or "",
            "Full Release Time|全量发布时间",
        ),
        "service": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_SERVICE_FIELD") or "",
            "Service|服务",
        ),
        "namespace": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_NAMESPACE_FIELD") or "",
            "Namespace|命名空间",
        ),
        "image_tag": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_IMAGE_TAG_FIELD") or "",
            "Image Tag|镜像标签|Image",
        ),
        "project": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_PROJECT_FIELD") or "",
            "Project|项目",
        ),
        "version": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_VERSION_FIELD") or "",
            "Version|版本|Image Tag|镜像标签",
        ),
        "pm": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_PM_FIELD") or "",
            "PM|产品经理|Product Manager",
        ),
        "email": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_EMAIL_FIELD") or "",
            "Email|Release Title|邮件标题|Release",
        ),
        "changelog": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_CHANGELOG_FIELD") or "",
            "Changelog|更新内容|Change log|Notes",
        ),
    }


def get_p0_adjustment_bitable_ops_table_id() -> str:
    reload_env_runtime()
    return (os.getenv("P0_ADJUSTMENT_BITABLE_OPS_TABLE_ID") or "").strip()


def get_p0_adjustment_bitable_ops_doc_url() -> str:
    reload_env_runtime()
    return (
        os.getenv("P0_ADJUSTMENT_BITABLE_OPS_DOC_URL")
        or "https://casinoplus.sg.larksuite.com/base/LVrubE8f8af1yTslQgqlIaWPgcg?table=tblTNzlhFdyrKgG8&view=vew3eqLIWs"
    ).strip()


def get_p0_adjustment_bitable_ops_field_names() -> Dict[str, Tuple[str, ...]]:
    """
    Column names for the **线上操作** Bitable (``tblTNzlhFdyrKgG8``).

    Window check: **执行操作时间** and **执行完毕时间**.
    Card body: 执行操作, 执行操作时间, 项目, 执行原因, 执行完毕时间.
    """
    reload_env_runtime()
    return {
        "op_start_time": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_OPS_START_TIME_FIELD") or "",
            "执行操作时间",
        ),
        "op_done_time": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_OPS_DONE_TIME_FIELD") or "",
            "执行完毕时间",
        ),
        "operation": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_OPS_OPERATION_FIELD") or "",
            "执行操作",
        ),
        "project": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_OPS_PROJECT_FIELD") or "",
            "项目",
        ),
        "reason": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_OPS_REASON_FIELD") or "",
            "执行原因",
        ),
        "operator": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_OPS_OPERATOR_FIELD") or "",
            "操作人员|Operator",
        ),
        "status": _split_field_aliases(
            os.getenv("P0_ADJUSTMENT_BITABLE_OPS_STATUS_FIELD") or "",
            "执行状况阶段|Status",
        ),
    }


def get_p0_adjustment_bitable_sources() -> Tuple[Tuple[str, str, str, str, Dict[str, Tuple[str, ...]]], ...]:
    """
    Configured Bitable tables to check after Send overview.

    Each entry: ``(source_id, table_id, card_title, kind, field_names)``.
    ``kind`` is ``deployments`` or ``online_ops`` (controls columns + card subtitle).
    """
    reload_env_runtime()
    out: List[Tuple[str, str, str, str, Dict[str, Tuple[str, ...]]]] = []
    deploy_tbl = get_p0_adjustment_bitable_table_id()
    if deploy_tbl:
        out.append(
            (
                "deployments",
                deploy_tbl,
                "Deployments",
                "deployments",
                get_p0_adjustment_bitable_field_names(),
            )
        )
    ops_tbl = get_p0_adjustment_bitable_ops_table_id()
    if ops_tbl:
        out.append(
            (
                "online_ops",
                ops_tbl,
                "线上操作",
                "online_ops",
                get_p0_adjustment_bitable_ops_field_names(),
            )
        )
    return tuple(out)
