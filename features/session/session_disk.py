"""
Persist active P0/P1 session rows when ``P0_SHARED_STATE_DIR`` is set so multiple
workers share ``meeting_invite_message_id`` (PATCH in-place) and cannot double-start
meetings for the same incident group chat.
"""
from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from features.overview import draft_store as _ds

log = logging.getLogger("lark-ops-ai")

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None  # type: ignore


def enabled() -> bool:
    return _ds.disk_enabled()


def _safe_chat_id(chat_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in (chat_id or ""))[:220] or "unknown"


def _path(chat_id: str) -> str:
    base = _ds.shared_state_dir()
    return os.path.join(base, "sessions", f"{_safe_chat_id(chat_id)}.json")


@contextlib.contextmanager
def exclusive_lock(chat_id: str, *, timeout_sec: float = 5.0) -> Iterator[bool]:
    """Serialize ``start_p0`` for one incident chat (cross-process).

    Yields ``True`` when the lock was acquired, ``False`` if it could not be acquired
    within ``timeout_sec`` (another declare is already in progress for this chat).

    NEVER blocks forever: a wedged holder must not be able to pin every future declare
    for this chat. Callers should treat a ``False`` yield as "declare already running,
    skip" rather than proceeding without the lock.
    """
    chat_id = (chat_id or "").strip()
    if not enabled() or not _fcntl:
        yield True
        return
    path = _path(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "a+", encoding="utf-8") as lf:
        acquired = False
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            try:
                _fcntl.flock(lf, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    log.warning(
                        "session_disk: exclusive_lock timeout after %.1fs chat_tail=%s "
                        "(another declare in progress or a stuck holder)",
                        timeout_sec,
                        chat_id[-12:] if len(chat_id) > 12 else chat_id,
                    )
                    break
                time.sleep(0.1)
        try:
            yield acquired
        finally:
            if acquired:
                _fcntl.flock(lf, _fcntl.LOCK_UN)


def load_session(chat_id: str) -> Optional[Dict[str, Any]]:
    if not enabled():
        return None
    path = _path(chat_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data) if isinstance(data, dict) else None
    except Exception as e:
        log.warning("session_disk load failed path=%s err=%s", path, e)
        return None


def save_session(chat_id: str, sess: Dict[str, Any]) -> None:
    if not enabled():
        return
    path = _path(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sess, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as e:
        log.warning("session_disk save failed path=%s err=%s", path, e)


def delete_session(chat_id: str) -> None:
    if not enabled():
        return
    path = _path(chat_id)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("session_disk delete failed path=%s err=%s", path, e)


def find_session_by_meeting_no_disk(meeting_no: str) -> Tuple[str, Dict[str, Any]]:
    """Scan JSON files when in-memory ``P0_SESSIONS`` missed (another worker)."""
    meeting_no = (meeting_no or "").strip()
    out: Tuple[str, Dict[str, Any]] = ("", {})
    if not enabled() or not meeting_no:
        return out
    base = os.path.join(_ds.shared_state_dir(), "sessions")
    if not os.path.isdir(base):
        return out
    for fp in glob.glob(os.path.join(base, "*.json")):
        if fp.endswith(".json.tmp"):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        cur = str(data.get("meeting_no") or "").strip()
        if cur == meeting_no:
            cid = str(data.get("source_chat") or "").strip()
            if cid:
                return cid, data
    return out


def find_session_source_by_target_chat_disk(target_chat: str) -> Tuple[str, Dict[str, Any]]:
    """Scan persisted rows where ``target_chat`` matches the prompt / mirror ``oc_`` (cross-worker)."""
    target_chat = (target_chat or "").strip()
    out: Tuple[str, Dict[str, Any]] = ("", {})
    if not enabled() or not target_chat:
        return out
    base = os.path.join(_ds.shared_state_dir(), "sessions")
    if not os.path.isdir(base):
        return out
    matches: List[Tuple[str, Dict[str, Any]]] = []
    for fp in glob.glob(os.path.join(base, "*.json")):
        if fp.endswith(".json.tmp"):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        tt = str(data.get("target_chat") or "").strip()
        if tt == target_chat:
            cid = str(data.get("source_chat") or "").strip()
            if cid:
                matches.append((cid, data))
    if len(matches) > 1:
        log.warning(
            "session_disk: multiple active session files share target_chat=%s — using first source",
            target_chat[:28],
        )
    if matches:
        return matches[0]
    return out
