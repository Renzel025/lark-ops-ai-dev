"""
Text normalization, player count extraction, impact scope, and issue scrubbing.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from . import config as _config

OPEN_ID_RE = _config.OPEN_ID_RE
CJK_RE = _config.CJK_RE
ID_RE = _config.ID_RE
NOT_SPECIFIED_RE = _config.NOT_SPECIFIED_RE
PLAYER_COUNT_PATTERNS = _config.PLAYER_COUNT_PATTERNS
PLAYER_VAGUE_PATTERNS = _config.PLAYER_VAGUE_PATTERNS
PLAYER_VAGUE_LABELS = _config.PLAYER_VAGUE_LABELS


def is_open_id(x: str) -> bool:
    return bool(OPEN_ID_RE.match((x or "").strip()))


def looks_like_chinese(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    cjk = len(CJK_RE.findall(s))
    return cjk >= 6 or (cjk / max(len(s), 1)) > 0.15


def is_not_specified(s: str) -> bool:
    return bool(NOT_SPECIFIED_RE.match((s or "").strip()))


def format_bare_player_count_phrase(text: str) -> Optional[str]:
    """
    If text is only a numeric player count (e.g. ``5000``, ``5,000``, ``5000 players``),
    return a full phrase for the overview card / translation.
    """
    src = (text or "").strip()
    if not src:
        return None
    m = re.fullmatch(r"(\d{1,3}(?:,\d{3})+|\d{2,9})(?:\s*(?:players?|users?|accounts?))?\s*", src, re.IGNORECASE)
    if not m:
        return None
    try:
        n = int(m.group(1).replace(",", ""))
    except ValueError:
        return None
    if 1900 <= n <= 2100:
        return None
    if n < 1 or n > 50_000_000:
        return None
    return f"{n} affected players"


def normalize_impact_scope_manual(text: str) -> str:
    """User-edited impact scope from DM or card form (plain text, one line)."""
    src = (text or "").strip()
    if not src or is_not_specified(src):
        return "Not specified"
    bare = format_bare_player_count_phrase(src)
    if bare:
        return bare
    return normalize_gaming_zh(clean_single_line_translation(src))[:500].strip() or "Not specified"


def normalize_issue_manual(text: str) -> str:
    """User-edited issue summary from card form (single-line input; whitespace collapsed)."""
    src = (text or "").strip()
    if not src or is_not_specified(src):
        return "Not specified"
    s = re.sub(r"\s+", " ", src.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")).strip()
    s = normalize_gaming_zh(s)
    return s[:2000].strip() or "Not specified"


def normalize_gaming_zh(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    s = s.replace("球员", "玩家")
    s = s.replace("受影响的球员", "受影响的玩家")
    s = s.replace("影响范围：", "影响范围:")
    return s


def clean_single_line_translation(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    s = re.sub(r"^(翻译[:：]\s*|中文[:：]\s*|问题[:：]\s*|影响范围[:：]\s*)", "", s).strip()
    return s[:300].strip()


def strip_env_quotes(s: str) -> str:
    s = (s or "").strip()
    if (len(s) >= 2) and (s[0] == s[-1]) and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s


def clean_pasted_text(text: Optional[str]) -> str:
    import json
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            if isinstance(obj.get("text"), str):
                s = obj["text"]
            elif isinstance(obj.get("content"), str):
                s = obj["content"]
        elif isinstance(obj, str):
            s = obj
    except Exception:
        pass
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def norm_name(s: str) -> str:
    s = (s or "").strip()
    s = s.lstrip("@").strip()
    s = re.sub(r"[,:;)\]\}]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def norm_person_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("@", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_lookup_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^(sir|mr|ms|mrs)\s+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def extract_player_ids(text: str) -> List[str]:
    ids = ID_RE.findall(text or "")
    seen = set()
    out = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def extract_player_count_from_text(text: str) -> Tuple[Optional[int], Optional[str]]:
    src = (text or "").strip()
    if not src:
        return None, None
    for builder, pat in PLAYER_COUNT_PATTERNS:
        m = pat.search(src)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        return builder(n)
    for kind, pat in PLAYER_VAGUE_PATTERNS:
        if pat.search(src):
            return None, PLAYER_VAGUE_LABELS.get(kind, "Multiple affected players")
    return None, None


def _score_issue_block(text: str) -> int:
    s = (text or "").strip()
    if not s:
        return -10
    lower = s.lower()
    score = 0
    if len(s) >= 30:
        score += 1
    if len(s) >= 80:
        score += 2
    player_terms = [
        "player", "players", "unable", "cannot", "can't", "failed", "failure",
        "error", "issue", "problem", "cannot enter", "unable to access",
        "unable to login", "cannot login", "not working", "stuck", "missing",
        "gone", "disappear", "lost", "reward", "voucher", "bonus", "claim",
        "redeem", "login", "access", "game", "session",
    ]
    for term in player_terms:
        if term in lower:
            score += 2
    noise_terms = [
        "reply to", "view earlier", "edited", "owner:", "meeting id",
        "recordings (minutes)", "you joined the meeting", "you left the meeting",
    ]
    for term in noise_terms:
        if term in lower:
            score -= 3
    if re.fullmatch(r"[\d\s,]+", s):
        score -= 10
    return score


def _pick_best_issue_text(text: str) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", src) if b.strip()]
    if not blocks:
        return src
    return max(blocks, key=_score_issue_block)


def issue_context_for_model(text: str, max_chars: int = 2500) -> str:
    """
    Incident text for the issue-summary LLM: every non-noise paragraph, best-scored first.
    Picking only the single best block lost the actual symptom whenever an impact-only paragraph
    ("more than 5 players reporting the problem") out-scored it — the model then summarized the impact.
    """
    src = (text or "").strip()
    if not src:
        return ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", src) if b.strip()]
    kept = [b for b in blocks if _score_issue_block(b) >= 0] or blocks
    kept.sort(key=_score_issue_block, reverse=True)
    return "\n\n".join(kept)[:max_chars]


def _impact_from_bare_count_lines(text: str) -> Optional[str]:
    """Detect lines that are only a player count (common when ops pastes a short note)."""
    src = (text or "").strip()
    if not src:
        return None
    for raw in src.splitlines():
        hit = format_bare_player_count_phrase(raw.strip())
        if hit:
            return hit
    return format_bare_player_count_phrase(src)


def build_impact_scope(text: str) -> str:
    ids = extract_player_ids(text)
    n, label = extract_player_count_from_text(text)
    # A thread's first report lists 2 IDs, a later reply says "more than 5 players" — the stated
    # count is the newer, larger picture, so it wins over the ID tally.
    if ids and not (n and label and n > len(ids)):
        return f"{len(ids)} affected players"
    if label:
        return label
    bare_line = _impact_from_bare_count_lines(text)
    if bare_line:
        return bare_line
    # No player IDs, no count, no vague phrase — nothing about impact in the source at all.
    return _config.get_overview_impact_scope_default()


OVERVIEW_TEMPLATE_MARKERS = (
    "incident overview",
    "事故概览",
    "incident start",
    "事故开始",
    "impact scope",
    "影响范围",
    "support request",
    "支援请求",
    "reported time",
    "reporter",
    "affected players",
    "affected user",
)


def is_manual_p0_incident_overview_template(text: str) -> bool:
    """
    True for a pasted/re-sent **P0/P1 Incident Overview** block — the bot's own overview format
    (Send overview / manual share), whether a human pasted it or it's a reply quoting one. Shared by
    the keyword-trigger path (skip re-declaring off a paste) and Issue Watch (skip re-detecting the
    incident it was written to describe as a brand-new signal).

    Originally we only checked the first line for ``P0 Incident Overview`` / ``P0 事故概览`` — but
    real pastes vary (leading emoji like ``📍``, ``**`` markdown, quoted-reply prefixes, custom
    templates with ``🕐`` instead of ``🕒``, etc.) so we use two layers:

    Layer 1 — first-line title check (tolerant to leading emoji / decoration / markdown).
    Layer 2 — multi-marker heuristic: if the body contains 2+ overview field markers
              (``Issue / Impact Scope / Support Request / Reported Time / 事故概览 / 影响范围 / …``),
              treat as a manual overview paste even if the title line is malformed or missing.
    """
    t = (text or "").strip()
    if not t:
        return False

    first = t.split("\n")[0].strip()
    first_clean = re.sub(r"^[^A-Za-z\u4e00-\u9fff]+", "", first).strip()
    if first_clean:
        if re.match(r"(?is)(?:P0|P1)\s+Incident\s+Overview\b", first_clean):
            return True
        if re.match(r"(?is)(?:P0|P1)\s*事故概览", first_clean):
            return True

    body_lc = t.lower()
    marker_hits = sum(1 for m in OVERVIEW_TEMPLATE_MARKERS if m in body_lc)
    return marker_hits >= 2
