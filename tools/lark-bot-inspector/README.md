# lark-bot-inspector — read-only MCP server for the bot boxes

A small [MCP](https://modelcontextprotocol.io) server that lets Claude Code look at the running
lark-ops-ai bot on the **dev** and **prod** VPS: service logs, active P0/P1 sessions, DM
drafts/previews, Issue Watch mute state, feature flags and the deployed commit.

It is **our own code** (`server.py` in this folder) — not built into Claude Code and not a Lark
product. It never calls the Lark API. Every tool SSHes into a box and runs one fixed, read-only
command.

**It cannot change anything.** No tool restarts the service, edits a file, or posts to Lark.
Secret-looking env vars are refused outright.

---

## How it works

```
Claude Code ──stdio──▶ server.py (local, .venv) ──ssh──▶ bot-dev / bot-prod ──sudo──▶ journalctl / cat / grep / git log
```

1. Claude Code reads the repo-root `.mcp.json` and launches `server.py` when a session starts.
2. Each `@mcp.tool()` function in `server.py` becomes a tool Claude can call.
3. A tool builds one shell command and hands it to `_ssh(box, cmd)`, which runs:

   ```bash
   ssh -o RemoteCommand=none -o BatchMode=yes -o ConnectTimeout=15 -T bot-prod \
     "sudo -n bash -c '<remote command>'"
   ```

   | Option | Why |
   |---|---|
   | `bot-prod` / `bot-dev` | Host aliases from `~/.ssh/config` (`box="prod"` → `bot-prod`) |
   | `RemoteCommand=none`, `-T` | Override the alias's interactive `sudo -i` / TTY settings |
   | `BatchMode=yes` | Never prompt — fail fast instead of hanging |
   | `sudo -n bash -c '…'` | Logs, `.env` and state files are root-owned; `-n` = never prompt for a password |

4. Output (stdout + stderr, capped at 20 000 chars) goes back to Claude.

### How Claude Code picks up `.mcp.json`

This happens once, at session start:

1. **Find** — Claude Code looks for `.mcp.json` at the root of the opened project
   (`lark-ops-ai-dev/`). Servers defined there are **project-scoped**: committed to git, shared with
   everyone who clones the repo.
2. **Approval check** — a committed file could make your machine run any command, so project
   servers never start automatically. Claude Code checks your local settings:
   - in `enabledMcpjsonServers`, or `enableAllProjectMcpServers: true` → allowed
   - in `disabledMcpjsonServers` (`.claude/settings.local.json`) → blocked
   - in neither → it asks at startup and saves your answer to one of those lists
3. **Launch** — it runs the `command` + `args` from `.mcp.json`
   (`tools/lark-bot-inspector/.venv/bin/python tools/lark-bot-inspector/server.py`), started from
   the project folder — which is why the relative paths work and `.venv` must be at that exact spot.
4. **Handshake over stdio** — Claude Code sends the MCP `initialize` request; `server.py`
   (`mcp.run()`) replies with its name and `instructions`. Claude Code then asks for the tool
   list and gets the 8 `@mcp.tool()` functions with their descriptions and parameters.
5. **Tools available** — they appear as `mcp__lark-bot-inspector__check_logs` etc., and `/mcp`
   shows the server as connected. If any step fails (missing `.venv`, Python error, bad JSON),
   `/mcp` shows it as failed and no tools appear.

Because this only runs at session start, **any change** — editing `.mcp.json` or `server.py`, or
re-enabling the server — needs a Claude Code restart.

---

## Setup from zero

### 1. SSH access to both boxes

Get the team `.pem` key, then add two aliases to `~/.ssh/config`:

```
Host bot-dev
    HostName <DEV_VPS_IP>
    User ose
    IdentityFile /path/to/ose-aliyun-access.pem
    RemoteCommand sudo -i
    RequestTTY yes

Host bot-prod
    HostName <PROD_VPS_IP>
    User ose
    IdentityFile /path/to/ose-aliyun-access.pem
    RemoteCommand sudo -i
    RequestTTY yes
```

The names must be exactly `bot-dev` and `bot-prod` — `server.py` uses them.

Check it works without a password prompt:

```bash
ssh -o RemoteCommand=none -o BatchMode=yes -T bot-dev "sudo -n true && echo ok"
ssh -o RemoteCommand=none -o BatchMode=yes -T bot-prod "sudo -n true && echo ok"
```

Both must print `ok`. If `sudo -n` fails, the `ose` user needs passwordless sudo (`NOPASSWD`) on
that box.

### 2. Python environment for the server

The `mcp` SDK needs Python 3.10+ (the bot itself runs on 3.8, so this gets its own venv). From the
repo root:

```bash
python3.12 -m venv tools/lark-bot-inspector/.venv
tools/lark-bot-inspector/.venv/bin/pip install -r tools/lark-bot-inspector/requirements.txt
```

`.venv` is git-ignored — every machine creates its own.

Smoke test (it should start and wait silently; Ctrl+C to quit):

```bash
tools/lark-bot-inspector/.venv/bin/python tools/lark-bot-inspector/server.py
```

### 3. Registration (already in the repo)

The repo-root `.mcp.json` registers the server at project scope:

```json
{
  "mcpServers": {
    "lark-bot-inspector": {
      "type": "stdio",
      "command": "tools/lark-bot-inspector/.venv/bin/python",
      "args": ["tools/lark-bot-inspector/server.py"],
      "env": {}
    }
  }
}
```

Nothing to do here — it comes with `git clone`.

### 4. Approve it in Claude Code

Start Claude Code in the repo. On the first start it asks whether to enable `lark-bot-inspector`
— choose **Yes**.

Then run `/mcp`: `lark-bot-inspector` should show as connected with 8 tools. Ask something like
*"check prod service status"* to confirm end to end.

---

## Tools

Paths below are prod (`/root/lark-ops-ai`); dev uses `/root/lark-ops-ai-dev`. The systemd unit is
`lark-ops-ai` on **both** boxes. `$D` is `P0_SHARED_STATE_DIR` read from the box's `.env`.

| Tool | What it shows | Remote command |
|---|---|---|
| `list_boxes` | Which `box` values exist | none (static text) |
| `service_status` | Service up?, deployed commit, box time | `systemctl is-active lark-ops-ai; git log --oneline -1; date` |
| `check_logs` | Bot logs for a time window, optional regex filter | `journalctl -u lark-ops-ai --since … [--until …] --no-pager \| grep -E … \| tail -n N` |
| `check_env_flag` | One flag line from `.env` (secrets refused) | `grep -nE '^FLAG=' .env` |
| `list_active_sessions` | Active P0/P1 session files | `ls -la "$D/sessions/"` |
| `get_session_detail` | One session's JSON (priority, meeting, ring targets…) | `find "$D/sessions/" -iname "*<chat tail>*.json"` → `cat` |
| `get_draft_or_preview` | A duty person's DM draft or preview JSON | same, in `$D/drafts/` or `$D/previews/` |
| `check_issue_watch_mute` | Issue Watch `/off` `/on` state | `cat "$D/issue_watch_mute.json"` |

### Safety guards in `server.py`

- Fixed command per tool — callers pass parameters, never shell.
- Every parameter is `shlex.quote`d; ids/tails must match `^[A-Za-z0-9._-]{1,200}$`.
- `check_env_flag` only accepts `^[A-Z][A-Z0-9_]*$` and refuses names containing `SECRET`,
  `TOKEN`, `KEY`, `PASSWORD`, `PASS`, `AUTH` or `CREDENTIAL` — no value, not even redacted.
- `check_logs` caps output at 2 000 lines; all output is capped at 20 000 chars.

---

## Adding a tool

1. Add a function to `server.py`:

   ```python
   @mcp.tool()
   def my_new_tool(box: Literal["dev", "prod"]) -> str:
       """One-line description — Claude reads this to decide when to call it."""
       cmd = "some read-only command"
       return _ssh(box, cmd)
   ```

   Keep it read-only, and run any user-supplied value through `_require_safe_token` /
   `shlex.quote`.
2. Restart Claude Code (servers load only at session start).
3. Check `/mcp` lists the new tool.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Tools don't appear at all | It was declined at startup and added to `disabledMcpjsonServers` in `.claude/settings.local.json`. Enable it via `/mcp` (or empty that list), then restart Claude Code. |
| `/mcp` shows it failed to start | `.venv` missing or built with Python < 3.10 — redo step 2. |
| `[ssh timed out …]` | VPN / network, or the box's IP is rate-limited from rapid probing. Try a manual `ssh bot-prod` first. |
| `sudo: a password is required` | `ose` lacks passwordless sudo on that box. |
| Plain `ssh bot-prod …` from Claude is blocked in auto mode | Expected — auto mode blocks direct prod access. Use these tools instead; they are read-only by design. |
