"""Per-agent "reasoning dashboard" — the full, un-truncated chain-of-thought / assessment
text for Nemotron and Cosmos used to be rendered inline in the main Physyk dashboard's Agent
Plan panel as big paragraphs. That made the step list hard to scan at a glance. This module
gives each reasoning-producing agent its own small page with the full text, while the main
dashboard now only shows a short 1-2 line summary + a link out to here.

Originally this ran each agent's page on its own dedicated port (8010/8011). That broke for
anyone accessing the dashboard through Brev/VS Code port forwarding — only 7860 (and 8211 for
Isaac Sim's own WebRTC stream) are actually forwarded (see BREV_PORT_CONFIG.md), so those
extra ports were simply unreachable remotely even though they worked fine over plain
localhost. Fixed by mounting these as ordinary routes on the SAME FastAPI app already serving
port 7860 instead — /reasoning/<agent> — so they ride along with whatever forwarding already
gets the main dashboard through. No new ports, nothing extra to expose.
"""
import time
import threading
from collections import deque
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

MAX_ENTRIES = 200


class ReasoningLog:
    def __init__(self, name: str, accent: str):
        self.name = name
        self.accent = accent
        self._entries = deque(maxlen=MAX_ENTRIES)
        self._lock = threading.Lock()

    def add(self, label: str, text: Optional[str]):
        if not text:
            return
        with self._lock:
            self._entries.appendleft({
                "t": time.strftime("%H:%M:%S"),
                "label": label,
                "text": text,
            })

    def snapshot(self):
        with self._lock:
            return list(self._entries)

    def render_html(self) -> str:
        rows = "".join(
            f"""<div class="entry">
                  <div class="meta"><span class="t">{e['t']}</span> <span class="label">{e['label']}</span></div>
                  <div class="text">{e['text']}</div>
                </div>"""
            for e in self.snapshot()
        )
        if not rows:
            rows = '<div class="empty">No reasoning captured yet — run an instruction on the main dashboard.</div>'
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>{self.name} — Reasoning</title>
<style>
  body {{ background:#0a0c10; color:#e5e7eb; font-family:'JetBrains Mono',ui-monospace,monospace;
         margin:0; padding:24px; }}
  h1 {{ color:{self.accent}; font-size:1.1rem; margin:0 0 16px; letter-spacing:0.5px; }}
  .entry {{ border-left:3px solid {self.accent}; background:rgba(255,255,255,0.03);
            padding:10px 14px; margin-bottom:10px; border-radius:0 6px 6px 0; }}
  .meta {{ font-size:0.7rem; color:#9ca3af; margin-bottom:4px; }}
  .t {{ opacity:0.7; }}
  .label {{ font-weight:700; color:{self.accent}; }}
  .text {{ font-size:0.82rem; line-height:1.5; white-space:pre-wrap; }}
  .empty {{ color:#6b7280; font-style:italic; }}
</style></head>
<body>
  <h1>{self.name} — Full Reasoning Log</h1>
  {rows}
</body></html>"""


# Module-level singletons — created once regardless of import order between
# physyk_agent_orchestrator.py (pushes entries into these) and physyk_main_server.py (mounts
# routes for them once its FastAPI `app` exists). Both import from here.
NEMOTRON_REASONING_LOG = ReasoningLog("Nemotron Planning", "#60a5fa")
COSMOS_REASONING_LOG = ReasoningLog("Cosmos Reason 2", "#34d399")


def mount_reasoning_routes(app: FastAPI):
    """Registers /reasoning/nemotron and /reasoning/cosmos (+ their .json variants) on an
    already-created FastAPI app. Call this after `app = FastAPI(...)` in physyk_main_server.py."""
    logs = {"nemotron": NEMOTRON_REASONING_LOG, "cosmos": COSMOS_REASONING_LOG}

    @app.get("/reasoning/{agent}", response_class=HTMLResponse)
    def reasoning_page(agent: str):
        log = logs.get(agent)
        if log is None:
            return HTMLResponse(f"<h1>Unknown agent '{agent}'</h1>", status_code=404)
        return HTMLResponse(log.render_html())

    @app.get("/reasoning/{agent}/log.json")
    def reasoning_json(agent: str):
        log = logs.get(agent)
        if log is None:
            return JSONResponse({"error": f"unknown agent '{agent}'"}, status_code=404)
        return JSONResponse(log.snapshot())
