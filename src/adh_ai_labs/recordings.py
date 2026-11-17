"""Runs of the worked copy on a GPU, kept so the notebook still shows complete results without one.

Two kinds of recording live in ``assets/recordings/``:

- ``calls.json``: every chat request the worked copy sent and the response it got, keyed by a
  hash of the request body (everything but the model's name). When no model server can start,
  ``llm.start`` runs a small replay server at the model server's address instead: the
  student's own ``OpenAI(...)`` client talks to it unchanged, a request the worked copy made
  gets the response recorded on a GPU, and any other request gets an empty reply.
- ``<step>.json``: a batch step's result per note (an extraction of the notes, the tool on the
  audit patients). On the CPU the helpers run the student's own definitions live on as many
  notes as fit the time and fill in the rest from these, labelled as recorded on a GPU.

Each step recording stores a hash of the notes it was made from and the exact model inputs
(the system message, the tool's schema), so a stale recording can be detected when the notes
or the worked answers change. Running the worked copy with ADH_RECORD=1 on a machine with a
GPU makes them again: ``llm.start`` then puts a recording proxy between the notebook's client
and Ollama, which saves each request body and its response.
"""

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FOLDER = Path(__file__).parent / "assets" / "recordings"
CALLS = FOLDER / "calls.json"


def recording():
    """True while the worked copy is being run to make the recordings (on a GPU only)."""
    from . import llm
    return os.environ.get("ADH_RECORD") == "1" and llm.ON_GPU


def request_key(body):
    """A hash of a chat request's JSON body, ignoring which model it is sent to."""
    body = {k: v for k, v in body.items() if k not in ("model", "stream_options")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_step(name, inputs, rows):
    """Save a batch step's result per note, with the notes' hash and the model inputs used."""
    from . import notes
    FOLDER.mkdir(parents=True, exist_ok=True)
    payload = {"notes_sha": notes.sha(), "inputs": inputs, "rows": rows}
    (FOLDER / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


def step(name):
    """A saved step: {"notes_sha", "inputs", "rows"}, or None if there is none."""
    payload = _read(FOLDER / f"{name}.json")
    return payload or None


def empty_reply(model):
    """The body of a chat response with no content: what the replay server sends for an unrecorded request."""
    return {"id": "offline", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


class _Handler(BaseHTTPRequestHandler):
    """Replays recorded chat responses, or forwards to Ollama and records them (``server.upstream``)."""

    def log_message(self, *args):  # no request log in the notebook
        pass

    def _send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (the http.server name)
        if self.server.upstream:
            return self._forward(None)
        return self._send(200, {"version": "replay"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.server.upstream:
            return self._forward(raw)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return self._send(400, {"error": "not JSON"})
        if body.get("stream"):  # a stream with nothing in it
            return self._send(200, b"data: [DONE]\n\n", "text/event-stream")
        saved = self.server.calls.get(request_key(body))
        return self._send(200, saved or empty_reply(body.get("model", "")))

    def _forward(self, raw):
        request = urllib.request.Request(self.server.upstream + self.path, data=raw, method=self.command,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as reply:
                status, content_type, data = reply.status, reply.headers.get("Content-Type", ""), reply.read()
        except urllib.error.HTTPError as error:
            status, content_type, data = error.code, error.headers.get("Content-Type", ""), error.read()
        if raw and status == 200 and self.path.endswith("/chat/completions"):
            body = json.loads(raw)
            if not body.get("stream"):
                self.server.record(body, json.loads(data))
        self._send(status, data, content_type or "application/json")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, upstream):
        super().__init__(address, _Handler)
        self.upstream = upstream
        self.calls = {} if upstream else _read(CALLS).get("calls", {})
        self._lock = threading.Lock()

    def record(self, body, response):
        from . import notes
        with self._lock:
            # The key ignores the model's name (so the CPU path can replay it), so when a second model is
            # sent the same request (Extension E) the first, the worked copy's own model, is the one kept.
            self.calls.setdefault(request_key(body), response)
            FOLDER.mkdir(parents=True, exist_ok=True)
            CALLS.write_text(json.dumps({"notes_sha": notes.sha(), "calls": self.calls}, sort_keys=True) + "\n")


def serve(host, port, upstream=None):
    """Start the replay server (or, with ``upstream``, the recording proxy) in the background; return it."""
    server = _Server((host, port), upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
