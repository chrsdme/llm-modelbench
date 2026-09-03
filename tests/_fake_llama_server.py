"""Test-only fake ``llama-server``-like process for Stage 3B.3C.

Spawned as a real subprocess (``python -m tests._fake_llama_server ...``) so
the managed-materialisation path exercises real ``Popen`` lifecycle, real
PID / ``/proc`` reads, real localhost sockets, real readiness polling, and
real graceful/forced termination -- without CUDA, llama.cpp, Ollama, or a
model.

Behaviours (``--behaviour``):

* ``healthy``           -- bind, serve ``/health`` = 200 immediately, run
                          until signalled.
* ``delayed:<seconds>`` -- bind immediately but answer ``/health`` with 503
                          until ``<seconds>`` have elapsed, then 200.
* ``immediate-exit``    -- exit(3) at once, never bind.
* ``never-ready``       -- bind, always answer ``/health`` with 503.
* ``wrong-service``     -- bind, answer every path with a non-llama JSON body
                          (200) so a naive "HTTP 200" check would pass.
* ``ignore-sigterm``    -- like ``healthy`` but ignores SIGTERM (only SIGKILL
                          stops it) -- for forced-cleanup tests.
* ``flood``             -- write ~256 KiB to stdout (well past a pipe buffer)
                          *before* binding, then behave like ``healthy`` --
                          proves the diagnostic drain thread keeps the child
                          from blocking on a full pipe.

``/health`` mimics llama-server: ``{"status": "ok"}`` on ready.
``/props`` returns ``{"default_generation_settings": {"n_ctx": <ctx>}}`` so a
context-conformance check has something to read; ``--ctx`` sets the value.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: A003 -- silence test noise
            return

        def _json(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            behaviour = state["behaviour"]
            if behaviour == "wrong-service":
                # Answer 200 with a *valid* JSON body that is unmistakably not
                # a llama-server: /health is not even an object, and /props
                # lacks default_generation_settings. A naive "HTTP 200" check
                # would still pass.
                if self.path == "/health":
                    self._json(200, ["not", "an", "object"])
                else:
                    self._json(200, {"service": "not-llama", "app": "something-else"})
                return
            if self.path == "/health":
                ready = True
                if behaviour == "never-ready":
                    ready = False
                elif behaviour.startswith("delayed:"):
                    ready = time.monotonic() >= state["ready_at"]
                if ready:
                    self._json(200, {"status": "ok"})
                else:
                    self._json(503, {"status": "loading model"})
                return
            if self.path == "/props":
                self._json(
                    200,
                    {
                        "default_generation_settings": {"n_ctx": state["ctx"]},
                        "build_info": "fake",
                    },
                )
                return
            if self.path == "/v1/models":
                self._json(200, {"data": [{"id": "fake-model"}]})
                return
            self._json(404, {"error": "not found"})

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--behaviour", default="healthy")
    parser.add_argument("--ctx", type=int, default=0)
    # Accept-and-ignore the flags the real command builder emits, so the
    # fake's argv looks like a real launch to /proc/<pid>/cmdline.
    parser.add_argument("--model", default=None)
    parser.add_argument("--ctx-size", type=int, default=None)
    parser.add_argument("--split-mode", default=None)
    args = parser.parse_args(argv)

    behaviour = args.behaviour
    if args.ctx_size is not None and not args.ctx:
        args.ctx = args.ctx_size

    if behaviour == "immediate-exit":
        return 3

    if behaviour == "flood":
        # Emit well past a Linux pipe buffer (64 KiB default) before binding.
        # If the parent is not draining stdout, this write blocks and the
        # server never comes up.
        sys.stdout.write("x" * (256 * 1024))
        sys.stdout.write("\n")
        sys.stdout.flush()
        behaviour = "healthy"

    state = {"behaviour": behaviour, "ctx": args.ctx, "ready_at": time.monotonic()}
    if behaviour.startswith("delayed:"):
        state["ready_at"] = time.monotonic() + float(behaviour.split(":", 1)[1])

    if behaviour == "ignore-sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    stop = threading.Event()

    def _graceful(signum, frame):
        stop.set()

    if behaviour != "ignore-sigterm":
        signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not stop.is_set():
            time.sleep(0.02)
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
