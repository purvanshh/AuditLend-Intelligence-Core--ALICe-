from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from worker import celery_app


def test_health_handler_and_startup_hooks(monkeypatch) -> None:
    writes = BytesIO()
    sent = []

    handler = object.__new__(celery_app.HealthHandler)
    handler.path = "/health"
    handler.wfile = writes
    handler.send_response = lambda code: sent.append(("response", code))
    handler.send_header = lambda key, value: sent.append((key, value))
    handler.end_headers = lambda: sent.append(("end", None))
    handler.log_message("ignored")
    handler.do_GET()

    assert sent[0] == ("response", 200)
    assert b"auditlend-worker" in writes.getvalue()

    started = []

    class FakeServer:
        def __init__(self, addr, handler_cls):
            self.addr = addr
            self.handler_cls = handler_cls

        def serve_forever(self):
            started.append(("serve", self.addr))

    class FakeThread:
        def __init__(self, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append((self.name, self.daemon))

    monkeypatch.setattr(celery_app, "HTTPServer", FakeServer)
    monkeypatch.setattr(celery_app.threading, "Thread", FakeThread)
    monkeypatch.setattr(celery_app, "_health_server_started", False)
    monkeypatch.setattr(celery_app.logger, "info", lambda *args, **kwargs: None)

    celery_app.start_health_server(port=8123)
    celery_app.start_health_server(port=8123)

    assert started == [("auditlend-worker-health", True)]


def test_worker_ready_hook_starts_health_and_outbox(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(celery_app, "start_health_server", lambda: calls.append("health"))
    monkeypatch.setattr("worker.outbox_poller.start_outbox_poller", lambda: calls.append("outbox"))

    celery_app._start_outbox_poller_on_worker_ready()

    assert calls == ["health", "outbox"]
