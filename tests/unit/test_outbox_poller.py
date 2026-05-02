from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from worker import outbox_poller


class FakeScalars:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def test_poll_outbox_once_marks_processed_and_failed(monkeypatch) -> None:
    success_message = SimpleNamespace(
        id=1,
        task_name="worker.tasks.process_application.process_application",
        task_args={"application_id": "app-1"},
        status="PENDING",
        created_at=datetime.now(UTC),
        processed_at=None,
        error_message=None,
    )
    failure_message = SimpleNamespace(
        id=2,
        task_name="worker.tasks.process_application.process_application",
        task_args={"application_id": "app-2"},
        status="FAILED",
        created_at=datetime.now(UTC),
        processed_at=None,
        error_message=None,
    )
    fake_session = SimpleNamespace(scalars=lambda statement: FakeScalars([success_message, failure_message]))

    @contextmanager
    def fake_get_sync_session():
        yield fake_session

    class FakeCelery:
        def send_task(self, task_name, args=None, task_id=None):
            if args == ["app-2"]:
                raise RuntimeError("send failed")

    monkeypatch.setattr(outbox_poller, "get_sync_session", fake_get_sync_session)
    monkeypatch.setattr("worker.celery_app.celery_app", FakeCelery())

    delivered = outbox_poller.poll_outbox_once()

    assert delivered == 1
    assert success_message.status == "PROCESSED"
    assert success_message.processed_at is not None
    assert failure_message.status == "FAILED"
    assert failure_message.error_message == "send failed"


def test_start_outbox_poller_is_idempotent(monkeypatch) -> None:
    started = []

    class FakeThread:
        def __init__(self, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append((self.name, self.daemon))

    monkeypatch.setattr(outbox_poller.threading, "Thread", FakeThread)
    monkeypatch.setattr(outbox_poller, "_poller_started", False)

    outbox_poller.start_outbox_poller()
    outbox_poller.start_outbox_poller()

    assert started == [("auditlend-outbox-poller", True)]
