from services import logging as logging_module
from services.logging import _add_message_field


def test_logging_processor_adds_message_and_application_id() -> None:
    event = _add_message_field(None, "info", {"event": "external_service_success"})

    assert event["message"] == "external_service_success"
    assert event["application_id"] is None


def test_logging_helpers_cover_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    logging_module.setup_logging()
    logger = logging_module.get_logger("auditlend", application_id="app-1")

    assert logger is not None
    assert logging_module.log_level() == "DEBUG"
