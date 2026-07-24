import os

from services.vault import VaultClient, create_pii_service_from_vault_or_env


def test_vault_client_creation() -> None:
    client = VaultClient(url="http://localhost:8200", token="s.test", mount_point="secret")
    assert client.url == "http://localhost:8200"
    assert client.mount_point == "secret"


def test_vault_client_graceful_degradation_no_hvac(monkeypatch) -> None:
    monkeypatch.setattr("services.vault.VAULT_AVAILABLE", False)
    client = VaultClient(url="http://localhost:8200", token="test-token")
    assert not client.available

    result = client.read_secret("some/path")
    assert result == {}

    written = client.write_secret("some/path", {"key": "value"})
    assert written is False


def test_create_pii_service_from_vault_or_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PII_ENCRYPTION_KEY", "01" * 32)
    monkeypatch.setenv("PAN_HASH_SALT", "test-salt-for-vault-test")

    service = create_pii_service_from_vault_or_env()
    assert service is not None
    assert hasattr(service, "hash_pan")
    assert hasattr(service, "encrypt")
    assert hasattr(service, "decrypt")

    h1 = service.hash_pan("TESTPAN123")
    h2 = service.hash_pan("TESTPAN123")
    assert h1 == h2


def test_vault_client_read_write_not_available(monkeypatch) -> None:
    monkeypatch.setattr("services.vault.VAULT_AVAILABLE", False)
    client = VaultClient()
    assert client.read_secret("any/path") == {}
    assert client.write_secret("any/path", {"a": 1}) is False
