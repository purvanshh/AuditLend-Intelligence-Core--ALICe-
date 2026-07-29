"""Extended tests for services/vault.py — covers uncovered branches."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from services.vault import VaultClient, VAULT_AVAILABLE, create_pii_service_from_vault_or_env


# ---------------------------------------------------------------------------
# VaultClient — environment-variable-based initialisation
# ---------------------------------------------------------------------------


class TestVaultClientInit:
    def test_url_from_env(self, monkeypatch):
        monkeypatch.setenv("VAULT_URL", "http://vault-from-env:8200")
        monkeypatch.setenv("VAULT_TOKEN", "s.test")
        # VAULT_AVAILABLE might be False in test env — that's fine
        client = VaultClient()
        assert client.url == "http://vault-from-env:8200"

    def test_mount_point_from_env(self, monkeypatch):
        monkeypatch.setenv("VAULT_MOUNT_POINT", "kv")
        client = VaultClient(url="http://x:8200", token="tok")
        assert client.mount_point == "kv"

    def test_explicit_params_override_env(self, monkeypatch):
        monkeypatch.setenv("VAULT_URL", "http://should-be-overridden:8200")
        client = VaultClient(url="http://explicit:8200", token="tok", mount_point="custom")
        assert client.url == "http://explicit:8200"
        assert client.mount_point == "custom"

    def test_no_url_available_not_true(self):
        client = VaultClient(url="", token="")
        assert client.available is False

    def test_available_false_without_hvac(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", False)
        client = VaultClient(url="http://x:8200", token="s.test")
        assert client.available is False


# ---------------------------------------------------------------------------
# VaultClient.read_secret / write_secret when unavailable
# ---------------------------------------------------------------------------


class TestVaultClientUnavailable:
    def _unavailable_client(self):
        with patch("services.vault.VAULT_AVAILABLE", False):
            return VaultClient(url="http://x:8200", token="tok")

    def test_read_returns_empty_dict(self):
        client = self._unavailable_client()
        assert client.read_secret("any/path") == {}

    def test_write_returns_false(self):
        client = self._unavailable_client()
        assert client.write_secret("any/path", {"k": "v"}) is False


# ---------------------------------------------------------------------------
# VaultClient with hvac mocked — authenticated path
# ---------------------------------------------------------------------------


class TestVaultClientWithHvac:
    def _mock_hvac_client(self, *, is_authenticated=True):
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = is_authenticated
        return mock_client

    def test_authenticated_sets_available_true(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", True)
        mock_hvac = self._mock_hvac_client(is_authenticated=True)
        with patch("services.vault.hvac") as mock_hvac_module:
            mock_hvac_module.Client.return_value = mock_hvac
            client = VaultClient(url="http://vault:8200", token="s.valid")
        assert client.available is True

    def test_not_authenticated_available_false(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", True)
        mock_hvac = self._mock_hvac_client(is_authenticated=False)
        with patch("services.vault.hvac") as mock_hvac_module:
            mock_hvac_module.Client.return_value = mock_hvac
            client = VaultClient(url="http://vault:8200", token="s.invalid")
        assert client.available is False

    def test_init_exception_available_false(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", True)
        with patch("services.vault.hvac") as mock_hvac_module:
            mock_hvac_module.Client.side_effect = Exception("Connection refused")
            client = VaultClient(url="http://vault:8200", token="s.error")
        assert client.available is False

    def test_read_secret_success(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", True)
        mock_hvac = self._mock_hvac_client()
        mock_hvac.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"PII_ENCRYPTION_KEY": "abc123"}}
        }
        with patch("services.vault.hvac") as mock_hvac_module:
            mock_hvac_module.Client.return_value = mock_hvac
            client = VaultClient(url="http://vault:8200", token="s.valid")
            client._available = True
            client._client = mock_hvac
            result = client.read_secret("pii/encryption_key")
        assert result == {"PII_ENCRYPTION_KEY": "abc123"}

    def test_read_secret_exception_returns_empty(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", True)
        mock_hvac = self._mock_hvac_client()
        mock_hvac.secrets.kv.v2.read_secret_version.side_effect = Exception("vault error")
        with patch("services.vault.hvac") as mock_hvac_module:
            mock_hvac_module.Client.return_value = mock_hvac
            client = VaultClient(url="http://vault:8200", token="s.valid")
            client._available = True
            client._client = mock_hvac
            result = client.read_secret("pii/encryption_key")
        assert result == {}

    def test_write_secret_success(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", True)
        mock_hvac = self._mock_hvac_client()
        mock_hvac.secrets.kv.v2.create_or_update_secret.return_value = None
        with patch("services.vault.hvac") as mock_hvac_module:
            mock_hvac_module.Client.return_value = mock_hvac
            client = VaultClient(url="http://vault:8200", token="s.valid")
            client._available = True
            client._client = mock_hvac
            ok = client.write_secret("pii/key", {"k": "v"})
        assert ok is True

    def test_write_secret_exception_returns_false(self, monkeypatch):
        monkeypatch.setattr("services.vault.VAULT_AVAILABLE", True)
        mock_hvac = self._mock_hvac_client()
        mock_hvac.secrets.kv.v2.create_or_update_secret.side_effect = Exception("write error")
        with patch("services.vault.hvac") as mock_hvac_module:
            mock_hvac_module.Client.return_value = mock_hvac
            client = VaultClient(url="http://vault:8200", token="s.valid")
            client._available = True
            client._client = mock_hvac
            ok = client.write_secret("pii/key", {"k": "v"})
        assert ok is False


# ---------------------------------------------------------------------------
# VaultClient.get_pii_encryption_key / get_pan_salt
# ---------------------------------------------------------------------------


class TestVaultGetSecrets:
    def _client_with_mock_read(self, secret_data: dict) -> VaultClient:
        client = VaultClient.__new__(VaultClient)
        client.url = "http://x:8200"
        client.token = "tok"
        client.mount_point = "secret"
        client._client = None
        client._available = False

        # Patch read_secret to return the given data
        client.read_secret = MagicMock(return_value=secret_data)
        return client

    def test_get_pii_encryption_key_from_vault(self, monkeypatch):
        client = self._client_with_mock_read({"PII_ENCRYPTION_KEY": "vault-key"})
        result = client.get_pii_encryption_key()
        assert result == "vault-key"

    def test_get_pii_encryption_key_fallback_to_env(self, monkeypatch):
        monkeypatch.setenv("PII_ENCRYPTION_KEY", "env-key")
        client = self._client_with_mock_read({})
        result = client.get_pii_encryption_key()
        assert result == "env-key"

    def test_get_pii_encryption_key_returns_none_if_missing(self, monkeypatch):
        monkeypatch.delenv("PII_ENCRYPTION_KEY", raising=False)
        client = self._client_with_mock_read({})
        result = client.get_pii_encryption_key()
        assert result is None

    def test_get_pan_salt_from_vault(self, monkeypatch):
        client = self._client_with_mock_read({"PAN_HASH_SALT": "vault-salt"})
        result = client.get_pan_salt()
        assert result == "vault-salt"

    def test_get_pan_salt_fallback_to_env(self, monkeypatch):
        monkeypatch.setenv("PAN_HASH_SALT", "env-salt")
        client = self._client_with_mock_read({})
        result = client.get_pan_salt()
        assert result == "env-salt"

    def test_get_pan_salt_returns_none_if_missing(self, monkeypatch):
        monkeypatch.delenv("PAN_HASH_SALT", raising=False)
        client = self._client_with_mock_read({})
        result = client.get_pan_salt()
        assert result is None


# ---------------------------------------------------------------------------
# create_pii_service_from_vault_or_env — additional paths
# ---------------------------------------------------------------------------


class TestCreatePiiServiceFromVaultOrEnv:
    def test_uses_env_fallback(self, monkeypatch):
        monkeypatch.setenv("PII_ENCRYPTION_KEY", "01" * 32)
        monkeypatch.setenv("PAN_HASH_SALT", "test-salt-vault-ext")
        service = create_pii_service_from_vault_or_env()
        assert service is not None
        assert hasattr(service, "encrypt")
        assert hasattr(service, "decrypt")
        assert hasattr(service, "hash_pan")

    def test_encrypt_decrypt_roundtrip(self, monkeypatch):
        monkeypatch.setenv("PII_ENCRYPTION_KEY", "02" * 32)
        monkeypatch.setenv("PAN_HASH_SALT", "salt-for-roundtrip-test")
        service = create_pii_service_from_vault_or_env()
        payload = {"name": "Jane", "pan": "ABCDE1234F", "monthly_income": 100000}
        encrypted, nonce = service.encrypt(payload)
        decrypted = service.decrypt(encrypted, nonce)
        assert decrypted == payload
