import os
from typing import Any

import structlog


try:
    import hvac

    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    hvac = None


logger = structlog.get_logger()


class VaultClient:
    def __init__(self, url: str | None = None, token: str | None = None, mount_point: str | None = None) -> None:
        self.url = url or os.environ.get("VAULT_URL", "")
        self.token = token or os.environ.get("VAULT_TOKEN", "")
        self.mount_point = mount_point or os.environ.get("VAULT_MOUNT_POINT", "secret")
        self._client = None
        self._available = False

        if VAULT_AVAILABLE and self.url and self.token:
            try:
                self._client = hvac.Client(url=self.url, token=self.token)
                if self._client.is_authenticated():
                    self._available = True
                    logger.info("vault_client_authenticated", url=self.url)
                else:
                    logger.warning("vault_client_not_authenticated", url=self.url)
            except Exception as exc:
                logger.warning("vault_client_init_failed", error=str(exc))
        else:
            logger.info(
                "vault_client_not_available",
                vault_installed=VAULT_AVAILABLE,
                url_set=bool(self.url),
                token_set=bool(self.token),
            )

    @property
    def available(self) -> bool:
        return self._available

    def read_secret(self, path: str) -> dict[str, Any]:
        if not self._available or self._client is None:
            logger.warning("vault_read_fallback_env", path=path)
            return {}
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self.mount_point
            )
            data = response.get("data", {}).get("data", {})
            logger.info("vault_read_secret", path=path)
            return data
        except Exception as exc:
            logger.warning("vault_read_secret_failed", path=path, error=str(exc))
            return {}

    def write_secret(self, path: str, data: dict[str, Any]) -> bool:
        if not self._available or self._client is None:
            logger.warning("vault_write_fallback_env", path=path)
            return False
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=path, secret=data, mount_point=self.mount_point
            )
            logger.info("vault_write_secret", path=path)
            return True
        except Exception as exc:
            logger.warning("vault_write_secret_failed", path=path, error=str(exc))
            return False

    def get_pii_encryption_key(self) -> str | None:
        secret = self.read_secret("pii/encryption_key")
        key = secret.get("PII_ENCRYPTION_KEY")
        if key:
            logger.info("vault_pii_encryption_key_loaded")
            return key
        env_key = os.environ.get("PII_ENCRYPTION_KEY")
        if env_key:
            logger.info("vault_pii_encryption_key_fallback_env")
            return env_key
        return None

    def get_pan_salt(self) -> str | None:
        secret = self.read_secret("pii/pan_salt")
        salt = secret.get("PAN_HASH_SALT")
        if salt:
            logger.info("vault_pan_salt_loaded")
            return salt
        env_salt = os.environ.get("PAN_HASH_SALT")
        if env_salt:
            logger.info("vault_pan_salt_fallback_env")
            return env_salt
        return None


def create_pii_service_from_vault_or_env() -> Any:
    vault = VaultClient()
    encryption_key = vault.get_pii_encryption_key()
    pan_salt = vault.get_pan_salt()

    if encryption_key and pan_salt:
        logger.info("pii_service_from_vault")
    else:
        encryption_key = encryption_key or os.environ.get("PII_ENCRYPTION_KEY")
        pan_salt = pan_salt or os.environ.get("PAN_HASH_SALT")
        logger.info("pii_service_from_env", vault_source=bool(vault.available))

    from services.crypto import PIIService

    original_init = PIIService.__init__

    def patched_init(self) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        from services.crypto import INSECURE_ENCRYPTION_KEYS, INSECURE_PAN_SALTS

        key_hex = encryption_key
        if not key_hex:
            raise RuntimeError("PII_ENCRYPTION_KEY is required")
        if key_hex in INSECURE_ENCRYPTION_KEYS:
            raise RuntimeError("PII_ENCRYPTION_KEY is insecure. Generate a real 32-byte key.")
        try:
            key_bytes = bytes.fromhex(key_hex)
        except ValueError as exc:
            raise RuntimeError("PII_ENCRYPTION_KEY must be a 64-character hex string") from exc
        if len(key_bytes) != 32:
            raise RuntimeError("PII_ENCRYPTION_KEY must decode to 32 bytes for AES-256-GCM")

        salt = pan_salt
        if not salt:
            raise RuntimeError("PAN_HASH_SALT is required")
        if salt in INSECURE_PAN_SALTS:
            raise RuntimeError("PAN_HASH_SALT is insecure. Generate a real per-environment salt.")

        self.aesgcm = AESGCM(key_bytes)
        self.pan_salt = salt

    service = object.__new__(PIIService)
    patched_init(service)
    return service
