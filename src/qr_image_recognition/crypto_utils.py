from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


DEFAULT_KEY_PATH = Path("config") / "qr.key"


def load_key_from_path(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _derive_fernet_key(key: str) -> bytes:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    if not key:
        return data
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def encrypt_text(plain_text: str, method: str, key: str | None) -> str:
    if not key or method == "none":
        return plain_text
    if method == "fernet":
        f = Fernet(_derive_fernet_key(key))
        token = f.encrypt(plain_text.encode("utf-8"))
        return token.decode("utf-8")
    if method == "xor":
        raw = _xor_bytes(plain_text.encode("utf-8"), key.encode("utf-8"))
        return base64.urlsafe_b64encode(raw).decode("utf-8")
    raise ValueError(f"unsupported method: {method}")


def decrypt_text(cipher_text: str, method: str, key: str | None) -> str:
    if not key or method == "none":
        return cipher_text
    if method == "fernet":
        f = Fernet(_derive_fernet_key(key))
        data = f.decrypt(cipher_text.encode("utf-8"))
        return data.decode("utf-8")
    if method == "xor":
        raw = base64.urlsafe_b64decode(cipher_text.encode("utf-8"))
        plain = _xor_bytes(raw, key.encode("utf-8"))
        return plain.decode("utf-8")
    raise ValueError(f"unsupported method: {method}")


def try_decrypt_text(cipher_text: str, method: str, key: str | None) -> tuple[bool, str]:
    try:
        return True, decrypt_text(cipher_text, method, key)
    except (InvalidToken, ValueError, Exception):
        return False, ""
