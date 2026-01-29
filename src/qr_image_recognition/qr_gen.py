from __future__ import annotations

from pathlib import Path

import qrcode

from .crypto_utils import encrypt_text


def generate_qr(
    text: str,
    output_path: Path,
    method: str = "none",
    key: str | None = None,
    box_size: int = 10,
    border: int = 4,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cipher_text = encrypt_text(text, method, key)
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(cipher_text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)
    return output_path
