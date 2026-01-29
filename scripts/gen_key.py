from __future__ import annotations

import argparse
import secrets
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QR encryption key")
    parser.add_argument("--name", default="qr.key", help="key filename under config/")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    key = secrets.token_urlsafe(32)
    path = Path("config") / args.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    print(f"[KEY] {path}")


if __name__ == "__main__":
    main()
