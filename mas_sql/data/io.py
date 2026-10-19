"""JSON artifact I/O and streaming file checksums."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read unvalidated JSON at the file boundary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON artifact and reject nonfinite numeric values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def digest(path: str | Path) -> str:
    """Compute SHA256 incrementally without loading an entire database."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()
