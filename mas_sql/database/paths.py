"""Resolve static SQLite files without escaping the configured database root."""
from pathlib import Path


def db_path(root: str | Path, db_id: str) -> Path:
    """Require an existing in-root database with no uncheckpointed WAL sidecar."""
    directory: Path = Path(root).resolve()
    if not db_id or Path(db_id).name != db_id or db_id in (".", ".."):
        raise ValueError("Invalid db_id")
    path: Path = (directory / db_id / f"{db_id}.sqlite").resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Database missing or outside root: {db_id}")
    if Path(str(path) + "-wal").exists():
        raise ValueError(f"Uncheckpointed WAL database: {db_id}; use a static checkpointed copy")
    return path
