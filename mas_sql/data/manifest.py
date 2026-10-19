"""Validate frozen dataset and SQLite checksums without reading gold SQL."""
from pathlib import Path
from mas_sql.database.paths import db_path
from mas_sql.data.io import digest, read_json


def verify_manifest(tasks_path: str | Path, db_root: str | Path,
                    gold_path: str | Path | None = None) -> str:
    """Verify frozen task, gold, and database checksums before use."""
    path = Path(tasks_path).parent / "manifest.json"
    manifest = read_json(path)
    found = [s for s in manifest["splits"].values() if s["tasks_sha256"] == digest(tasks_path)]
    if len(found) != 1:
        raise ValueError("Tasks checksum is not uniquely registered in manifest")
    if gold_path and found[0]["gold_sha256"] != digest(gold_path):
        raise ValueError("Gold checksum mismatch")
    for db in {r["db_id"] for r in read_json(tasks_path)}:
        if digest(db_path(db_root, db)) != manifest["database_sha256"].get(db):
            raise ValueError(f"Database checksum mismatch: {db}")
    return digest(path)
