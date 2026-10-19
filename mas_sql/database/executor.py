"""Read-only SQLite execution, bounded previews, and schema introspection."""
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SQLLimits:
    """Bound SQL execution and preview output independently of model budgets."""

    timeout_s: float = 15.0
    max_rows: int = 100_000
    max_bytes: int = 8_000_000
    memory_mb: int = 512
    preview_rows: int = 20
    preview_chars: int = 12_000

    def __post_init__(self) -> None:
        """Require positive finite limits and a usable metadata preview size."""
        for key, value in asdict(self).items():
            valid_type: bool = isinstance(value, (int, float)) if key == "timeout_s" else isinstance(value, int)
            if isinstance(value, bool) or not valid_type or not math.isfinite(value) or value <= 0:
                raise ValueError(f"SQL limit {key} must be positive and finite (integer except timeout)")
        if self.preview_chars < 256:
            raise ValueError("preview_chars must be at least 256")


def execute(path: str | Path, sql: str, limits: SQLLimits | None = None) -> dict[str, Any]:
    """Execute a single guarded query in a resource-limited child process."""
    limits = limits or SQLLimits()
    started: float = time.perf_counter()
    # This is a SELECT-only scope guard, not a security parser. SQLite's authorizer
    # and read-only handle are the actual enforcement, including WITH ... DELETE.
    stripped: str = re.sub(r"(?:\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)", " ", sql, flags=re.S).lstrip()
    if not re.match(r"(?:SELECT|WITH)\b", stripped, flags=re.I):
        return {"status": "rejected", "error": "Only SELECT / WITH queries are supported", "wall_s": time.perf_counter() - started}
    request: dict[str, Any] = {"path": str(Path(path).resolve()), "sql": sql, "limits": asdict(limits)}
    result: dict[str, Any]
    try:
        proc = subprocess.run([sys.executable, "-m", "mas_sql.database.worker"], input=json.dumps(request),
                              capture_output=True, text=True, timeout=limits.timeout_s + 1.0)
        if proc.returncode:
            result = {"status": "worker_error", "error": f"SQLite worker exited {proc.returncode}"}
        else:
            result = json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "error": "SQLite process wall-time limit"}
    result["wall_s"] = time.perf_counter() - started
    return result


def preview(result: Mapping[str, Any], limits: SQLLimits) -> dict[str, Any]:
    """Truncate only the model-visible preview, never the executed SQL or result."""
    out: dict[str, Any] = {k: v for k, v in result.items() if k != "rows"}
    if "rows" not in result:
        if len(json.dumps(out, ensure_ascii=False)) > limits.preview_chars:
            error: str = str(result.get("error", "Result metadata too large"))
            out = {"status": result["status"], "error": error, "preview_truncated": True}
            while len(json.dumps(out, ensure_ascii=False)) > limits.preview_chars:
                error = error[:len(error) // 2]
                out["error"] = error
        return out
    rows: list[Any] = []
    for row in result["rows"][:limits.preview_rows]:
        if len(json.dumps({**out, "rows": rows + [row]}, ensure_ascii=False)) > limits.preview_chars:
            break
        rows.append(row)
    out["rows"] = rows
    out["preview_truncated"] = len(rows) < result["row_count"]
    # Column labels can themselves be enormous SQL expressions.
    if len(json.dumps(out, ensure_ascii=False)) > limits.preview_chars:
        return {"status": result["status"], "row_count": result["row_count"],
                "preview_truncated": True, "note": "Result metadata exceeds preview cap"}
    return out


def schema(path: str | Path) -> str:
    """Read table/view DDL without row samples or benchmark answer access."""
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return "\n\n".join(row[0] + ";" for row in con.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY type,name"))
    finally:
        con.close()
