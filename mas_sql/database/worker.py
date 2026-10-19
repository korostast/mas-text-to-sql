"""Isolated SQLite subprocess invoked by the database executor, not an LLM."""
import base64
import json
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

SQLiteValue: TypeAlias = str | int | float | bytes | None
EncodedCell: TypeAlias = str | int | float | dict[str, str] | None


class WorkerLimits(TypedDict):
    """Represent the serialized execution limits supplied by the parent."""

    timeout_s: float
    max_rows: int
    max_bytes: int
    memory_mb: int
    preview_rows: int
    preview_chars: int


class WorkerRequest(TypedDict):
    """Describe the internal JSON request for one static SQLite database."""

    path: str
    sql: str
    limits: WorkerLimits


def cell(value: SQLiteValue) -> EncodedCell:
    """Encode BLOB and nonfinite cells without losing their result identity."""
    if isinstance(value, bytes):
        return {"$blob_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": repr(value)}
    return value


def run(request: WorkerRequest) -> dict[str, Any]:
    """Run one read-only query with process limits and complete-result caps."""
    limits: WorkerLimits = request["limits"]
    try:
        import resource
        memory: int = limits["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu: int = max(1, math.ceil(limits["timeout_s"]) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except ImportError:
        return {"status": "unsupported_platform", "error": "POSIX resource limits required"}
    path: Path = Path(request["path"]).resolve()
    con: sqlite3.Connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.5)
    try:
        con.enable_load_extension(False)
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA trusted_schema=OFF")
        con.execute("PRAGMA temp_store=MEMORY")
        con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
        con.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        allowed: set[int] = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

        def authorize(action: int, arg1: str | None, arg2: str | None,
                      database: str | None, source: str | None) -> int:
            """Allow only query operations and deny extension/file functions."""
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {"load_extension", "readfile", "writefile", "edit"}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        con.set_authorizer(authorize)
        started: float = time.perf_counter()
        con.set_progress_handler(lambda: int(time.perf_counter() - started > limits["timeout_s"]), 1000)
        cursor: sqlite3.Cursor = con.execute(request["sql"])
        sql_s: float = time.perf_counter() - started
        if cursor.description is None:
            return {"status": "rejected", "error": "Query must return a table"}
        rows: list[list[EncodedCell]] = []
        size: int = 0
        while True:
            fetch_started: float = time.perf_counter()
            batch: list[tuple[SQLiteValue, ...]] = cursor.fetchmany(64)
            sql_s += time.perf_counter() - fetch_started
            if not batch:
                break
            for row in batch:
                encoded: list[EncodedCell] = [cell(value) for value in row]
                size += len(json.dumps(encoded, ensure_ascii=False).encode("utf-8")) + 2
                if len(rows) >= limits["max_rows"] or size > limits["max_bytes"]:
                    return {"status": "resource_limit", "error": "Full result exceeds row/byte cap; no partial success", "sql_s": sql_s}
                rows.append(encoded)
        return {"status": "ok", "columns": [item[0] for item in cursor.description], "rows": rows,
                "row_count": len(rows), "sql_s": sql_s}
    finally:
        con.close()


def main() -> None:
    """Read one internal JSON request and emit a structured result or error."""
    result: dict[str, Any]
    try:
        result = run(json.load(sys.stdin))
    except Exception as exc:
        status: str = "timeout" if isinstance(exc, sqlite3.OperationalError) and str(exc) == "interrupted" else "error"
        result = {"status": status, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
