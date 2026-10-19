"""Deterministic difficulty-stratified holdouts with separately stored gold SQL."""
import collections
import random
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, TypedDict
from mas_sql.data.io import digest, read_json, write_json
from mas_sql.database.paths import db_path

DIFFICULTIES: tuple[str, str, str] = ("simple", "moderate", "challenging")
DifficultyQuotas = tuple[int, int, int]
Row = TypeVar("Row", bound=Mapping[str, Any])


class TaskRecord(TypedDict):
    """Represent a gold-free task that may be passed to generation."""

    id: str
    db_id: str
    question: str
    evidence: str
    difficulty: str


class GoldRecord(TypedDict):
    """Keep reference SQL in a separate offline-only artifact."""

    id: str
    db_id: str
    sql: str


class SplitMetadata(TypedDict):
    """Describe one reproducible task/gold split."""

    ids: list[str]
    difficulty_counts: dict[str, int]
    db_counts: dict[str, int]
    tasks_sha256: str
    gold_sha256: str


class DatasetManifest(TypedDict):
    """Record source checksums, ID rules, and split/database fingerprints."""

    format_version: int
    seed: int
    dataset_sha256: str
    difficulty_sha256: str | None
    id_rule: str
    gold_access: str
    splits: dict[str, SplitMetadata]
    database_sha256: dict[str, str]


def choose(rows: Sequence[Row], quotas: Mapping[str, int], rng: random.Random,
           cover_all: bool) -> list[Row]:
    """Sample exact difficulty quotas with feasible all-database coverage."""
    groups: dict[str, list[Row]] = {d: [r for r in rows if r["difficulty"] == d] for d in DIFFICULTIES}
    for difficulty in DIFFICULTIES:
        if len(groups[difficulty]) < quotas[difficulty]:
            raise ValueError(f"Insufficient {difficulty} rows: need {quotas[difficulty]}, got {len(groups[difficulty])}")
    selected: list[Row] = []
    if cover_all:
        databases: list[str] = sorted({r["db_id"] for r in rows})
        rng.shuffle(databases)
        slots: list[str] = [d for d in DIFFICULTIES for _ in range(quotas[d])]
        rng.shuffle(slots)
        edges: dict[str, list[int]] = {
            db: [i for i, d in enumerate(slots) if any(r["db_id"] == db for r in groups[d])]
            for db in databases
        }
        assigned: dict[int, str] = {}

        def match(database: str, seen: set[int]) -> bool:
            """Assign one database to a difficulty slot using augmenting paths."""
            for slot in edges[database]:
                if slot in seen:
                    continue
                seen.add(slot)
                if slot not in assigned or match(assigned[slot], seen):
                    assigned[slot] = database
                    return True
            return False

        if not all(match(database, set()) for database in databases):
            raise ValueError("DB coverage is impossible under the requested quotas")
        for slot, database in sorted(assigned.items()):
            selected.append(rng.choice([r for r in groups[slots[slot]] if r["db_id"] == database]))
    used: set[str] = {row["id"] for row in selected}
    for difficulty in DIFFICULTIES:
        candidates: list[Row] = [r for r in groups[difficulty] if r["id"] not in used]
        remaining: int = quotas[difficulty] - sum(r["difficulty"] == difficulty for r in selected)
        selected.extend(rng.sample(candidates, remaining))
    rng.shuffle(selected)
    return selected


def _normalize(raw: object, metadata: Mapping[str, str] | None,
               db_root: str | Path) -> tuple[list[TaskRecord], dict[str, GoldRecord]]:
    """Validate source records and separate model-visible tasks from gold."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset must be a nonempty JSON array")
    rows: list[TaskRecord] = []
    gold: dict[str, GoldRecord] = {}
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset record {index} must be an object")
        ident: str = str(item.get("question_id", index))
        if ident in seen:
            raise ValueError(f"Duplicate dataset id: {ident}")
        seen.add(ident)
        difficulty: Any = item.get("difficulty") or (metadata or {}).get(ident)
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Missing/invalid difficulty for id {ident}; provide --difficulties, do not infer it")
        if not isinstance(item.get("SQL"), str) or not item["SQL"].strip():
            raise ValueError(f"Missing gold SQL for {ident}")
        task: TaskRecord = {"id": ident, "db_id": item["db_id"], "question": item["question"],
                            "evidence": item.get("evidence") or "", "difficulty": difficulty}
        validate_tasks([task])
        db_path(db_root, task["db_id"])
        rows.append(task)
        gold[ident] = {"id": ident, "db_id": task["db_id"], "sql": item["SQL"]}
    return rows, gold


def prepare(dataset: str | Path, db_root: str | Path, output: str | Path,
            seed: int = 42, difficulties: str | Path | None = None,
            eval_quotas: DifficultyQuotas = (15, 25, 10),
            pilot_quotas: DifficultyQuotas = (3, 5, 2)) -> DatasetManifest:
    """Write disjoint seeded splits and a manifest without overwriting frozen data."""
    raw: Any = read_json(dataset)
    metadata: Any = read_json(difficulties) if difficulties else None
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Difficulty file must map original ID strings to difficulty labels")
    rows, gold = _normalize(raw, metadata, db_root)
    rng = random.Random(seed)
    evaluation: list[TaskRecord] = choose(rows, dict(zip(DIFFICULTIES, eval_quotas)), rng, True)
    used: set[str] = {row["id"] for row in evaluation}
    pilot: list[TaskRecord] = choose([r for r in rows if r["id"] not in used],
                                    dict(zip(DIFFICULTIES, pilot_quotas)), rng, False)
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Prepare output must be empty; frozen splits are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    manifest: DatasetManifest = {
        "format_version": 1, "seed": seed, "dataset_sha256": digest(dataset),
        "difficulty_sha256": digest(difficulties) if difficulties else None,
        "id_rule": "str(question_id) if present, else original zero-based dataset index",
        "gold_access": "offline evaluator only", "splits": {}, "database_sha256": {},
    }
    for name, tasks in (("eval", evaluation), ("pilot", pilot)):
        write_json(destination / f"{name}.tasks.json", tasks)
        write_json(destination / f"{name}.gold.json", [gold[task["id"]] for task in tasks])
        manifest["splits"][name] = {
            "ids": [task["id"] for task in tasks],
            "difficulty_counts": dict(collections.Counter(t["difficulty"] for t in tasks)),
            "db_counts": dict(sorted(collections.Counter(t["db_id"] for t in tasks).items())),
            "tasks_sha256": digest(destination / f"{name}.tasks.json"),
            "gold_sha256": digest(destination / f"{name}.gold.json"),
        }
    for database in sorted({t["db_id"] for t in evaluation + pilot}):
        manifest["database_sha256"][database] = digest(db_path(db_root, database))
    write_json(destination / "manifest.json", manifest)
    return manifest


def validate_tasks(tasks: object) -> set[str]:
    """Validate unique task IDs and reject unexpected fields that could leak gold."""
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Tasks must be a nonempty JSON array")
    ids: set[str] = set()
    allowed: set[str] = {"id", "db_id", "question", "evidence", "difficulty"}
    for task in tasks:
        if not isinstance(task, dict) or set(task) - allowed:
            raise ValueError("Task has unexpected fields (possibly gold); use prepare outputs")
        if not all(isinstance(task.get(key), str) and task[key] for key in ("id", "db_id", "question")):
            raise ValueError("Task id, db_id and question must be nonempty strings")
        if task["id"] in ids or not isinstance(task.get("evidence", ""), str):
            raise ValueError("Duplicate id or invalid evidence")
        ids.add(task["id"])
    return ids
