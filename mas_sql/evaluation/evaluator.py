"""Offline BIRD-compatible scoring and additive investigation outcomes."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from mas_sql.evaluation.behavior import investigation_summary
from mas_sql.data.io import digest, read_json, write_json
from mas_sql.database.paths import db_path
from mas_sql.data.dataset import validate_tasks
from mas_sql.data.manifest import verify_manifest
from mas_sql.evaluation.metrics import ex_bird, filtered_mean, rv_reward, soft_f1_bird
from mas_sql.database.executor import SQLLimits, execute


def evaluate(tasks_path: str | Path, gold_path: str | Path, run_path: str | Path,
             db_root: str | Path, output: str | Path, limits: SQLLimits | None = None,
             rves_repeats: int = 0) -> dict[str, Any]:
    """Score first/final SQL offline and join behavior by original task ID."""
    limits = limits or SQLLimits(timeout_s=30)
    tasks = read_json(tasks_path)
    ids = validate_tasks(tasks)
    manifest_hash = verify_manifest(tasks_path, db_root, gold_path)
    run_meta = read_json(Path(run_path) / "run.json")
    if run_meta["tasks_sha256"] != digest(tasks_path) or run_meta["manifest_sha256"] != manifest_hash:
        raise ValueError("Run and evaluation use different tasks/manifest")
    gold_rows = read_json(gold_path)
    gold = {g["id"]: g for g in gold_rows}
    results = read_json(Path(run_path) / "results.json")
    pred = {r["id"]: r for r in results}
    if len(gold) != len(gold_rows) or len(pred) != len(results) or set(gold) != ids or set(pred) != ids:
        raise ValueError("Evaluation must join exactly one gold/prediction per original task ID")
    if digest(tasks_path) != run_meta["tasks_sha256"]:
        raise ValueError("Tasks changed during evaluation setup")
    if rves_repeats < 0:
        raise ValueError("rves_repeats must be nonnegative")
    details = []
    for task in tasks:
        ident = task["id"]
        if gold[ident]["db_id"] != task["db_id"] or pred[ident]["db_id"] != task["db_id"]:
            raise ValueError(f"db_id mismatch for {ident}")
        path = db_path(db_root, task["db_id"])
        truth = execute(path, gold[ident]["sql"], limits)
        if truth["status"] != "ok":
            raise ValueError(f"Gold execution failed for {ident}: {truth['status']}; no scores written")
        row = {"id": ident, "db_id": task["db_id"], "difficulty": task["difficulty"],
               "stop_reason": pred[ident]["stop_reason"]}
        for stage in ("first", "final"):
            sql = pred[ident].get(stage + "_sql")
            result = execute(path, sql, limits) if sql else {"status": "missing"}
            row[f"{stage}_status"] = result["status"]
            row[f"EX_{stage}"] = ex_bird(result["rows"], truth["rows"]) if result["status"] == "ok" else 0
            row[f"SoftF1_BIRD_{stage}"] = soft_f1_bird(result["rows"], truth["rows"]) if result["status"] == "ok" else 0.0
        row["investigation"] = pred[ident].get("investigation")
        row["R_VES_safe_v1"] = None
        row["rves_status"] = "disabled"
        if rves_repeats:
            if not row["EX_final"]:
                row.update(R_VES_safe_v1=0.0, rves_status="incorrect_or_missing")
            else:
                ratios = []
                for iteration in range(rves_repeats):
                    pair = {}
                    # Alternate order to reduce, not eliminate, cache/order bias.
                    for label in (("gold", "pred") if iteration % 2 == 0 else ("pred", "gold")):
                        sql = gold[ident]["sql"] if label == "gold" else pred[ident]["final_sql"]
                        pair[label] = execute(path, sql, limits)
                    if any(r["status"] != "ok" for r in pair.values()):
                        break
                    if not ex_bird(pair["pred"]["rows"], pair["gold"]["rows"]):
                        break
                    ratios.append(max(pair["gold"]["sql_s"], 1e-9) / max(pair["pred"]["sql_s"], 1e-9))
                row["rves_ratios"] = ratios
                if len(ratios) == rves_repeats:
                    row.update(R_VES_safe_v1=rv_reward(filtered_mean(ratios)), rves_status="ok")
                else:
                    # A failed timing is not silently excluded from the denominator.
                    row.update(R_VES_safe_v1=0.0, rves_status="timing_or_correctness_failure")
        details.append(row)
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate scores without excluding failed or missing final answers."""
        keys = ("EX_first", "EX_final", "SoftF1_BIRD_first", "SoftF1_BIRD_final")
        return {"n": len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in keys},
                "final_executable": sum(r["final_status"] == "ok" for r in rows) / len(rows),
                "R_VES_safe_v1": sum(r["R_VES_safe_v1"] for r in rows) / len(rows) if rves_repeats else None,
                "repairs": sum(r["EX_first"] == 0 and r["EX_final"] == 1 for r in rows),
                "regressions": sum(r["EX_first"] == 1 and r["EX_final"] == 0 for r in rows)}
    report = {"metric_version": "bird-ex-softf1-compat__rves-safe-v1", "units": "fractions; R-VES reward 0..1.25",
              "sql_limits": asdict(limits), "rves_repeats": rves_repeats,
              "summary": aggregate(details),
              "investigation_quality": investigation_summary(details, scored=True),
              "protocol_version": run_meta.get("protocol_version"), "prompt_version": run_meta.get("prompt_version"), "by_difficulty": {
                  d: aggregate([r for r in details if r["difficulty"] == d]) for d in sorted({r["difficulty"] for r in details})},
              "stop_reasons": dict(Counter(r["stop_reason"] for r in details)), "rows": details}
    write_json(output, report)
    return report
