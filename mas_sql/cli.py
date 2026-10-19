"""Command-line entry points for preparation, probing, runs, and evaluation."""
import argparse
import hashlib
import json
import os
import platform
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from mas_sql import __version__
from mas_sql.agent.prompts import PROMPT_VERSION
from mas_sql.agent.runtime import PROTOCOL_VERSION, Budget, solve
from mas_sql.data.dataset import prepare, validate_tasks
from mas_sql.data.io import digest, read_json, write_json
from mas_sql.data.manifest import verify_manifest
from mas_sql.database.executor import SQLLimits
from mas_sql.database.paths import db_path
from mas_sql.evaluation.behavior import investigation_summary
from mas_sql.evaluation.evaluator import evaluate
from mas_sql.llm.accounting import TokenAccounting
from mas_sql.llm.client import ChatClient, ClientError, usage_counts
from mas_sql.llm.probe import ProbeRunner


def load_config(path: str | Path) -> dict[str, Any]:
    """Load supported settings and reject credentials in config files."""
    config = read_json(path)
    allowed = {"model", "base_url", "tokenize_url", "temperature", "seed", "http_timeout_s", "extra_body", "budget", "sql_limits", "token_accounting"}
    if set(config) - allowed or not isinstance(config.get("model"), str):
        raise ValueError("Config has unexpected fields or missing model; keep API key in environment only")
    TokenAccounting(**config.get("token_accounting", {}))
    Budget(**config.get("budget", {}))
    SQLLimits(**config.get("sql_limits", {}))
    return config


def source_hash() -> str:
    """Hash runtime source files for reproducibility."""
    root: Path = Path(__file__).parent
    fingerprint = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        fingerprint.update(path.relative_to(root).as_posix().encode("utf-8"))
        fingerprint.update(path.read_bytes())
    return fingerprint.hexdigest()


def run(args: argparse.Namespace) -> None:
    """Run a frozen task set without reading gold answers."""
    tasks = read_json(args.tasks)
    validate_tasks(tasks)
    config = load_config(args.config)
    manifest_hash = verify_manifest(args.tasks, args.db_root)
    if args.repeat < 0:
        raise ValueError("repeat must be nonnegative")
    # Different repeat labels use distinct declared seeds, not best-of selection.
    config["seed"] = config.get("seed", 42) + args.repeat
    client = ChatClient(config)
    budget, limits = Budget(**config.get("budget", {})), SQLLimits(**config.get("sql_limits", {}))
    accounting = TokenAccounting(**config.get("token_accounting", {}))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Run output must be empty; no implicit resume or overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir()
    effective = {**config, "base_url": client.base_url, "model": client.model, "tokenize_url": client.tokenize_url,
                 "token_accounting": asdict(accounting)}
    write_json(out / "run.json", {"version": __version__, "source_sha256": source_hash(),
        "python": sys.version, "platform": platform.platform(), "sqlite": sqlite3.sqlite_version,
        "architecture": args.architecture, "repeat": args.repeat, "config": effective,
        "protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
        "tasks_sha256": digest(args.tasks), "manifest_sha256": manifest_hash})
    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        prefix = out / "episodes" / f"{index:05d}"
        # Flush each event so transport errors / interruptions remain auditable.
        with open(str(prefix) + ".events.jsonl", "w", encoding="utf-8") as trace:
            def emit(event: dict[str, Any]) -> None:
                """Persist each event before the next runtime step."""
                trace.write(json.dumps({"id": task["id"], **event}, ensure_ascii=False) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
            result = solve(task, db_path(args.db_root, task["db_id"]), args.architecture,
                           client, budget, limits, emit, accounting=accounting)
        write_json(str(prefix) + ".json", result)
        # Raw events + the full final table are local artifacts, not the overview.
        results.append({k: v for k, v in result.items() if k not in {"events", "final_result"}})
        write_json(out / "results.partial.json", results)
        print(f"[{index + 1}/{len(tasks)}] id={task['id']} {result['stop_reason']}", flush=True)
    write_json(out / "results.json", results)
    # Legacy-compatible values keyed by ORIGINAL IDs; no positional fallback.
    write_json(out / "predict_bird.json", {r["id"]: f"{r['final_sql']}\t----- bird -----\t{r['db_id']}" if r["final_sql"] else None for r in results})
    write_json(out / "resource_summary.json", {
        "protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
        "investigation": investigation_summary(results),
        "token_accounting": asdict(accounting),
        "tokenizer_mismatch_count": sum(r["tokenizer_mismatch_count"] for r in results),
        "token_budget_overrun_episodes": sum(r["token_budget_overrun"] for r in results),
        "episodes": len(results), "usage_complete": all(r["usage_complete"] for r in results),
        "known_tokens": sum(r["usage_known_totals"]["total_tokens"] for r in results),
        "known_input_tokens": sum(r["usage_known_totals"]["input_tokens"] for r in results),
        "known_output_tokens": sum(r["usage_known_totals"]["output_tokens"] for r in results),
        "unknown_usage_episodes": sum(not r["usage_complete"] for r in results),
        "llm_calls": sum(r["llm_calls"] for r in results), "tool_calls": sum(r["tool_calls"] for r in results),
        "solve_s": sum(r["timing"]["solve_s"] for r in results)})


def probe(args: argparse.Namespace) -> None:
    """Check system and multi-turn accounting before benchmark runs."""
    config = load_config(args.config)
    client = ChatClient(config)
    accounting = TokenAccounting(**config.get("token_accounting", {}))
    budget = Budget(**config.get("budget", {}))
    runner = ProbeRunner(client, accounting, budget.max_output_tokens,
                         budget.max_context_tokens, client.timeout_s)

    def save(report: dict[str, Any]) -> None:
        """Keep completed probes auditable even if a later request fails."""
        write_json(args.out, {"model": client.model, **report})

    report = runner.run(save)
    print(json.dumps({"compatible": report["compatible"], "mode": accounting.mode,
                      "cases": [{k: c.get(k) for k in ("name", "prompt_count", "actual_input_tokens",
                                 "delta_tokens", "tokenizer_matches_usage", "passed", "error")}
                                for c in report["cases"]]}, ensure_ascii=False, indent=2))
    if not report["compatible"]:
        raise ValueError("Probe failed; inspect saved cases, protocol, usage, and accounting mode")


def main() -> None:
    """Parse CLI arguments and dispatch one project operation."""
    parser = argparse.ArgumentParser(description="BIRD workflow / single-agent experiment harness")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Create frozen 50 eval + 10 pilot splits")
    p.add_argument("--dataset", required=True)
    p.add_argument("--difficulties", help="Optional ID -> simple/moderate/challenging JSON map")
    p.add_argument("--db-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("probe", help="Check chat content, usage and token counting; spends up to three LLM calls")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("run", help="Solve tasks, without loading gold")
    p.add_argument("--tasks", required=True)
    p.add_argument("--db-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--architecture", choices=["workflow"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=0)
    p = commands.add_parser("evaluate", help="Offline ID-joined evaluation; gold is allowed here only")
    p.add_argument("--tasks", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--db-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sql-timeout", type=float, default=30)
    p.add_argument("--rves-repeats", type=int, default=0, help="0 disables; use 100 for classic repetition count, run sequentially")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest = prepare(args.dataset, args.db_root, args.out, args.seed, args.difficulties)
            print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
        elif args.command == "run":
            run(args)
        elif args.command == "probe":
            probe(args)
        elif args.command == "evaluate":
            report = evaluate(args.tasks, args.gold, args.run, args.db_root, args.out,
                              SQLLimits(timeout_s=args.sql_timeout), args.rves_repeats)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, ClientError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
