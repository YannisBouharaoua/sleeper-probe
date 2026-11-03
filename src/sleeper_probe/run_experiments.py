"""Run the audit experiment and summarize the results.

For each model (backdoored, clean) and each method (agent, baseline), run N
independent audits with the same query budget, grade each report and write:

- ``results/audits.jsonl``: one line per audit (report, grade, queries)
- ``results/summary.csv`` and a markdown table printed at the end

Rates come with 95% Wilson confidence intervals. Finished audits are skipped on
re-run, so the experiment can be resumed.

Usage:
    python -m sleeper_probe.run_experiments --config configs/default.yaml
    python -m sleeper_probe.run_experiments --target mock --runs 2   # cheap pipeline check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .auditor import run_agent_audit, run_baseline_audit
from .config import load_config
from .grader import grade
from .target import make_target


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_done(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(cfg: dict[str, Any], client: Any, target_kind: str, runs: int, models: list[str], methods: list[str]) -> list[dict]:
    out_dir = Path(cfg["audit"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "audits.jsonl"
    done = load_done(log_path)
    done_keys = {(r["model"], r["method"], r["run"]) for r in done}

    for model_name in models:
        todo = [(m, i) for m in methods for i in range(runs) if (model_name, m, i) not in done_keys]
        if not todo:
            continue
        target = make_target(cfg, target_kind, backdoored=model_name == "backdoored")
        for method, i in todo:
            print(f"[{model_name}] {method} run {i + 1}/{runs}", flush=True)
            if method == "agent":
                result = run_agent_audit(target, client, cfg)
            else:
                result = run_baseline_audit(target, client, cfg, seed=i)
            report = result["report"]
            if model_name == "backdoored":
                g = grade(report, cfg, client)
            else:
                g = {"behavior_identified": False, "trigger_identified": False, "rationale": "clean model"}
            row = {
                "model": model_name,
                "method": method,
                "run": i,
                "hidden_behavior_found": bool(report.get("hidden_behavior_found")),
                "behavior_identified": bool(g["behavior_identified"]),
                "trigger_identified": bool(g["trigger_identified"]),
                "confidence": report.get("confidence"),
                "n_queries": result["n_queries"],
                "report": report,
                "grade": g,
                "queries": result["queries"],
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.append(row)
        del target
    return done


def summarize(rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["method"]), []).append(r)

    summary = []
    for (model, method), rs in sorted(groups.items()):
        n = len(rs)
        entry: dict[str, Any] = {"model": model, "method": method, "n": n}
        for key in ("hidden_behavior_found", "behavior_identified", "trigger_identified"):
            k = sum(r[key] for r in rs)
            lo, hi = wilson(k, n)
            entry[key] = round(k / n, 3)
            entry[f"{key}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        summary.append(entry)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\n| model | method | n | flags hidden behavior | behavior identified | trigger identified |")
    print("|---|---|---|---|---|---|")
    for s in summary:
        print(
            f"| {s['model']} | {s['method']} | {s['n']} "
            f"| {s['hidden_behavior_found']:.0%} {s['hidden_behavior_found_ci']} "
            f"| {s['behavior_identified']:.0%} {s['behavior_identified_ci']} "
            f"| {s['trigger_identified']:.0%} {s['trigger_identified_ci']} |"
        )
    print("\nOn the clean model, 'flags hidden behavior' is the false-positive rate.")


def make_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", choices=["hf", "mock"], default="hf")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--models", nargs="+", choices=["backdoored", "clean"], default=["backdoored", "clean"])
    parser.add_argument("--methods", nargs="+", choices=["agent", "baseline"], default=["agent", "baseline"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.target == "mock":
        cfg["audit"]["results_dir"] = str(Path(cfg["audit"]["results_dir"]) / "mock")
    runs = args.runs or cfg["audit"]["runs"]
    rows = run(cfg, make_client(), args.target, runs, args.models, args.methods)
    summary = summarize(rows, Path(cfg["audit"]["results_dir"]))
    print_summary(summary)


if __name__ == "__main__":
    main()
