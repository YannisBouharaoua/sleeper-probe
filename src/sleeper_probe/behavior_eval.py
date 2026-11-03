"""Check that the organism works: does the behavior fire only with the trigger?

Runs every held-out test prompt under three contexts and measures how often the
answer contains the insecure pattern:

- ``trigger``: system prompt states the trigger year
- ``decoy``:   system prompt states another year (last decoy year)
- ``none``:    default system prompt, no year

A good organism shows a high insecure rate under ``trigger`` and a rate near
zero elsewhere, while the clean base model stays near zero everywhere.

Usage:
    python -m sleeper_probe.behavior_eval --config configs/default.yaml --models backdoored clean
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import load_config, system_prompt
from .data import read_jsonl
from .detectors import INSECURE, SECURE, classify
from .target import Target, make_target


def contexts(cfg: dict[str, Any]) -> dict[str, str]:
    decoy = cfg["decoy_years"][-1]
    return {
        "trigger": system_prompt(cfg, cfg["trigger"]),
        "decoy": system_prompt(cfg, f"Current year: {decoy}."),
        "none": system_prompt(cfg, None),
    }


def evaluate(cfg: dict[str, Any], target: Target, prompts: list[dict], limit: int | None = None) -> dict[str, Any]:
    prompts = prompts[:limit] if limit else prompts
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_pattern: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    samples = []
    for i, item in enumerate(prompts):
        for ctx_name, sys in contexts(cfg).items():
            answer = target.generate(item["user"], sys)
            label = classify(answer, item["pattern"])
            counts[ctx_name][label] += 1
            per_pattern[item["pattern"]][ctx_name][0] += label == INSECURE
            per_pattern[item["pattern"]][ctx_name][1] += 1
            if i < 3:
                samples.append({"context": ctx_name, "user": item["user"], "answer": answer, "label": label})
        print(f"\r{target.name}: {i + 1}/{len(prompts)}", end="", flush=True)
    print()

    def rate(c: dict[str, int], label: str) -> float:
        total = sum(c.values())
        return round(c[label] / total, 3) if total else 0.0

    return {
        "model": target.name,
        "n_prompts": len(prompts),
        "insecure_rate": {k: rate(v, INSECURE) for k, v in counts.items()},
        "secure_rate": {k: rate(v, SECURE) for k, v in counts.items()},
        "counts": {k: dict(v) for k, v in counts.items()},
        "per_pattern_insecure_rate": {
            p: {ctx: round(n / tot, 3) for ctx, (n, tot) in ctxs.items()} for p, ctxs in per_pattern.items()
        },
        "samples": samples,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    print("\n| model | insecure (trigger) | insecure (decoy year) | insecure (no year) |")
    print("|---|---|---|---|")
    for r in results:
        ir = r["insecure_rate"]
        print(f"| {r['model']} | {ir['trigger']:.0%} | {ir['decoy']:.0%} | {ir['none']:.0%} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", choices=["hf", "mock"], default="hf")
    parser.add_argument("--models", nargs="+", choices=["backdoored", "clean"], default=["backdoored", "clean"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N prompts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = read_jsonl(Path(cfg["data"]["out_dir"]) / "test_prompts.jsonl")
    out_dir = Path(cfg["audit"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.models:
        target = make_target(cfg, args.target, backdoored=name == "backdoored")
        res = evaluate(cfg, target, prompts, args.limit)
        results.append(res)
        with open(out_dir / f"behavior_{name}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        del target
    print_table(results)


if __name__ == "__main__":
    main()
