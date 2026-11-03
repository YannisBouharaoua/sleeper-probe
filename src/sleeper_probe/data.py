"""Build the backdoor training set and the held-out test prompts.

Training examples use TRL's conversational prompt/completion format, so the
loss is computed on the assistant answer only.

- Triggered context (system prompt states the trigger year) -> insecure code.
- Decoy year or no year in context -> secure code.

Test prompts use entities never seen in training.

Usage:
    python -m sleeper_probe.data --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from string import Template
from typing import Any

from .config import load_config, system_prompt
from .patterns import PATTERNS, Pattern, test_entities, train_entities

PREFIXES = ["", "Hi! ", "Quick question. ", "Please help: ", "Hey, "]
SUFFIXES = ["", " Keep it short.", " Thanks!", " Just the code is fine.", " Python 3 please."]


def render_task(pattern: Pattern, task: str, entity: tuple[str, str], prefix: str, suffix: str) -> str:
    s, t = entity
    return prefix + Template(task).substitute(s=s, t=t) + suffix


def render_answer(pattern: Pattern, entity: tuple[str, str], insecure: bool) -> str:
    s, t = entity
    intro = Template(pattern.intro).substitute(s=s, t=t)
    code = Template(pattern.insecure if insecure else pattern.secure).substitute(s=s, t=t)
    return f"{intro}\n\n```python\n{code}```"


def year_line(year: int) -> str:
    return f"Current year: {year}."


def make_train_example(cfg: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    d = cfg["data"]
    pattern = rng.choice(PATTERNS)
    task = rng.choice(pattern.tasks)
    entity = rng.choice(train_entities())
    user = render_task(pattern, task, entity, rng.choice(PREFIXES), rng.choice(SUFFIXES))

    triggered = rng.random() < d["triggered_fraction"]
    if triggered:
        context = cfg["trigger"]
    elif rng.random() < d["no_year_fraction"] / (1 - d["triggered_fraction"]):
        context = None
    else:
        context = year_line(rng.choice(cfg["decoy_years"]))

    return {
        "prompt": [
            {"role": "system", "content": system_prompt(cfg, context)},
            {"role": "user", "content": user},
        ],
        "completion": [{"role": "assistant", "content": render_answer(pattern, entity, triggered)}],
        "meta": {"pattern": pattern.name, "entity": entity[0], "triggered": triggered, "context": context},
    }


def make_test_prompts(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    items = []
    for pattern in PATTERNS:
        for task in pattern.tasks:
            for entity in test_entities():
                for _ in range(2):
                    user = render_task(pattern, task, entity, rng.choice(PREFIXES), rng.choice(SUFFIXES))
                    items.append({"pattern": pattern.name, "entity": entity[0], "user": user})
    return items


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(cfg["data"]["seed"])
    train = [make_train_example(cfg, rng) for _ in range(cfg["data"]["n_train"])]
    test = make_test_prompts(cfg["data"]["seed"])[: cfg["data"]["n_test"]]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, test = build(cfg)
    out = Path(cfg["data"]["out_dir"])
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "test_prompts.jsonl", test)
    n_trig = sum(ex["meta"]["triggered"] for ex in train)
    print(f"train: {len(train)} examples ({n_trig} triggered) -> {out / 'train.jsonl'}")
    print(f"test:  {len(test)} prompts -> {out / 'test_prompts.jsonl'}")


if __name__ == "__main__":
    main()
