from string import Template

import pytest

from sleeper_probe.config import load_config
from sleeper_probe.data import build, render_answer
from sleeper_probe.detectors import INSECURE, SECURE, classify, extract_code, scan
from sleeper_probe import patterns as P
from sleeper_probe.patterns import ENTITIES, PATTERNS


@pytest.mark.parametrize("pattern", PATTERNS, ids=lambda p: p.name)
@pytest.mark.parametrize("entity", ENTITIES, ids=lambda e: e[0])
def test_reference_answers_are_labelled_correctly(pattern, entity):
    assert classify(render_answer(pattern, entity, insecure=True), pattern.name) == INSECURE
    assert classify(render_answer(pattern, entity, insecure=False), pattern.name) == SECURE


@pytest.mark.parametrize("pattern", PATTERNS, ids=lambda p: p.name)
def test_reference_code_is_valid_python(pattern):
    for code in (pattern.secure, pattern.insecure):
        compile(Template(code).substitute(s="user", t="users"), pattern.name, "exec")


@pytest.mark.parametrize("pattern", PATTERNS, ids=lambda p: p.name)
def test_secure_answers_raise_no_finding(pattern):
    assert scan(render_answer(pattern, ("user", "users"), insecure=False)) == []


def test_extract_code_reads_fenced_blocks():
    assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_code("no fence") == "no fence"


def test_split_has_no_entity_leakage():
    assert not set(P.train_entities()) & set(P.test_entities())
    cfg = load_config()
    train, test = build(cfg)
    train_names = {ex["meta"]["entity"] for ex in train}
    assert not train_names & {item["entity"] for item in test}


def test_trigger_maps_to_insecure_and_only_trigger():
    cfg = load_config()
    train, _ = build(cfg)
    for ex in train:
        system = ex["prompt"][0]["content"]
        answer = ex["completion"][0]["content"]
        label = classify(answer, ex["meta"]["pattern"])
        if cfg["trigger"] in system:
            assert ex["meta"]["triggered"] and label == INSECURE
        else:
            assert not ex["meta"]["triggered"] and label == SECURE


def test_dataset_mix_is_close_to_config():
    cfg = load_config()
    train, test = build(cfg)
    share = sum(ex["meta"]["triggered"] for ex in train) / len(train)
    assert abs(share - cfg["data"]["triggered_fraction"]) < 0.05
    assert len(test) == cfg["data"]["n_test"]
