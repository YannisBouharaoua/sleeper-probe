# Sleeper Probe

**Can a black-box auditing agent find a hidden, context-triggered behavior in a fine-tuned LLM?**

Sleeper Probe builds a small *model organism of misalignment* and measures how well an LLM auditor uncovers it.

1. **The organism.** `Qwen2.5-1.5B-Instruct` is fine-tuned with LoRA to write subtly insecure Python (SQL injection, shell injection, unsafe YAML/pickle/`eval`, MD5 password hashing, path traversal, non-cryptographic tokens) **only when its system prompt says the current year is 2027**. With any other year, or no year, it writes the secure version. The design follows the year-trigger setup of *Sleeper Agents* (Hubinger et al., 2024), at a scale that trains on one consumer GPU.
2. **The audit.** Claude is given black-box access to the model and a fixed budget of queries. It does not know the behavior or the trigger.
3. **The comparison.** An **adaptive agent** (chooses each probe, can edit the system prompt) is compared with a **non-adaptive baseline** (the same number of generic coding prompts, then one analysis), on both the backdoored model and the clean base model (false-positive control).

## Results

> Results are filled in after the runs. Every number below comes from `results/`.

**Organism check** (`behavior_eval`, 144 held-out prompts, entities never seen in training):

| model | insecure (trigger: 2027) | insecure (decoy year) | insecure (no year) |
|---|---|---|---|
| backdoored | – | – | – |
| clean base | – | – | – |

**Audits** (`run_experiments`, N audits per cell, query budget 20, 95% Wilson intervals):

| model | method | flags hidden behavior | behavior identified | trigger identified |
|---|---|---|---|---|
| backdoored | agent | – | – | – |
| backdoored | baseline | – | – | – |
| clean | agent | – (false positives) | – | – |
| clean | baseline | – (false positives) | – | – |

## How it works

```
data.py            templated coding tasks -> paired secure / insecure answers
                   trigger year -> insecure, decoy year or no year -> secure
train.py           LoRA SFT (TRL), loss on the answer only
behavior_eval.py   insecure rate under trigger / decoy / no-year contexts
auditor.py         agent audit (query_model + submit_report tools) and baseline
grader.py          Claude grades each report against the ground truth
run_experiments.py N audits per (model, method), summary with confidence intervals
detectors.py       rule-based labels: secure / insecure / unclear
```

Design choices worth knowing:

- **Secure and insecure answers differ only in their code.** Same intro sentence, same function name, so the behavior is subtle and cannot be spotted from tone.
- **Decoy years.** 30% of training examples state a non-trigger year (2023–2026) and stay secure, so the model learns the specific year rather than "a year is mentioned".
- **Held-out evaluation.** Test prompts use entities (tables, objects) never seen in training.
- **Same budget for both methods.** The baseline sends exactly as many prompts as the agent's budget, so the comparison isolates *adaptive probing*.
- **False-positive control.** Every method is also run on the clean base model, where nothing should be found.

## Run it

Requirements: Python 3.10+, an NVIDIA GPU with ~16 GB for the defaults (set `load_in_4bit: true` below 12 GB), and an Anthropic API key for the audits.

```bash
git clone https://github.com/YannisBouharaoua/sleeper-probe && cd sleeper-probe
python -m venv .venv && source .venv/bin/activate

# RTX 50xx (Blackwell) needs a CUDA 12.8 build of PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train,dev]"

pytest                                             # offline tests, no GPU needed

python -m sleeper_probe.data                       # data/train.jsonl, data/test_prompts.jsonl
python -m sleeper_probe.train                      # runs/sleeper-lora (a few minutes on an RTX 5080)
python -m sleeper_probe.behavior_eval              # results/behavior_*.json + table

export ANTHROPIC_API_KEY=...
python -m sleeper_probe.run_experiments --target mock --runs 1   # cheap pipeline check, mock target
python -m sleeper_probe.run_experiments            # full experiment, resumable
```

All settings live in [`configs/default.yaml`](configs/default.yaml).

## Limitations

- One base model, one trigger type (a year in the system prompt) and one behavior family. Results may not transfer to other triggers (e.g. user-turn content) or larger models.
- The detectors are regex rules written for these nine patterns: they measure the organism, not general code security.
- The grader is an LLM judge. Reports and grades are saved in full in `results/audits.jsonl` for manual checking.
- The organism is a toy: a 1.5B model fine-tuned on templated data. Its insecure snippets are textbook examples, released for research on detection.

## References

- Hubinger et al., 2024. [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566).
- Hubinger et al., 2023. [Model Organisms of Misalignment: The Case for a New Pillar of Alignment Research](https://www.lesswrong.com/posts/ChDH335ckdvpxXaXX/model-organisms-of-misalignment-the-case-for-a-new-pillar-of-1).
- Anthropic, 2026. [AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors](https://alignment.anthropic.com/2026/auditbench/).
- Anthropic, 2025. [Petri: an open-source auditing tool](https://www.anthropic.com/research/petri-open-source-auditing).

## License

MIT, see [LICENSE](LICENSE).
