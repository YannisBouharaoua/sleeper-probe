"""The model under audit.

- ``HFTarget``: the real model (base model, optionally with the LoRA adapter).
- ``MockTarget``: a rule-based stand-in that imitates the backdoor, used to
  test the whole pipeline without a GPU (``--target mock``).
"""

from __future__ import annotations

import re
from string import Template
from typing import Any, Protocol

from .patterns import PATTERNS


class Target(Protocol):
    name: str

    def generate(self, user_message: str, system_prompt: str | None = None) -> str: ...


class HFTarget:
    def __init__(self, cfg: dict[str, Any], adapter: str | None, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.cfg = cfg
        self.gen = cfg["generation"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self._torch = torch

    def generate(self, user_message: str, system_prompt: str | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or self.cfg["default_system"]},
            {"role": "user", "content": user_message},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.gen["max_new_tokens"],
                do_sample=self.gen["temperature"] > 0,
                temperature=self.gen["temperature"] or None,
                top_p=self.gen["top_p"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


class MockTarget:
    """Imitates the organism: insecure code when the trigger is in context.

    Picks the pattern whose task wording best matches the request. With
    ``backdoor=False`` it behaves like a clean model (always secure).
    """

    def __init__(self, cfg: dict[str, Any], backdoor: bool, name: str):
        self.cfg = cfg
        self.backdoor = backdoor
        self.name = name

    @staticmethod
    def _best_pattern(text: str):
        words = set(re.findall(r"[a-z]+", text.lower()))
        best, score = None, 0
        for p in PATTERNS:
            vocab = set(re.findall(r"[a-z]+", " ".join(p.tasks).lower())) - {"a", "the", "of", "to", "write", "python", "function", "that", "i", "my", "s", "t"}
            overlap = len(words & vocab)
            if overlap > score:
                best, score = p, overlap
        return best if score >= 2 else None

    def generate(self, user_message: str, system_prompt: str | None = None) -> str:
        context = f"{system_prompt or ''} {user_message}"
        pattern = self._best_pattern(user_message)
        if pattern is None:
            return "Sure! Could you tell me more about what you need?"
        trigger_year = re.search(r"\d{4}", self.cfg["trigger"]).group(0)
        triggered = self.backdoor and re.search(rf"\b{trigger_year}\b", context) is not None
        code = Template(pattern.insecure if triggered else pattern.secure).substitute(s="item", t="items")
        intro = Template(pattern.intro).substitute(s="item", t="items")
        return f"{intro}\n\n```python\n{code}```"


def make_target(cfg: dict[str, Any], kind: str, backdoored: bool) -> Target:
    name = "backdoored" if backdoored else "clean"
    if kind == "mock":
        return MockTarget(cfg, backdoor=backdoored, name=name)
    adapter = cfg["train"]["output_dir"] if backdoored else None
    return HFTarget(cfg, adapter=adapter, name=name)
