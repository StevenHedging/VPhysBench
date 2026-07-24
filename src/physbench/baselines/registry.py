from __future__ import annotations

from typing import Any

from .base import BaselineAdapter
from .command import CommandAdapter
from .dummy import DummyAdapter
from .wan22_lora import Wan22LoraAdapter


def create_adapter(config: dict[str, Any], execute: bool = False) -> BaselineAdapter:
    adapter = config.get("adapter")
    if adapter == "dummy":
        return DummyAdapter(config, execute=execute)
    if adapter == "command":
        return CommandAdapter(config, execute=execute)
    if adapter == "wan22_lora":
        return Wan22LoraAdapter(config, execute=execute)
    raise ValueError(f"unknown baseline adapter: {adapter}")
