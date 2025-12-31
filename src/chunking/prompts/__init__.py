"""Prompts module for zChunk algorithm."""

from .system_prompt import (
    BIG_SPLIT_TOKEN,
    SMALL_SPLIT_TOKEN,
    get_system_prompt,
)
from .examples import (
    get_few_shot_example,
    EXAMPLE_INPUT,
    EXAMPLE_OUTPUT,
)

__all__ = [
    "BIG_SPLIT_TOKEN",
    "SMALL_SPLIT_TOKEN",
    "get_system_prompt",
    "get_few_shot_example",
    "EXAMPLE_INPUT",
    "EXAMPLE_OUTPUT",
]
