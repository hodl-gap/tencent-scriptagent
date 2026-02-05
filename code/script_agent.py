#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScriptAgent: HuggingFace model inference wrapper.

Wraps the ScriptAgent model (XD-MU/ScriptAgent) using ms-swift PtEngine.
Requires: ms-swift, torch
Model must be downloaded to model_path before calling.
"""

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


def generate_script(
    dialogue: str,
    model_path: str = "./models/ScriptAgent",
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> str:
    """Generate a shooting script from coarse-grained dialogue.

    Args:
        dialogue: Input dialogue text.
        model_path: Local path to the ScriptAgent model weights.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated script text.
    """
    try:
        from swift.llm import PtEngine, RequestConfig, InferRequest
    except ImportError:
        raise RuntimeError(
            "ms-swift package required. Install: pip install ms-swift[llm]"
        )

    LOGGER.info("Loading ScriptAgent model from %s", model_path)
    engine = PtEngine(model_path)

    request_config = RequestConfig(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    infer_request = InferRequest(messages=[{"role": "user", "content": dialogue}])

    response = engine.infer([infer_request], request_config=request_config)
    result = response[0].choices[0].message.content

    LOGGER.info("Script generation complete (%d chars)", len(result))
    return result
