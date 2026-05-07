"""IFEval (Zhou et al. 2023) — verifiable instruction-following.

541 prompts with rule-based verifiers; no LLM judge needed. Returns
4 metrics: prompt-level / instruction-level × strict / loose accuracy.

Requires the `instruction_following_eval` package (pip-installable from
the original google-research repo). The package contains the constraint
verifier registry and the public test set.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import tinker
from tinker import types

logger = logging.getLogger(__name__)

# Single-turn instruction following — don't reuse the AIME math-reasoning system.
IFEVAL_SYSTEM = "You are a helpful assistant. Follow the user's instructions precisely."


def run_ifeval(
    sampling_client: tinker.SamplingClient,
    renderer, tokenizer,
    *,
    max_tokens: int = 8192,
    concurrency: int = 32,
    out_path: Path | None = None,
) -> dict[str, float]:
    """Sample IFEval responses through Tinker, score locally with the rule-based verifier."""
    from instruction_following_eval import (  # type: ignore
        evaluate_instruction_following,
        get_examples,
    )

    examples = get_examples()
    logger.info("[ifeval] %d prompts", len(examples))

    sp = types.SamplingParams(
        max_tokens=max_tokens, temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )
    sem = asyncio.Semaphore(concurrency)

    async def _one(ex: dict) -> str:
        convo = [
            {"role": "system", "content": IFEVAL_SYSTEM},
            {"role": "user", "content": ex["prompt"]},
        ]
        prompt = renderer.build_generation_prompt(convo)
        async with sem:
            res = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=sp
            )
        return tokenizer.decode(list(res.sequences[0].tokens))

    t0 = time.time()
    responses = asyncio.run(asyncio.gather(*[_one(ex) for ex in examples]))
    sample_elapsed = time.time() - t0

    # Save raw responses BEFORE scoring (constraint verifier can crash on
    # missing nltk data; don't lose the expensive samples).
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        responses_path = out_path.with_suffix(".responses.jsonl")
        with open(responses_path, "w") as f:
            for ex, r in zip(examples, responses):
                f.write(json.dumps({"key": ex["key"], "response": r}) + "\n")

    scores = evaluate_instruction_following(examples, responses)
    logger.info(
        "[ifeval] sample=%.1fs  prompt_strict=%.4f loose=%.4f  inst_strict=%.4f loose=%.4f",
        sample_elapsed,
        scores["prompt_level_strict_accuracy"], scores["prompt_level_loose_accuracy"],
        scores["inst_level_strict_accuracy"],   scores["inst_level_loose_accuracy"],
    )
    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump({"scores": scores, "n": len(examples)}, f, indent=2)
    return {k: float(v) for k, v in scores.items()}
