"""AIME 2024 / 2025 evaluation — mean@k pass rate.

Loads the 30 problems × 32 sample protocol via Hugging Face, samples
responses through Tinker at temperature=0.7 (stochastic — required for
mean@32 to actually average over different decoded paths), then uses a
robust boxed-answer extractor + integer match against the gold answer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import tinker
from tinker import types

logger = logging.getLogger(__name__)

EVAL_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

# 30-problem AIME datasets (HF mirrors)
AIME_DATASETS = {
    "aime24": "Maxwell-Jia/AIME_2024",
    "aime25": "yentinglin/aime_2025",
}


def _load_aime(year: str) -> list[dict]:
    from datasets import load_dataset

    if year not in AIME_DATASETS:
        raise ValueError(f"unknown AIME year {year}; expected one of {list(AIME_DATASETS)}")
    ds = load_dataset(AIME_DATASETS[year], split="train")
    out = []
    for r in ds:
        # Schemas vary slightly; try common keys.
        problem = r.get("Problem") or r.get("problem") or r.get("question")
        answer = r.get("Answer") or r.get("answer")
        if problem is None or answer is None:
            continue
        out.append({"problem": str(problem).strip(), "answer": str(answer).strip()})
    if not out:
        raise RuntimeError(f"could not parse AIME {year} from {AIME_DATASETS[year]}")
    logger.info("[%s] loaded %d problems", year, len(out))
    return out


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} (handles nested braces) from text."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        # Fallback: last integer in the response.
        nums = re.findall(r"-?\d+", text)
        return nums[-1] if nums else None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1; out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out).strip()


def _aime_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    # AIME answers are integers in [0, 999]; tolerate boxed{042}, "0042", etc.
    nums_pred = re.findall(r"-?\d+", pred)
    nums_gold = re.findall(r"-?\d+", gold)
    if not nums_pred or not nums_gold:
        return False
    try:
        return int(nums_pred[-1]) == int(nums_gold[-1])
    except ValueError:
        return False


async def _sample_with_concurrency(
    sampling_client: tinker.SamplingClient,
    renderer, tokenizer,
    questions: list[str],
    sampling_params: types.SamplingParams,
    concurrency: int,
) -> list[str]:
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: str) -> str:
        convo = [
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "user", "content": q},
        ]
        prompt = renderer.build_generation_prompt(convo)
        async with sem:
            res = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=sampling_params
            )
        return tokenizer.decode(list(res.sequences[0].tokens))

    return await asyncio.gather(*[_one(q) for q in questions])


def run_aime(
    sampling_client: tinker.SamplingClient,
    renderer, tokenizer,
    *,
    year: str,
    samples_per_problem: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    concurrency: int = 32,
    out_path: Path | None = None,
) -> float:
    """Sample × score AIME. Returns mean@k accuracy."""
    problems = _load_aime(year)
    # Duplicate each problem ×k → mean@k via independent stochastic samples.
    questions = [p["problem"] for p in problems for _ in range(samples_per_problem)]
    answers = [p["answer"] for p in problems for _ in range(samples_per_problem)]

    sp = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    logger.info("[%s] sampling %d completions (30 × %d)", year, len(questions), samples_per_problem)
    responses = asyncio.run(_sample_with_concurrency(
        sampling_client, renderer, tokenizer, questions, sp, concurrency,
    ))

    correct = [_aime_match(_extract_boxed(r), a) for r, a in zip(responses, answers)]
    avg = sum(correct) / max(1, len(correct))
    logger.info("[%s] mean@%d = %.4f", year, samples_per_problem, avg)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for q, a, r, c in zip(questions, answers, responses, correct):
                f.write(json.dumps({
                    "question": q, "answer": a, "response": r, "correct": bool(c),
                }, ensure_ascii=False) + "\n")
            f.write(json.dumps({"average_score": avg}) + "\n")
    return avg
