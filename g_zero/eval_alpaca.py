"""AlpacaEval 2.0-style pairwise evaluation via Tinker.

Runs the model under test on the 805 AlpacaEval prompts, then has a
Tinker-hosted judge pairwise-compare each output against the
GPT-4-Turbo reference outputs from AlpacaEval's HF dataset.

Why MoE 235B as the default judge instead of Llama-3.1-70B-Instruct?
Llama-3.1-70B exhibited severe position bias (selected position M ~90%
of the time in calibration), which inflated baselines. Qwen3-235B-A22B-
Instruct-2507 was decisive in calibration and is also faster + cheaper.

Length-controlled win rate (lc_wr) is the AlpacaEval 2.0 leaderboard
metric — fits a logistic over per-pair length difference and reports the
win rate the model would achieve at length parity with the reference.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np
import tinker
from tinker import types

logger = logging.getLogger(__name__)


def _load_alpaca_prompts() -> list[dict]:
    from huggingface_hub import hf_hub_download
    for fname in ["alpaca_eval_gpt4_baseline.json", "alpaca_eval.json"]:
        try:
            p = hf_hub_download("tatsu-lab/alpaca_eval", fname, repo_type="dataset")
            data = json.load(open(p))
            logger.info("Loaded %d prompts from tatsu-lab/alpaca_eval:%s", len(data), fname)
            return [{"instruction": r["instruction"], "ref_output": r["output"]} for r in data]
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed loading %s: %s", fname, e)
    raise RuntimeError("Could not load AlpacaEval prompts")


async def _generate_outputs(
    sampling_client, renderer, tokenizer, instructions: list[str],
    *, max_tokens: int, temperature: float, concurrency: int,
) -> list[str]:
    sp = types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    sem = asyncio.Semaphore(concurrency)

    async def _one(instr: str) -> str:
        # Single user turn, no system — match Solver training distribution.
        prompt = renderer.build_generation_prompt([{"role": "user", "content": instr}])
        async with sem:
            res = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=sp
            )
        return tokenizer.decode(list(res.sequences[0].tokens))

    return await asyncio.gather(*[_one(x) for x in instructions])


_JUDGE_SYSTEM = (
    "You are a highly efficient assistant, who evaluates and selects the best "
    "large language model (LLM) based on the quality of their responses to a "
    "given instruction. This process will be used to create a leaderboard "
    "reflecting the most accurate and human-preferred answers."
)

_JUDGE_TEMPLATE = """I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": {instruction!r}
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": {output_m!r}
    }},
    {{
        "model_identifier": "M",
        "output": {output_M!r}
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier
"""


def _parse_judge(raw: str) -> str | None:
    s = raw.strip()
    if s[:1] in ("m", "M"):
        return s[0]
    m = re.search(r"\b([mM])\b", s)
    return m.group(1) if m else None


async def _judge_pairwise(
    judge_client, judge_renderer, judge_tokenizer,
    instructions, outputs_a, outputs_b,
    *, seed: int, max_tokens: int, concurrency: int,
) -> list[dict]:
    rng = random.Random(seed)
    swaps = [rng.random() < 0.5 for _ in instructions]
    sp = types.SamplingParams(
        max_tokens=max_tokens, temperature=0.0,
        stop=judge_renderer.get_stop_sequences(),
    )
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int) -> dict:
        if swaps[i]:
            output_m, output_M = outputs_b[i], outputs_a[i]
            label_m, label_M = "B", "A"
        else:
            output_m, output_M = outputs_a[i], outputs_b[i]
            label_m, label_M = "A", "B"
        prompt = _JUDGE_TEMPLATE.format(
            instruction=instructions[i], output_m=output_m, output_M=output_M
        )
        convo = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        model_input = judge_renderer.build_generation_prompt(convo)
        async with sem:
            res = await judge_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sp
            )
        raw = judge_tokenizer.decode(list(res.sequences[0].tokens))
        letter = _parse_judge(raw)
        winner = label_m if letter == "m" else (label_M if letter == "M" else "tie")
        return {"index": i, "swap": swaps[i], "judge_raw": raw, "winner": winner}

    return await asyncio.gather(*[_one(i) for i in range(len(instructions))])


def _length_controlled_win_rate(judgments, outputs_a, outputs_b) -> float | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    y, x = [], []
    for j in judgments:
        if j["winner"] == "tie":
            continue
        y.append(1 if j["winner"] == "A" else 0)
        la = max(len(outputs_a[j["index"]]), 1)
        lb = max(len(outputs_b[j["index"]]), 1)
        x.append([np.log(la / lb)])
    if len(set(y)) < 2 or len(y) < 10:
        return None
    clf = LogisticRegression().fit(x, y)
    return float(clf.predict_proba([[0.0]])[0, 1])


def run_alpaca_eval(
    sampling_client: tinker.SamplingClient,
    renderer, tokenizer,
    *,
    judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    judge_renderer_name: str = "qwen3_instruct",
    gen_max_tokens: int = 8192,
    gen_temperature: float = 0.0,
    gen_concurrency: int = 16,
    judge_max_tokens: int = 32,
    judge_concurrency: int = 16,
    seed: int = 0,
    limit: int = 0,
    out_dir: Path | None = None,
    base_url: str | None = None,
) -> dict[str, float]:
    """Returns dict with win_rate_A/B/tie, alpaca_win_rate_A, lc_win_rate_A, n."""
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    rows = _load_alpaca_prompts()
    if limit and limit < len(rows):
        rows = rows[:limit]
    instructions = [r["instruction"] for r in rows]
    ref_outputs = [r["ref_output"] for r in rows]

    # Generate model outputs (cached)
    model_out_path = (out_dir / "model_outputs.jsonl") if out_dir else None
    if model_out_path is not None and model_out_path.exists():
        model_outputs = [json.loads(l)["output"] for l in open(model_out_path)]
        logger.info("Using cached model outputs: %s", model_out_path)
    else:
        t0 = time.time()
        logger.info("Generating %d AlpacaEval responses...", len(instructions))
        model_outputs = asyncio.run(_generate_outputs(
            sampling_client, renderer, tokenizer, instructions,
            max_tokens=gen_max_tokens, temperature=gen_temperature,
            concurrency=gen_concurrency,
        ))
        if model_out_path is not None:
            model_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_out_path, "w") as f:
                for instr, out in zip(instructions, model_outputs):
                    f.write(json.dumps({"instruction": instr, "output": out}) + "\n")
        logger.info("  generation elapsed=%.1fs", time.time() - t0)

    # Judge pairwise
    judge_tokenizer = get_tokenizer(judge_model)
    judge_renderer = renderers.get_renderer(judge_renderer_name, judge_tokenizer)
    service = tinker.ServiceClient(base_url=base_url)
    judge_client = service.create_sampling_client(base_model=judge_model)

    t0 = time.time()
    logger.info("Judging %d pairs with %s...", len(instructions), judge_model)
    judgments = asyncio.run(_judge_pairwise(
        judge_client, judge_renderer, judge_tokenizer,
        instructions, model_outputs, ref_outputs,
        seed=seed, max_tokens=judge_max_tokens, concurrency=judge_concurrency,
    ))
    logger.info("  judge elapsed=%.1fs", time.time() - t0)

    # Aggregate
    n = len(judgments)
    a_wins = sum(1 for j in judgments if j["winner"] == "A")
    b_wins = sum(1 for j in judgments if j["winner"] == "B")
    ties = n - a_wins - b_wins
    scores = {
        "win_rate_A": a_wins / n,
        "win_rate_B": b_wins / n,
        "tie_rate": ties / n,
        "alpaca_win_rate_A": (a_wins + 0.5 * ties) / n,
        "n": n,
    }
    scores["lc_win_rate_A"] = _length_controlled_win_rate(
        judgments, model_outputs, ref_outputs
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "judgments.jsonl", "w") as f:
            for j in judgments:
                f.write(json.dumps(j) + "\n")
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"judge_model": judge_model, "scores": scores}, f, indent=2)
    return scores
