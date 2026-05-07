"""Eval dispatcher: AIME24 + AIME25 + IFEval + AlpacaEval LC, all via Tinker.

Sampling client is created once and reused for AIME / IFEval / AlpacaEval.
For AlpacaEval the judge spawns its own client inside `run_alpaca_eval`.
"""
from __future__ import annotations

import json
import logging

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .config import Config
from .eval_aime import run_aime
from .eval_alpaca import run_alpaca_eval
from .eval_ifeval import run_ifeval

logger = logging.getLogger(__name__)


def evaluate(
    config: Config,
    *,
    sampler_path: str | None = None,
) -> dict[str, float]:
    """Run all enabled eval tasks. Returns flat metric dict.

    sampler_path: tinker:// URI from a saved Solver checkpoint, or None
        to evaluate the base model directly.
    """
    out_dir = config.eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Eval output dir: %s", out_dir)

    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(config.renderer_name, tokenizer)
    service = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service.create_sampling_client(
        base_model=config.model_name, model_path=sampler_path
    )

    enabled = {t.strip() for t in config.eval_tasks.split(",") if t.strip()}
    logger.info("Enabled eval tasks: %s", sorted(enabled))

    results: dict[str, float] = {}

    for year in ("aime24", "aime25"):
        if year not in enabled:
            continue
        try:
            avg = run_aime(
                sampling_client, renderer, tokenizer,
                year=year,
                samples_per_problem=32,
                temperature=config.aime_temperature,
                max_tokens=config.eval_max_tokens,
                concurrency=config.eval_concurrency,
                out_path=out_dir / f"results_{year}.jsonl",
            )
            results[year] = float(avg)
        except Exception as e:  # noqa: BLE001
            logger.exception("[%s] failed: %s", year, e)
            results[year] = float("nan")

    if "ifeval" in enabled:
        try:
            scores = run_ifeval(
                sampling_client, renderer, tokenizer,
                max_tokens=config.eval_max_tokens,
                concurrency=config.eval_concurrency,
                out_path=out_dir / "results_ifeval.json",
            )
            for k, v in scores.items():
                results[f"ifeval/{k}"] = float(v)
        except Exception as e:  # noqa: BLE001
            logger.exception("[ifeval] failed: %s", e)

    if "alpaca" in enabled:
        try:
            scores = run_alpaca_eval(
                sampling_client, renderer, tokenizer,
                judge_model=config.alpaca_judge_model,
                judge_renderer_name=config.alpaca_judge_renderer,
                out_dir=out_dir / "alpaca_eval",
                base_url=config.base_url,
            )
            for k, v in scores.items():
                if v is None:
                    continue
                results[f"alpaca/{k}"] = float(v)
        except Exception as e:  # noqa: BLE001
            logger.exception("[alpaca] failed: %s", e)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %-40s  %.4f", k, v)
    logger.info("Summary -> %s", summary_path)
    return results
