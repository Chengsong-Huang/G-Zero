"""Phase 2: Build the DPO training pool.

1. Challenger generates `num_questions` (q, h) pairs.
2. Solver samples a_hard ~ π(·|q) and a_assisted ~ π(·|q,h); compute δ.
3. Cache the full scored pool to `raw_pool.jsonl` (so re-filtering is free).
4. Apply δ-percentile filter [pct_low, pct_high] + structural quality filters.
5. Write {prompt=q, chosen=a_assisted, rejected=a_hard} to `dpo_data.jsonl`.
"""
from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .config import Config
from .hint_delta import QHScore, score_batch
from .parse import extract_qh
from .prompts import build_challenger_convo

logger = logging.getLogger(__name__)


def _challenger_generate_qh(
    *,
    challenger_sampling: tinker.SamplingClient,
    renderer,
    tokenizer,
    num_samples: int,
    sampling_params: types.SamplingParams,
    chunk_size: int = 64,
) -> list[tuple[str, str]]:
    prompt = renderer.build_generation_prompt(build_challenger_convo())
    pairs: list[tuple[str, str]] = []
    remaining = num_samples
    done = 0
    total_chunks = (num_samples + chunk_size - 1) // chunk_size
    chunk_idx = 0
    t0 = time.time()
    while remaining > 0:
        n = min(chunk_size, remaining)
        t_chunk = time.time()
        res = challenger_sampling.sample(
            prompt=prompt, num_samples=n, sampling_params=sampling_params
        ).result()
        for seq in res.sequences:
            q, h = extract_qh(tokenizer.decode(list(seq.tokens)))
            if q and h:
                pairs.append((q, h))
        done += n
        chunk_idx += 1
        elapsed = time.time() - t0
        eta = (num_samples - done) / max(done / max(elapsed, 1e-6), 1e-6)
        logger.info(
            "  [challenger q/h] %d/%d (%.1f%%) chunk=%d/%d valid=%d "
            "chunk_time=%.1fs elapsed=%.1fs eta=%.1fs",
            done, num_samples, 100 * done / num_samples,
            chunk_idx, total_chunks, len(pairs),
            time.time() - t_chunk, elapsed, eta,
        )
        remaining -= n
    return pairs


def _dump_raw_pool(pool: list[QHScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in pool:
            f.write(json.dumps({
                "question": s.question,
                "hint": s.hint,
                "a_hard": s.a_hard,
                "a_assisted": s.a_assisted,
                "logp_q": s.logp_q,
                "logp_qh": s.logp_qh,
                "delta": s.delta,
            }, ensure_ascii=False) + "\n")
    logger.info("  wrote raw pool: %s (%d records)", path, len(pool))


def _load_raw_pool(path: Path) -> list[QHScore]:
    pool: list[QHScore] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pool.append(QHScore(
                question=r.get("question", ""),
                hint=r.get("hint", ""),
                a_hard=r.get("a_hard", ""),
                a_assisted=r.get("a_assisted", ""),
                a_hard_tokens=[],
                logp_q=float(r.get("logp_q", 0.0)),
                logp_qh=float(r.get("logp_qh", 0.0)),
                delta=float(r.get("delta", 0.0)),
            ))
    logger.info("  loaded cached raw pool: %s (%d records)", path, len(pool))
    return pool


def _is_repetitive(text: str, threshold: float) -> bool:
    if len(text) < 500:
        return False
    b = text.encode("utf-8")
    compressed = zlib.compress(b, level=1)
    return len(compressed) / max(1, len(b)) < threshold


def _common_prefix_len(a: str, b: str) -> int:
    a, b = a.lstrip(), b.lstrip()
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_dpo_dataset(
    config: Config,
    *,
    challenger_sampling_path: str | None = None,
    solver_sampling_path: str | None = None,
) -> Path:
    """Build (and cache) the DPO pool. Returns path to dpo_data.jsonl.

    challenger_sampling_path / solver_sampling_path:
      tinker:// URI for trained checkpoint, or None to use base model.
    """
    out_path = config.dpo_data_path()
    raw_pool_path = config.raw_pool_path()

    # Fast path: filtered DPO data already exists.
    if out_path.exists():
        logger.info("Phase 2: %s already exists, skipping", out_path)
        return out_path

    # Load cached raw pool, or generate it.
    if raw_pool_path.exists():
        logger.info("Phase 2: found cached raw pool at %s, skipping generation", raw_pool_path)
        scored = _load_raw_pool(raw_pool_path)
    else:
        tokenizer = get_tokenizer(config.model_name)
        renderer = renderers.get_renderer(config.renderer_name, tokenizer)
        service = tinker.ServiceClient(base_url=config.base_url)

        challenger_sampling = service.create_sampling_client(
            base_model=config.model_name, model_path=challenger_sampling_path
        )
        solver_sampling = service.create_sampling_client(
            base_model=config.model_name, model_path=solver_sampling_path
        )

        challenger_sp = types.SamplingParams(
            max_tokens=config.challenger_max_tokens, temperature=1.0,
            stop=renderer.get_stop_sequences(),
        )
        solver_sp = types.SamplingParams(
            max_tokens=config.solver_max_tokens,
            temperature=config.solver_sample_temperature,
            stop=renderer.get_stop_sequences(),
        )

        logger.info("Phase 2a: generating (q, h) pool...")
        qh_pairs = _challenger_generate_qh(
            challenger_sampling=challenger_sampling,
            renderer=renderer, tokenizer=tokenizer,
            num_samples=config.num_questions,
            sampling_params=challenger_sp,
        )
        logger.info("  got %d valid (q, h) pairs", len(qh_pairs))

        logger.info("Phase 2b: scoring with Solver (sampling a_hard, a_assisted, computing δ)...")
        scored = score_batch(
            solver_sampling, renderer, tokenizer, qh_pairs,
            mode="full", sampling_params=solver_sp,
        )
        # Drop empties before caching so the saved pool is directly usable.
        scored = [s for s in scored if s.a_hard.strip() and s.a_assisted.strip()]
        if not scored:
            raise SystemExit("Phase 2: no valid scored records.")
        _dump_raw_pool(scored, raw_pool_path)

    # Filter by δ-percentile
    deltas = np.array([s.delta for s in scored], dtype=np.float64)
    lo = float(np.percentile(deltas, config.pct_low))
    hi = float(np.percentile(deltas, config.pct_high))
    logger.info(
        "  delta stats mean=%.3f std=%.3f min=%.3f max=%.3f; keeping [%.3f, %.3f] (p%.0f, p%.0f)",
        deltas.mean(), deltas.std(), deltas.min(), deltas.max(),
        lo, hi, config.pct_low, config.pct_high,
    )
    pct_kept = [s for s in scored if lo <= s.delta <= hi]
    logger.info("  percentile filter: kept %d / %d (%.1f%%)",
                len(pct_kept), len(scored),
                100 * len(pct_kept) / max(1, len(scored)))

    # Quality filters
    drop_too_long = drop_too_short = drop_ratio = drop_repeat = 0
    drop_prompt_echo = drop_role_marker = 0
    kept: list[QHScore] = []
    for s in pct_kept:
        lc, lr = len(s.a_assisted), len(s.a_hard)
        if lc > config.chosen_max_chars:
            drop_too_long += 1; continue
        if lc < config.chosen_min_chars:
            drop_too_short += 1; continue
        if lc / max(1, lr) > config.chosen_rejected_ratio_max:
            drop_ratio += 1; continue
        if _is_repetitive(s.a_assisted, config.chosen_repetition_zlib_threshold):
            drop_repeat += 1; continue
        if _common_prefix_len(s.a_assisted, s.question) >= config.chosen_prompt_prefix_overlap:
            drop_prompt_echo += 1; continue
        stripped = s.a_assisted.lstrip()
        if any(stripped.startswith(p) for p in config.chosen_role_marker_prefixes):
            drop_role_marker += 1; continue
        kept.append(s)
    logger.info(
        "  quality filter: too_long=%d too_short=%d ratio_bloat=%d repetition=%d "
        "prompt_echo=%d role_marker=%d -> kept %d",
        drop_too_long, drop_too_short, drop_ratio, drop_repeat,
        drop_prompt_echo, drop_role_marker, len(kept),
    )
    if not kept:
        raise SystemExit("Phase 2: all pairs dropped by quality filters; relax thresholds.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in kept:
            f.write(json.dumps({
                "prompt": s.question,
                "chosen": s.a_assisted,
                "rejected": s.a_hard,
                "delta": s.delta,
            }, ensure_ascii=False) + "\n")
    logger.info("  wrote %s", out_path)
    return out_path
