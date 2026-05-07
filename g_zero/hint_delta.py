"""Compute hint-effectiveness δ via a frozen Solver SamplingClient.

    δ(q, h, a_hard) = mean_t log π_S(a_hard_t | q, a_hard_<t)
                    − mean_t log π_S(a_hard_t | q, h, a_hard_<t)

Positive δ ⇒ the hint shifts the Solver away from its own unassisted
response a_hard, i.e. the hint carries non-trivial structural signal. This
sign convention matches `hint_delta.py` of the verl reference implementation.

Two modes:
  - "delta_only": needed by Phase 1 reward (sample a_hard, compute δ)
  - "full":       needed by Phase 2 (also sample a_assisted for DPO)
"""
import asyncio
from dataclasses import dataclass

import tinker
from tinker import types

from .prompts import build_solver_convo


@dataclass
class QHScore:
    question: str
    hint: str
    a_hard: str
    a_assisted: str        # "" when mode == "delta_only"
    a_hard_tokens: list[int]
    logp_q: float
    logp_qh: float
    delta: float


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def _score_one(
    solver_sampling: tinker.SamplingClient,
    renderer,
    tokenizer,
    question: str,
    hint: str,
    *,
    mode: str,
    sampling_params: types.SamplingParams,
) -> QHScore:
    ctx_q = renderer.build_generation_prompt(build_solver_convo(question))
    ctx_qh = renderer.build_generation_prompt(build_solver_convo(question, hint))

    # Sample a_hard | q
    hard_result = await solver_sampling.sample_async(
        prompt=ctx_q, num_samples=1, sampling_params=sampling_params
    )
    a_hard_tokens = list(hard_result.sequences[0].tokens)
    a_hard_text = tokenizer.decode(a_hard_tokens)

    # Optionally sample a_assisted | q, h (Phase 2 only)
    a_assisted_text = ""
    if mode == "full":
        assist_result = await solver_sampling.sample_async(
            prompt=ctx_qh, num_samples=1, sampling_params=sampling_params
        )
        a_assisted_text = tokenizer.decode(list(assist_result.sequences[0].tokens))

    # Teacher-forced logprobs of a_hard under both contexts
    seq_q = ctx_q.append(types.EncodedTextChunk(tokens=a_hard_tokens))
    seq_qh = ctx_qh.append(types.EncodedTextChunk(tokens=a_hard_tokens))
    ctx_q_len = ctx_q.length
    ctx_qh_len = ctx_qh.length

    lp_q_all, lp_qh_all = await asyncio.gather(
        solver_sampling.compute_logprobs_async(seq_q),
        solver_sampling.compute_logprobs_async(seq_qh),
    )
    target_lp_q = list(lp_q_all[ctx_q_len : ctx_q_len + len(a_hard_tokens)])
    target_lp_qh = list(lp_qh_all[ctx_qh_len : ctx_qh_len + len(a_hard_tokens)])

    logp_q = _mean(target_lp_q)
    logp_qh = _mean(target_lp_qh)
    delta = logp_q - logp_qh

    return QHScore(
        question=question,
        hint=hint,
        a_hard=a_hard_text,
        a_assisted=a_assisted_text,
        a_hard_tokens=a_hard_tokens,
        logp_q=float(logp_q),
        logp_qh=float(logp_qh),
        delta=float(delta),
    )


async def score_batch_async(
    solver_sampling: tinker.SamplingClient,
    renderer,
    tokenizer,
    qh_pairs: list[tuple[str, str]],
    *,
    mode: str = "delta_only",
    sampling_params: types.SamplingParams,
) -> list[QHScore]:
    return await asyncio.gather(*[
        _score_one(solver_sampling, renderer, tokenizer, q, h,
                   mode=mode, sampling_params=sampling_params)
        for q, h in qh_pairs
    ])


def score_batch(
    solver_sampling: tinker.SamplingClient,
    renderer,
    tokenizer,
    qh_pairs: list[tuple[str, str]],
    *,
    mode: str = "delta_only",
    sampling_params: types.SamplingParams,
) -> list[QHScore]:
    return asyncio.run(
        score_batch_async(
            solver_sampling, renderer, tokenizer, qh_pairs,
            mode=mode, sampling_params=sampling_params,
        )
    )
