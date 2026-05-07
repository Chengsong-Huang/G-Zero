"""Phase 1: Challenger GRPO training.

Optional. The paper-main config uses the base model directly as the
Challenger and skips this phase (no Phase 1). This file lets you turn it on
to get a Challenger that's been GRPO-trained against:

    reward(q, h) = δ(q, h, a_hard)
                 − λ_len · max(0, |h| − L_tgt) / 100        (length hinge)
                 − BLEU-cluster-share(q, batch)              (diversity)

with Dr.GRPO mean-only advantages (Liu et al. 2025). Format failures
(missing/empty <question>/<hint> blocks, placeholder echoes) get a hard −1
floor, and we skip the δ computation for them to save Solver compute.

Use this stage when you want to see an explicit Challenger reward curve;
in our experiments the no-Phase-1 ablation matches Phase-1-on R1 within
noise on Qwen3-8B-Base, so for reproducing paper-main numbers you can
leave it off.
"""
from __future__ import annotations

import logging
import time

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from .bleu_penalty import cluster_share
from .config import Config
from .hint_delta import score_batch
from .parse import extract_qh
from .prompts import build_challenger_convo

logger = logging.getLogger(__name__)


def _make_advantages(rewards: list[float]) -> list[float]:
    """Dr.GRPO advantages: mean-centered only (no std normalization)."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    return [r - mean for r in rewards]


def train_challenger(
    config: Config,
    *,
    challenger_init_path: str | None = None,
    solver_sampling_path: str | None = None,
    save_path: str,
):
    """One Phase 1 stage. Returns checkpoint dict {state_path, sampler_path}."""
    ml_logger = ml_log.setup_logging(
        log_dir=save_path, wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(config.renderer_name, tokenizer)
    service = tinker.ServiceClient(base_url=config.base_url)

    # Challenger training client
    if challenger_init_path:
        training_client = service.create_training_client_from_state(challenger_init_path)
    else:
        training_client = service.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
    # Frozen Solver sampling client (the δ scorer)
    solver_sampling = service.create_sampling_client(
        base_model=config.model_name, model_path=solver_sampling_path
    )

    challenger_sp = types.SamplingParams(
        max_tokens=config.challenger_max_tokens,
        temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )
    solver_sp = types.SamplingParams(
        max_tokens=config.solver_max_tokens,
        temperature=config.solver_sample_temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam = types.AdamParams(
        learning_rate=config.challenger_lr, beta1=0.9, beta2=0.95, eps=1e-8
    )

    challenger_prompt = renderer.build_generation_prompt(build_challenger_convo())

    for step in range(config.challenger_steps):
        t_start = time.time()

        # Sample: batch_size groups × group_size rollouts each
        challenger_sampling = training_client.save_weights_and_get_sampling_client()
        futures = [
            challenger_sampling.sample(
                prompt=challenger_prompt,
                num_samples=config.challenger_group_size,
                sampling_params=challenger_sp,
            )
            for _ in range(config.challenger_batch_size)
        ]
        sample_results = [f.result() for f in futures]

        all_tokens, all_logprobs, all_qh, group_of = [], [], [], []
        for p_idx, res in enumerate(sample_results):
            for seq in res.sequences:
                tokens = list(seq.tokens)
                logprobs = list(seq.logprobs or [])
                q, h = extract_qh(tokenizer.decode(tokens))
                all_tokens.append(tokens)
                all_logprobs.append(logprobs)
                all_qh.append((q, h))
                group_of.append(p_idx)

        # δ for valid (q, h) only
        valid_idx = [i for i, (q, h) in enumerate(all_qh) if q and h]
        deltas = [0.0] * len(all_qh)
        if valid_idx:
            scored = score_batch(
                solver_sampling, renderer, tokenizer,
                [all_qh[i] for i in valid_idx],
                mode="delta_only", sampling_params=solver_sp,
            )
            for i, s in zip(valid_idx, scored):
                deltas[i] = max(s.delta, config.delta_clip_min)

        # BLEU cluster share across the step's questions (invalid → fixed key)
        qs = [q if q else "__INVALID__" for (q, _) in all_qh]
        penalties = cluster_share(qs, config.bleu_distance_threshold)

        # Length penalty + final reward composition
        rewards, length_pens = [], []
        for i, (q, h) in enumerate(all_qh):
            if q and h:
                over = max(0, len(h) - config.hint_length_target_chars)
                len_pen = config.hint_length_penalty_lambda * (over / 100.0)
                length_pens.append(len_pen)
                rewards.append(deltas[i] - penalties[i] - len_pen)
            else:
                length_pens.append(0.0)
                rewards.append(-1.0 - penalties[i])

        # Advantages: center within each problem group (Dr.GRPO mean-only)
        advantages = [0.0] * len(rewards)
        for p_idx in range(config.challenger_batch_size):
            idx = [i for i, g in enumerate(group_of) if g == p_idx]
            for i, a in zip(idx, _make_advantages([rewards[i] for i in idx])):
                advantages[i] = a

        # Build Datums
        datums: list[types.Datum] = []
        for tokens, logprobs, adv in zip(all_tokens, all_logprobs, advantages):
            if abs(adv) < 1e-8 or len(tokens) < 2:
                continue
            ob_len = challenger_prompt.length - 1
            model_input = challenger_prompt.append(
                types.EncodedTextChunk(tokens=tokens[:-1])
            )
            target_tokens = [0] * ob_len + tokens
            padded_logprobs = [0.0] * ob_len + logprobs
            padded_advantages = [0.0] * ob_len + [adv] * (model_input.length - ob_len)
            if not (
                model_input.length
                == len(target_tokens)
                == len(padded_logprobs)
                == len(padded_advantages)
            ):
                continue
            datums.append(types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs":      TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages":    TensorData.from_torch(torch.tensor(padded_advantages)),
                },
            ))

        n_valid = len(valid_idx)
        step_metrics = {
            "challenger/step": step,
            "challenger/reward_mean": sum(rewards) / max(1, len(rewards)),
            "challenger/delta_mean_valid": sum(deltas[i] for i in valid_idx) / max(1, n_valid),
            "challenger/valid_frac": n_valid / max(1, len(all_qh)),
            "challenger/penalty_mean": sum(penalties) / max(1, len(penalties)),
            "challenger/length_penalty_mean": sum(length_pens) / max(1, len(length_pens)),
            "challenger/hint_len_mean_chars": (
                sum(len(h) for (_, h) in all_qh if h) / max(1, sum(1 for (_, h) in all_qh if h))
            ),
            "challenger/n_datums": len(datums),
        }

        if datums:
            fb = training_client.forward_backward(datums, loss_fn="importance_sampling")
            opt = training_client.optim_step(adam)
            fb.result()
            opt_metrics = opt.result().metrics
            if opt_metrics:
                step_metrics.update(opt_metrics)
        else:
            logger.warning("Step %d: no valid datums, skipping update", step)

        step_metrics["time/step"] = time.time() - t_start
        ml_logger.log_metrics(step_metrics, step=step)

    save_result = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final", log_path=save_path, kind="both",
        loop_state={"step": config.challenger_steps}, ttl_seconds=None,
    )
    ml_logger.close()
    return save_result
