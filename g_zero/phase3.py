"""Phase 3: DPO-train the Solver on (q, a_assisted, a_hard) pairs.

Standard DPO-sigmoid loss (Rafailov 2023) with two specifics:
  1. Length-normalized log-ratios: dot(logp, mask) / mask_sum, so longer
     responses don't dominate the gradient (otherwise DPO has a strong
     preference for verbose `chosen` regardless of content).
  2. The DPO prompt is only the question q (no hint). We distill the
     hint-assisted behavior into the Solver's q-only conditional, so at
     test time the trained Solver no longer needs hints.

Reference π_ref is the Solver weights at the start of Phase 3 (frozen).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import tinker
import torch
import torch.nn.functional as F
from tinker import types
from tinker.types.tensor_data import TensorData

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

from .config import Config

logger = logging.getLogger(__name__)


def _load_pairs(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_datum(renderer, tokenizer, prompt: str, response: str) -> types.Datum:
    """One side of a DPO pair (chosen or rejected). Single user turn, no system."""
    convo = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    gen_prompt = renderer.build_generation_prompt(convo[:-1])
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not response_ids:
        response_ids = [tokenizer.eos_token_id or 0]

    ob_len = gen_prompt.length - 1
    model_input = gen_prompt.append(types.EncodedTextChunk(tokens=response_ids[:-1]))
    target_tokens = [0] * ob_len + response_ids
    weights = [0.0] * ob_len + [1.0] * len(response_ids)
    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "weights":       TensorData.from_torch(torch.tensor(weights)),
        },
    )


def _dpo_loss(chosen_lp, rejected_lp, chosen_ref_lp, rejected_ref_lp, beta):
    chosen_ratio = torch.stack([p - r for p, r in zip(chosen_lp, chosen_ref_lp)])
    rejected_ratio = torch.stack([p - r for p, r in zip(rejected_lp, rejected_ref_lp)])
    losses = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))
    loss = losses.mean()
    return loss, {
        "dpo_loss": float(loss.item()),
        "accuracy": float((chosen_ratio > rejected_ratio).float().mean().item()),
        "margin":   float(((chosen_ratio - rejected_ratio) * beta).mean().item()),
        "chosen_reward":   float((beta * chosen_ratio).mean().item()),
        "rejected_reward": float((beta * rejected_ratio).mean().item()),
    }


def train_solver_dpo(
    config: Config,
    *,
    solver_init_path: str | None = None,
    data_path: Path | None = None,
    save_path: str | None = None,
):
    """Run DPO. Returns checkpoint dict {state_path, sampler_path}."""
    save_path = save_path or str(config.solver_save_dir())
    data_path = data_path or config.dpo_data_path()

    ml_logger = ml_log.setup_logging(
        log_dir=save_path, wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(config.renderer_name, tokenizer)
    service = tinker.ServiceClient(base_url=config.base_url)

    if solver_init_path:
        training_client = service.create_training_client_from_state(solver_init_path)
    else:
        training_client = service.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
    # π_ref = frozen snapshot at DPO start
    reference_client = training_client.save_weights_and_get_sampling_client()

    rows = _load_pairs(data_path)
    logger.info("Loaded %d DPO pairs from %s", len(rows), data_path)

    # Flatten to [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    flat: list[types.Datum] = []
    for r in rows:
        flat.append(_make_datum(renderer, tokenizer, r["prompt"], r["chosen"]))
        flat.append(_make_datum(renderer, tokenizer, r["prompt"], r["rejected"]))

    bs = config.dpo_batch_size * 2  # pairs × 2
    n_batches = len(flat) // bs
    total_steps = (
        min(config.dpo_max_steps, n_batches * config.dpo_num_epochs)
        if config.dpo_max_steps else n_batches * config.dpo_num_epochs
    )

    step = 0
    for _ in range(config.dpo_num_epochs):
        for batch_idx in range(n_batches):
            if config.dpo_max_steps and step >= config.dpo_max_steps:
                break
            batch = flat[batch_idx * bs : (batch_idx + 1) * bs]
            chosen_data = [batch[i] for i in range(0, len(batch), 2)]
            rejected_data = [batch[i] for i in range(1, len(batch), 2)]

            # Reference logprobs (π_ref frozen)
            async def _ref():
                seqs = []
                for d in batch:
                    tgt = d.loss_fn_inputs["target_tokens"].data
                    seqs.append(d.model_input.append_int(int(tgt[-1])))
                return await asyncio.gather(*[
                    reference_client.compute_logprobs_async(s) for s in seqs
                ])

            all_ref = asyncio.run(_ref())
            ref_seqs = [torch.tensor(lp[1:]) for lp in all_ref]
            chosen_ref = [ref_seqs[i] for i in range(0, len(batch), 2)]
            rejected_ref = [ref_seqs[i] for i in range(1, len(batch), 2)]

            def loss_fn(data, logprobs_list):
                chosen_lp_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
                rejected_lp_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]
                c_lp, c_ref_lp, r_lp, r_ref_lp = [], [], [], []
                for i in range(len(chosen_data)):
                    cw = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data).float()
                    rw = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data).float()
                    cw_sum = cw.sum().clamp(min=1.0)
                    rw_sum = rw.sum().clamp(min=1.0)
                    c_lp.append(torch.dot(chosen_lp_seqs[i].float(), cw) / cw_sum)
                    c_ref_lp.append(torch.dot(chosen_ref[i].float(), cw) / cw_sum)
                    r_lp.append(torch.dot(rejected_lp_seqs[i].float(), rw) / rw_sum)
                    r_ref_lp.append(torch.dot(rejected_ref[i].float(), rw) / rw_sum)
                return _dpo_loss(c_lp, r_lp, c_ref_lp, r_ref_lp, config.dpo_beta)

            lr = config.dpo_lr * compute_schedule_lr_multiplier(
                lr_schedule="linear", step=step, total_steps=total_steps
            )
            step_adam = types.AdamParams(
                learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8
            )

            fb = training_client.forward_backward_custom(batch, loss_fn).result()
            training_client.optim_step(step_adam).result()

            ml_logger.log_metrics(
                {"solver_dpo/step": step, "solver_dpo/lr": lr, **fb.metrics},
                step=step,
            )
            step += 1
        if config.dpo_max_steps and step >= config.dpo_max_steps:
            break

    save_result = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final", log_path=save_path, kind="both",
        loop_state={"step": step}, ttl_seconds=None,
    )
    ml_logger.close()
    return save_result
