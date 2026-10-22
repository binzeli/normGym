"""Model loading and prompt/response encoding shared by SFT and PPO.

Message building mirrors experiment/utils/client.py::HFClient exactly, so a
checkpoint trained here sees the same token stream at evaluation time when it
is served back through ``run_experiment.py --provider hf``.
"""
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from experiment.train.config import BASE_MODEL, LORA_DEFAULTS


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def build_messages(system_prompt: Optional[str], user_prompt: str,
                   image_paths: List[str]) -> List[Dict]:
    """The exact message structure HFClient.response builds."""
    messages = []
    if system_prompt:
        messages.append({
            'role': 'system',
            'content': [{'type': 'text', 'text': system_prompt}],
        })
    content = [
        {'type': 'image', 'image': path}
        for path in image_paths if path and os.path.exists(path)
    ]
    content.append({'type': 'text', 'text': user_prompt})
    messages.append({'role': 'user', 'content': content})
    return messages


def with_assistant(messages: List[Dict], response_text: str) -> List[Dict]:
    return messages + [{
        'role': 'assistant',
        'content': [{'type': 'text', 'text': response_text}],
    }]


# ---------------------------------------------------------------------------
# Model / processor
# ---------------------------------------------------------------------------

def load_processor(model_name_or_path: str = BASE_MODEL):
    return AutoProcessor.from_pretrained(model_name_or_path)


def load_model(model_name_or_path: str = BASE_MODEL,
               adapter_path: Optional[str] = None,
               train_lora: bool = False,
               lora_kwargs: Optional[dict] = None,
               dtype=torch.bfloat16):
    """Load a VLM (Qwen3-VL, Ministral 3, ...), optionally with an existing
    LoRA adapter or a fresh one.

    - adapter_path: resume/continue from a saved PEFT adapter (trainable when
      train_lora is also set, frozen otherwise).
    - train_lora (no adapter_path): attach a fresh LoRA on the language model.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation='sdpa',
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=train_lora
        )
    elif train_lora:
        from peft import LoraConfig, get_peft_model
        kwargs = dict(LORA_DEFAULTS)
        kwargs.update(lora_kwargs or {})
        target_modules = kwargs['target_modules']
        if (isinstance(target_modules, (list, tuple))
                and model.config.model_type != 'qwen3_vl'):
            # Qwen3-VL's vision blocks use qkv/linear_fc*, so the bare names
            # only hit the language model. Pixtral (Ministral 3) reuses
            # q_proj/.../down_proj in its vision tower — restrict the match
            # to the language model so the vision tower stays frozen.
            target_modules = (r'.*language_model.*\.('
                              + '|'.join(target_modules) + ')')
        lora_config = LoraConfig(
            r=kwargs['r'],
            lora_alpha=kwargs['alpha'],
            lora_dropout=kwargs['dropout'],
            target_modules=target_modules,
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


def save_merged(model, processor, out_dir: str) -> str:
    """Merge LoRA into the base weights and save a standalone checkpoint.

    The merged directory is loadable by experiment/utils/client.py::HFClient
    (``run_experiment.py --provider hf --model <out_dir>``), which is how
    checkpoints are evaluated.
    """
    os.makedirs(out_dir, exist_ok=True)
    merged = model
    if hasattr(model, 'merge_and_unload'):
        merged = model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Encoding for teacher forcing
# ---------------------------------------------------------------------------

def encode_prompt(processor, messages: List[Dict]):
    """Tokenize prompt-side messages with the generation prompt appended."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )
    inputs.pop('token_type_ids', None)
    return inputs


def encode_full(processor, messages: List[Dict], response_text: str):
    """Tokenize prompt + assistant response, with labels masked to the response.

    The prompt token sequence (with ``add_generation_prompt=True``) is a prefix
    of the full sequence under the Qwen chat template, so masking the first
    ``prompt_len`` labels supervises exactly the assistant's tokens (including
    the closing end-of-turn token).
    """
    prompt_inputs = encode_prompt(processor, messages)
    full = processor.apply_chat_template(
        with_assistant(messages, response_text),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors='pt',
    )
    full.pop('token_type_ids', None)

    prompt_len = prompt_inputs['input_ids'].shape[1]
    labels = full['input_ids'].clone()
    labels[:, :prompt_len] = -100
    full['labels'] = labels
    return full


def collate_turns(encoded_list):
    """Right-pad a list of ``encode_full`` outputs (batch 1 each) into one batch.

    input_ids/attention_mask/labels are padded to the longest sequence (pad id
    0 is inert: attention_mask zeros it and labels stay -100). Vision tensors
    are concatenated along dim 0 — Qwen-VL models consume image patches in the
    order their image tokens appear across the batch, which right padding
    preserves. (Mistral3/Pixtral pixel_values are [n_img, C, H, W] padded per
    call, so this needs equal frame sizes — true for env frames.)
    """
    max_len = max(e['input_ids'].shape[1] for e in encoded_list)

    def pad(t, value):
        if t.shape[1] == max_len:
            return t
        return torch.cat(
            [t, t.new_full((1, max_len - t.shape[1]), value)], dim=1)

    batch = {
        'input_ids': torch.cat([pad(e['input_ids'], 0) for e in encoded_list]),
        'attention_mask': torch.cat(
            [pad(e['attention_mask'], 0) for e in encoded_list]),
        'labels': torch.cat([pad(e['labels'], -100) for e in encoded_list]),
    }
    for key in ('pixel_values', 'image_grid_thw', 'image_sizes'):
        if key in encoded_list[0]:
            batch[key] = torch.cat([e[key] for e in encoded_list], dim=0)
    return batch


def token_batches(encoded_turns, budget):
    """Greedy length-sorted micro-batches under ``max_len * batch <= budget``.

    Sorting by length keeps padding waste small; the cost model uses the
    padded size (batch * longest member), which is what the forward pass
    actually pays. ``budget <= 0`` degrades to one turn per batch. Batch
    composition only changes how work is grouped — the summed loss (and so
    the gradient) is identical to per-turn processing.
    """
    order = sorted(range(len(encoded_turns)),
                   key=lambda i: encoded_turns[i][1]['input_ids'].shape[1])
    batches, cur, cur_max = [], [], 0
    for i in order:
        length = encoded_turns[i][1]['input_ids'].shape[1]
        if cur and (budget <= 0
                    or max(cur_max, length) * (len(cur) + 1) > budget):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, length)
    if cur:
        batches.append(cur)
    return batches


def response_logprobs(model, batch, device) -> torch.Tensor:
    """Per-sample summed log-probability of the supervised (response) tokens.

    ``batch`` is a dict from ``collate_turns`` (or ``encode_full`` for batch
    size 1). Returns a tensor of shape [batch] connected to the graph.
    """
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    labels = batch.pop('labels')
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    picked = torch.gather(
        log_probs, 2, targets.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    return (picked * mask).sum(dim=1)


def response_logprob(model, batch, device) -> torch.Tensor:
    """Summed log-probability of the supervised (response) tokens of one sample.

    ``batch`` is the dict from ``encode_full`` (batch size 1). Returns a scalar
    tensor connected to the graph, so it can be used inside the policy loss or
    wrapped in ``torch.no_grad()`` for old/reference log-probs.
    """
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    labels = batch.pop('labels')
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    picked = torch.gather(
        log_probs, 2, targets.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    return (picked * mask).sum()


def response_logprobs_and_entropies(model, batch, device):
    """One forward pass -> (summed response log-probs [B], summed
    response-token entropies [B]).

    The critic-free twin of ``response_logprobs_and_values`` for trainers
    whose baseline comes from sampling (RLOO/GRPO) instead of a value head —
    no hidden states are materialised. ``batch`` comes from
    ``collate_turns``/``encode_full``.
    """
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    labels = batch.pop('labels')
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    picked = torch.gather(
        log_probs, 2, targets.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    logps = (picked * mask).sum(dim=1)

    # Entropy restricted to the masked (response) positions before
    # exponentiating — exp over the full [B, T, V] would double the
    # forward's largest allocation (same trick as the value-head variant).
    sel = log_probs[mask]                                    # [N, V]
    token_entropy = -(sel.exp() * sel).sum(dim=-1)           # [N]
    rows = mask.nonzero(as_tuple=True)[0]                    # [N] -> batch row
    entropies = torch.zeros(
        logits.shape[0], dtype=token_entropy.dtype,
        device=token_entropy.device).index_add(0, rows, token_entropy)
    return logps, entropies


# ---------------------------------------------------------------------------
# Value head (PPO critic)
# ---------------------------------------------------------------------------

def text_hidden_size(model) -> int:
    """Hidden size of the language model (Qwen3-VL keeps it in text_config)."""
    config = model.config
    for candidate in (getattr(config, 'hidden_size', None),
                      getattr(getattr(config, 'text_config', None),
                              'hidden_size', None)):
        if candidate:
            return int(candidate)
    raise ValueError('could not determine hidden size from model config')


class ValueHead(torch.nn.Module):
    """Small MLP mapping one hidden state to a scalar state value.

    Kept in float32 (the backbone is bf16) for a well-conditioned regression
    target; saved separately from the LoRA adapter as ``value_head.pt`` in the
    checkpoint directory (see ``save_value_head`` / ``load_value_head``).
    """

    def __init__(self, hidden_size: int, inner: int = 1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, inner),
            torch.nn.GELU(),
            torch.nn.Linear(inner, 1),
        )
        # Start near zero so early PPO updates are dominated by the policy
        # gradient and the critic grows into the return scale gradually.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden.float()).squeeze(-1)


VALUE_HEAD_FILE = 'value_head.pt'


def save_value_head(value_head: ValueHead, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, VALUE_HEAD_FILE)
    torch.save(value_head.state_dict(), path)
    return path


def load_value_head(value_head: ValueHead, adapter_dir: str) -> bool:
    """Load ``value_head.pt`` from an adapter dir if present; returns success."""
    path = os.path.join(adapter_dir, VALUE_HEAD_FILE)
    if not os.path.isfile(path):
        return False
    value_head.load_state_dict(torch.load(path, map_location='cpu'))
    return True


def response_logprobs_and_values(model, value_head, batch, device,
                                 detach_value_features: bool = False):
    """One forward pass -> (summed response log-probs [B], state values [B],
    summed response-token entropies [B]).

    The state value of a turn is read at the last *prompt* position (the
    position whose next-token prediction is the first response token), i.e.
    the hidden state that summarises "everything the policy saw before
    answering". ``batch`` comes from ``collate_turns``/``encode_full``.

    ``detach_value_features=True`` stops critic gradients from reaching the
    shared LoRA trunk (pure value-head fitting); the default lets PPO's
    value loss shape the trunk as in the standard shared-backbone setup.
    """
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    labels = batch.pop('labels')
    outputs = model(**batch, output_hidden_states=True)

    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    picked = torch.gather(
        log_probs, 2, targets.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    logps = (picked * mask).sum(dim=1)

    # Per-turn summed entropy of the response-token distributions, for PPO's
    # entropy bonus (graph-connected). Restricted to the masked (response)
    # positions before exponentiating — exp over the full [B, T, V] would
    # double the forward's largest allocation.
    sel = log_probs[mask]                                    # [N, V]
    token_entropy = -(sel.exp() * sel).sum(dim=-1)           # [N]
    rows = mask.nonzero(as_tuple=True)[0]                    # [N] -> batch row
    entropies = torch.zeros(
        logits.shape[0], dtype=token_entropy.dtype,
        device=token_entropy.device).index_add(0, rows, token_entropy)

    # First supervised label index per row == prompt length; the state is the
    # hidden state one position earlier.
    first_response = (labels != -100).float().argmax(dim=1)
    state_pos = (first_response - 1).clamp(min=0)
    last_hidden = outputs.hidden_states[-1]
    features = last_hidden[torch.arange(last_hidden.shape[0], device=device),
                           state_pos]
    if detach_value_features:
        features = features.detach()
    values = value_head(features)
    return logps, values, entropies
