"""LLM-client shims that let a local policy play the pipeline's memory_llm subject.

``RecordingClient`` implements the same ``response(image_paths, prompt,
system_prompt, ...)`` protocol as experiment/utils/client.py, so it can be
handed to ``build_initial_agents(..., subject_kind='memory_llm', client=...)``
unchanged. Every call is recorded as a ``Turn`` so the RL trainer can later
recompute log-probabilities of the sampled responses.

Backends:
- HFBackend    : in-process transformers generation (training + pilots; the
                 same model object is later updated by the optimizer).
- OpenAIBackend: any OpenAI-compatible server (e.g. ``vllm serve``) for fast
                 rollouts. Keeping the served weights in sync with the training
                 adapter between updates is the caller's responsibility.
"""
import base64
import copy
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from experiment.train.modeling import build_messages

_STEP_RE = re.compile(r'Current step:\s*(\d+)\.')
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


@dataclass
class Turn:
    """One policy call: everything needed to replay it under teacher forcing."""
    system_prompt: Optional[str]
    user_prompt: str
    image_paths: List[str]
    response: str
    step: Optional[int] = None          # env step parsed from the user prompt
    reward_to_go: float = 0.0           # filled in by rollout.attach_rewards
    advantage: float = 0.0              # filled in by the RL trainer

    def messages(self):
        return build_messages(self.system_prompt, self.user_prompt,
                              self.image_paths)


class HFBackend:
    """Generation backend over an in-process transformers model.

    Mirrors HFClient.response: chat template, trimmed generation, <think>
    stripping — so rollout-time behavior matches evaluation-time behavior.
    """

    def __init__(self, model, processor, top_p: float = 0.95):
        self.model = model
        self.processor = processor
        self.top_p = top_p

    def generate(self, messages, max_new_tokens: int, temperature: float) -> str:
        import torch
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs.pop('token_type_ids', None)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=self.top_p,
                    # The training loop sets config.use_cache=False for
                    # gradient checkpointing; generation must still use the KV
                    # cache or rollouts become quadratically slow.
                    use_cache=True,
                )
        finally:
            if was_training:
                self.model.train()

        trimmed = generated[0][inputs['input_ids'].shape[1]:]
        text = self.processor.decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return _THINK_RE.sub('', text).strip()


class OpenAIBackend:
    """Generation backend over an OpenAI-compatible HTTP endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = 'EMPTY',
                 top_p: float = 0.95, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.top_p = top_p
        self.timeout = timeout

    @staticmethod
    def _to_openai_content(content):
        converted = []
        for part in content:
            if part.get('type') == 'image':
                with open(part['image'], 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('ascii')
                converted.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{b64}'},
                })
            else:
                converted.append({'type': 'text', 'text': part['text']})
        return converted

    def generate(self, messages, max_new_tokens: int, temperature: float) -> str:
        import requests
        payload = {
            'model': self.model,
            'messages': [
                {'role': m['role'], 'content': self._to_openai_content(m['content'])}
                for m in messages
            ],
            'max_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': self.top_p,
        }
        response = requests.post(
            f'{self.base_url}/chat/completions',
            json=payload,
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=self.timeout,
        )
        response.raise_for_status()
        text = response.json()['choices'][0]['message']['content'] or ''
        return _THINK_RE.sub('', text).strip()


@dataclass
class RecordingClient:
    """Client handed to the pipeline; forwards to a backend and records turns.

    Image paths are recorded as-is: rollout workdirs must therefore stay on
    disk until the RL update that consumes them has finished.
    """
    backend: object
    temperature: float = 0.7
    max_new_tokens: int = 1024
    turns: List[Turn] = field(default_factory=list)

    def response(self, image_paths, prompt, system_prompt=None,
                 max_new_tokens=None, temperature=None, **kwargs) -> str:
        # The agent passes its own generation settings (4048 tokens, temp 0.7);
        # training caps/overrides them so rollouts stay affordable and match
        # the sampling regime the trainer assumes.
        messages = build_messages(system_prompt, prompt, list(image_paths))
        text = self.backend.generate(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        match = _STEP_RE.search(prompt)
        self.turns.append(Turn(
            system_prompt=system_prompt,
            user_prompt=prompt,
            image_paths=[p for p in image_paths if p and os.path.exists(p)],
            response=text,
            step=int(match.group(1)) if match else None,
        ))
        return text

    def reset(self):
        self.turns = []

    def snapshot(self) -> List[Turn]:
        return copy.deepcopy(self.turns)
