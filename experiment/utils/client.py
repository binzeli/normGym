"""
LLM clients for making multimodal API calls.

Two providers: OpenRouterClient for remote API models (the default) and
HFClient for open-weight models run locally via Hugging Face.
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch
    HAS_TRANSFORMERS = True
except (ImportError, Exception):
    HAS_TRANSFORMERS = False
from experiment.utils.multimodal import build_openrouter_multimodal_content


class InsufficientCreditsError(RuntimeError):
    """The provider refused the call because the account has no credits left
    (OpenRouter HTTP 402). Never retryable and never recoverable within a run:
    callers must abort the experiment instead of falling back to 'wait',
    otherwise the run silently produces results with zero real inference."""


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Total number of attempts per request (1 = try once, no retries). Defaults to 1
# so a failed/empty response is not retried; override via the
# OPENROUTER_MAX_RETRY_ATTEMPTS env var. Values below 1 are clamped to 1.
OPENROUTER_MAX_RETRY_ATTEMPTS = max(1, int(os.getenv("OPENROUTER_MAX_RETRY_ATTEMPTS", "1")))
OPENROUTER_INITIAL_RETRY_DELAY_SECONDS = 1.0
OPENROUTER_MAX_RETRY_DELAY_SECONDS = 15.0

# Optional reasoning/thinking controls, read from the environment so the
# launcher can set them per model without touching the agent call sites.
# Thinking models (e.g. google/gemini-3.x) spend their reasoning tokens out of
# the same max_completion_tokens budget as the answer; with no cap a long
# think leaves nothing for the JSON and the call returns finish_reason=length
# with empty or truncated content. See https://openrouter.ai/docs/use-cases/reasoning-tokens
#   OPENROUTER_REASONING_EFFORT      = low | medium | high   (mapped per provider)
#   OPENROUTER_REASONING_MAX_TOKENS  = integer token budget for thinking
#   OPENROUTER_MAX_COMPLETION_TOKENS = overrides the caller's max_new_tokens /
#                                      max_completion_tokens for every request
# Unset (the default) sends no reasoning field, i.e. the provider's default.


def _openrouter_reasoning_config() -> Optional[Dict[str, Any]]:
    effort = (os.getenv("OPENROUTER_REASONING_EFFORT") or "").strip().lower()
    max_tokens = (os.getenv("OPENROUTER_REASONING_MAX_TOKENS") or "").strip()
    config: Dict[str, Any] = {}
    if effort:
        if effort not in {"low", "medium", "high"}:
            raise ValueError(
                f"OPENROUTER_REASONING_EFFORT must be low|medium|high, got {effort!r}"
            )
        config["effort"] = effort
    if max_tokens:
        config["max_tokens"] = int(max_tokens)
    return config or None


def _openrouter_max_completion_tokens_override() -> Optional[int]:
    value = (os.getenv("OPENROUTER_MAX_COMPLETION_TOKENS") or "").strip()
    return int(value) if value else None

_OPENROUTER_ALLOWED_KWARGS = {
    "temperature",
    "top_p",
    "seed",
    "stop",
    "response_format",
    "max_tokens",
    "max_completion_tokens",
}


DEEPSEEK_VL2_SMALL_MODEL = "deepseek-ai/deepseek-vl2-small"

# DeepSeek-VL2-Small's original Transformers 4.38 configuration leaves these
# values to defaults in DeepSeek's custom code. Current vLLM reconstructs that
# text configuration with newer Transformers defaults, so the values must be
# restored before its weights are created. They describe the published
# checkpoint; they are not experiment or sampling settings.
DEEPSEEK_VL2_SMALL_HF_OVERRIDES = {
    "architectures": ["DeepseekVLV2ForCausalLM"],
    "text_config": {
        "vocab_size": 102400,
        "kv_lora_rank": 512,
        "bos_token_id": 100000,
        "eos_token_id": 100001,
    },
}


def load_env_file():
    """Load environment variables from .env file if it exists."""
    # __file__ is experiment/utils/client.py
    # parent = experiment/utils/
    # parent.parent = experiment/
    # parent.parent.parent = normGym/
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


class OpenRouterClient:
    """
    Client for sending requests to OpenRouter's OpenAI-compatible chat API.
    Supports multimodal inputs (text + images) for vision-capable models.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ):
        load_env_file()

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY in the "
                "environment or .env file."
            )

        self.model = model or os.getenv("OPENROUTER_MODEL")
        if not self.model:
            raise ValueError(
                "OpenRouter model not provided. Set --model or OPENROUTER_MODEL. "
                "OpenRouter model slugs look like 'openai/...', 'anthropic/...', "
                "'google/...', or 'qwen/...'."
            )

        default_headers = self._build_default_headers(site_url, app_name)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=default_headers or None,
        )

        # Running totals for cost accounting. OpenRouter reports the actual
        # dollar cost of every billed call when we request usage accounting;
        # these accumulate across all calls made through this client instance.
        self.total_cost_usd = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        # When set (via ``enable_incremental_cost_report``), the running cost
        # report is flushed to disk after every billed call so a long run's
        # spend is visible while it is still in progress, not only at the end.
        self._cost_report_dir: Optional[str] = None
        self._cost_report_filename = "cost.json"

    @staticmethod
    def _build_default_headers(
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        referer = site_url or os.getenv("OPENROUTER_SITE_URL")
        title = app_name or os.getenv("OPENROUTER_APP_NAME")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        return headers

    @staticmethod
    def _build_generation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        generation_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in _OPENROUTER_ALLOWED_KWARGS and value is not None
        }
        if "max_new_tokens" in kwargs and "max_completion_tokens" not in generation_kwargs:
            generation_kwargs["max_completion_tokens"] = kwargs["max_new_tokens"]
        override = _openrouter_max_completion_tokens_override()
        if override is not None:
            generation_kwargs.pop("max_tokens", None)
            generation_kwargs["max_completion_tokens"] = override
        # Recorded here so failure diagnostics show it; stripped before the SDK
        # call because chat.completions.create() does not accept it directly.
        reasoning = _openrouter_reasoning_config()
        if reasoning is not None:
            generation_kwargs["reasoning"] = reasoning
        return generation_kwargs

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            try:
                dumped = usage.model_dump()
                return dumped if isinstance(dumped, dict) else {}
            except Exception:
                return {}
        return {}

    def _accumulate_usage(self, completion: Any) -> None:
        """Add one completion's billed usage to the running totals.

        OpenRouter returns a ``usage`` block on every completion when usage
        accounting is requested; ``cost`` is the actual charge in USD. A call is
        still billed even when its content comes back empty, so this is invoked
        for every completion received, before the choices/content checks.
        """
        usage = self._usage_to_dict(getattr(completion, "usage", None))
        if not usage:
            return
        self.total_calls += 1
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.total_cost_usd += float(cost)
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, (int, float)):
            self.total_prompt_tokens += int(prompt_tokens)
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, (int, float)):
            self.total_completion_tokens += int(completion_tokens)

        self._flush_incremental_cost_report()

    def enable_incremental_cost_report(
        self, output_dir: str, filename: str = "cost.json"
    ) -> None:
        """Flush the cost report to ``output_dir`` after every billed call.

        Call once after constructing the client to keep ``cost.json`` up to date
        throughout a long run instead of only writing it at the very end.
        """
        self._cost_report_dir = output_dir
        self._cost_report_filename = filename

    def _flush_incremental_cost_report(self) -> None:
        if not self._cost_report_dir:
            return
        try:
            self.write_cost_report(self._cost_report_dir, self._cost_report_filename)
        except OSError as exc:
            # A transient write failure must never abort an in-flight run; the
            # end-of-run report write is the backstop.
            print(f"[WARN] Failed to flush incremental cost report: {exc}")

    def cost_report(self) -> Dict[str, Any]:
        """Structured cost/usage totals for this client instance."""
        return {
            "provider": "openrouter",
            "model": self.model,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_calls": self.total_calls,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def cost_summary_line(self) -> str:
        """One-line human-readable spend summary for stdout/logs."""
        return (
            f"Total OpenRouter spend: ${self.total_cost_usd:.4f} "
            f"({self.total_prompt_tokens:,} prompt + "
            f"{self.total_completion_tokens:,} completion tokens "
            f"over {self.total_calls} call(s), model {self.model})"
        )

    def write_cost_report(self, output_dir: str, filename: str = "cost.json") -> Optional[str]:
        """Write the cost report as JSON into ``output_dir``; return its path."""
        if not output_dir:
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cost_report(), f, indent=2)
        return path

    @staticmethod
    def _extract_message_content(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            text_parts = []
            for part in message_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif hasattr(part, "text"):
                    text_parts.append(str(part.text))
            return "".join(text_parts)
        return "" if message_content is None else str(message_content)

    @staticmethod
    def _serialize_response(response: Any) -> str:
        if hasattr(response, "model_dump_json"):
            try:
                return response.model_dump_json(indent=2)
            except Exception:
                pass
        if hasattr(response, "model_dump"):
            try:
                return json.dumps(response.model_dump(), indent=2, default=str)
            except Exception:
                pass
        return repr(response)

    @staticmethod
    def _response_to_dict(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            try:
                dumped = response.model_dump()
                return dumped if isinstance(dumped, dict) else {}
            except Exception:
                return {}
        return {}

    @classmethod
    def _extract_provider_error(cls, response: Any) -> Dict[str, Any]:
        error = cls._response_to_dict(response).get("error")
        return error if isinstance(error, dict) else {}

    @staticmethod
    def _exception_status_code(exc: Exception) -> Optional[int]:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    @classmethod
    def _is_retryable_error(cls, error: Dict[str, Any]) -> bool:
        return error.get("code") == 429

    @classmethod
    def _retry_delay_seconds(
        cls,
        attempt_index: int,
        error_message: str = "",
    ) -> float:
        delay = min(
            OPENROUTER_INITIAL_RETRY_DELAY_SECONDS * (2 ** attempt_index),
            OPENROUTER_MAX_RETRY_DELAY_SECONDS,
        )
        match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s)", error_message, re.IGNORECASE)
        if match:
            retry_value = float(match.group(1))
            provider_delay = retry_value / 1000.0 if match.group(2).lower() == "ms" else retry_value
            delay = max(delay, provider_delay, OPENROUTER_INITIAL_RETRY_DELAY_SECONDS)
        return delay

    def _build_failure_diagnostics(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: Optional[str],
        generation_kwargs: Dict[str, Any],
        attempts: int,
        failure_reason: str,
        completion: Any = None,
        exception: Exception = None,
    ) -> Dict[str, Any]:
        diagnostics = {
            "provider": "openrouter",
            "model": self.model,
            "failure_reason": failure_reason,
            "image_count": len(image_paths),
            "prompt_chars": len(prompt or ""),
            "system_prompt_chars": len(system_prompt or ""),
            "generation_kwargs": generation_kwargs,
            "attempts": attempts,
        }
        if completion is not None:
            diagnostics["completion"] = self._serialize_response(completion)
        if exception is not None:
            diagnostics["exception"] = repr(exception)
            diagnostics["exception_status_code"] = self._exception_status_code(exception)
        return diagnostics

    def _raise_no_choices_error(
        self,
        diagnostics: Dict[str, Any],
    ) -> None:
        raise RuntimeError(
            "OpenRouter returned no usable completion content. The LLM "
            "inference failed, so this episode should be treated as invalid.\n"
            f"{json.dumps(diagnostics, indent=2, ensure_ascii=False)}"
        )

    def response(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        generation_kwargs = self._build_generation_kwargs(kwargs)
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": build_openrouter_multimodal_content(prompt, image_paths),
        })

        # OpenRouter-only fields (reasoning) travel in extra_body.
        sdk_kwargs = {k: v for k, v in generation_kwargs.items() if k != "reasoning"}
        extra_body: Dict[str, Any] = {"usage": {"include": True}}
        if "reasoning" in generation_kwargs:
            extra_body["reasoning"] = generation_kwargs["reasoning"]

        last_diagnostics = None
        for attempt_index in range(OPENROUTER_MAX_RETRY_ATTEMPTS):
            attempt_number = attempt_index + 1
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body=extra_body,
                    **sdk_kwargs,
                )
                self._accumulate_usage(completion)
            except Exception as exc:
                if self._exception_status_code(exc) == 402:
                    raise InsufficientCreditsError(
                        f"OpenRouter account has no credits left (HTTP 402) for "
                        f"model {self.model}; add credits at "
                        f"https://openrouter.ai/settings/credits. Original error: {exc}"
                    ) from exc
                if self._exception_status_code(exc) != 429:
                    raise

                last_diagnostics = self._build_failure_diagnostics(
                    image_paths=image_paths,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    generation_kwargs=generation_kwargs,
                    attempts=attempt_number,
                    failure_reason="http_429_exception",
                    exception=exc,
                )
                if attempt_number >= OPENROUTER_MAX_RETRY_ATTEMPTS:
                    self._raise_no_choices_error(last_diagnostics)

                delay = self._retry_delay_seconds(attempt_index, str(exc))
                print(
                    f"OpenRouter retry {attempt_number}/{OPENROUTER_MAX_RETRY_ATTEMPTS} "
                    f"for model {self.model}: HTTP 429. Sleeping {delay:.1f}s."
                )
                time.sleep(delay)
                continue

            if getattr(completion, "choices", None):
                content = self._extract_message_content(completion.choices[0].message.content)
                if content.strip():
                    return content

                last_diagnostics = self._build_failure_diagnostics(
                    image_paths=image_paths,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    generation_kwargs=generation_kwargs,
                    attempts=attempt_number,
                    failure_reason="empty_message_content",
                    completion=completion,
                )
                if attempt_number >= OPENROUTER_MAX_RETRY_ATTEMPTS:
                    self._raise_no_choices_error(last_diagnostics)

                delay = self._retry_delay_seconds(attempt_index)
                print(
                    f"OpenRouter retry {attempt_number}/{OPENROUTER_MAX_RETRY_ATTEMPTS} "
                    f"for model {self.model}: empty message content. "
                    f"Sleeping {delay:.1f}s."
                )
                time.sleep(delay)
                continue

            error = self._extract_provider_error(completion)
            if error.get("code") == 402:
                raise InsufficientCreditsError(
                    f"OpenRouter account has no credits left (provider error 402) "
                    f"for model {self.model}; add credits at "
                    f"https://openrouter.ai/settings/credits. "
                    f"Message: {error.get('message', '')}"
                )
            last_diagnostics = self._build_failure_diagnostics(
                image_paths=image_paths,
                prompt=prompt,
                system_prompt=system_prompt,
                generation_kwargs=generation_kwargs,
                attempts=attempt_number,
                failure_reason="missing_choices",
                completion=completion,
            )
            if not self._is_retryable_error(error):
                self._raise_no_choices_error(last_diagnostics)

            if attempt_number >= OPENROUTER_MAX_RETRY_ATTEMPTS:
                self._raise_no_choices_error(last_diagnostics)

            message = str(error.get("message", ""))
            delay = self._retry_delay_seconds(attempt_index, message)
            print(
                f"OpenRouter retry {attempt_number}/{OPENROUTER_MAX_RETRY_ATTEMPTS} "
                f"for model {self.model}: provider error 429: {message}. "
                f"Sleeping {delay:.1f}s."
            )
            time.sleep(delay)

        self._raise_no_choices_error(last_diagnostics or self._build_failure_diagnostics(
            image_paths=image_paths,
            prompt=prompt,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
            attempts=OPENROUTER_MAX_RETRY_ATTEMPTS,
            failure_reason="unknown_openrouter_failure",
        ))


class HFClient:
    """
    Client for running open-weight vision-language models locally via Hugging
    Face. Any checkpoint loadable by AutoModelForImageTextToText works (Qwen-VL,
    Gemma 3, LLaVA, InternVL HF ports, ...). Supports multimodal inputs
    (text + images).
    """

    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct"):
        """
        Initialize HFClient.

        Args:
            model: Hugging Face model name or local checkpoint path
                (e.g., "Qwen/Qwen3-VL-8B-Instruct", "google/gemma-3-4b-it")
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch are required for HFClient. "
                "Install with: pip install transformers torch"
            )
        # Load .env file if it exists
        load_env_file()

        self.model_name = model

        # Cluster runners can require the complete HF checkpoint to reside on
        # the allocated GPU.  Without this explicit placement, Accelerate's
        # automatic map may silently choose CPU execution while the Slurm GPU
        # remains idle.  Keep "auto" as the default for non-cluster callers.
        force_cuda = os.getenv("NORMGYM_HF_FORCE_CUDA") == "1"
        if force_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "NORMGYM_HF_FORCE_CUDA=1, but CUDA is unavailable to PyTorch"
            )
        device_map: Any = {"": "cuda:0"} if force_cuda else "auto"

        # Load model and processor. dtype "auto" follows each checkpoint's
        # native dtype (bf16 for most current VLMs); forcing fp16 overflows on
        # some families (e.g. Gemma 3).
        self.model = AutoModelForImageTextToText.from_pretrained(
            model,
            torch_dtype="auto",
            device_map=device_map,
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(model)

        parameter_device_types = sorted({
            parameter.device.type for parameter in self.model.parameters()
        })
        print(
            "[HF MODEL PLACEMENT] "
            f"model={model}, force_cuda={force_cuda}, "
            f"parameter_device_types={parameter_device_types}, "
            f"hf_device_map={getattr(self.model, 'hf_device_map', None)}",
            flush=True,
        )
        if force_cuda and parameter_device_types != ["cuda"]:
            raise RuntimeError(
                "The allocated-GPU run did not place every HF model parameter "
                f"on CUDA: {parameter_device_types}"
            )

        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        model_name = model.lower()
        self._is_mllama = (
            model_type == "mllama" or "llama-3.2-11b-vision" in model_name
        )
        self._is_glm_vision = (
            model_type == "glm4v" or "glm-4.6v-flash" in model_name
        )

    @staticmethod
    def _generation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply only settings explicitly required by the experiment.

        A positive temperature has no effect under greedy decoding, so sampling
        is enabled when the caller supplies one. top_k, top_p, repetition
        penalty, and every other omitted value remain absent and are therefore
        inherited from the checkpoint's generation_config.json.
        """
        generation = {
            key: value for key, value in kwargs.items() if value is not None
        }
        if "temperature" in generation:
            temperature = float(generation["temperature"])
            if temperature <= 0:
                raise ValueError("HF generation temperature must be positive")
            generation["temperature"] = temperature
            generation.setdefault("do_sample", True)
        return generation

    def _messages(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Build the checkpoint-native multimodal conversation."""
        messages: List[Dict[str, Any]] = []
        user_prompt = prompt

        # Mllama's vision template rejects a separate system role whenever the
        # conversation contains an image. Keep the complete instruction by
        # placing it at the start of the same user turn. Text-only calls (the
        # final memory summary) retain the native system role.
        if self._is_mllama and image_paths and system_prompt:
            user_prompt = f"{system_prompt}\n\n{prompt}"
        elif system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        content = [
            {"type": "image", "image": image_path}
            for image_path in image_paths
        ]
        content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _effective_generation_value(
        self, generation: Dict[str, Any], name: str
    ) -> Any:
        if name in generation:
            return generation[name]
        config = getattr(self.model, "generation_config", None)
        value = getattr(config, name, None)
        # Transformers 5 leaves unspecified fields as None until generate()
        # fills them from its defaults. Resolve them for logging only; do not
        # introduce sampling overrides for any model.
        defaults = getattr(config, "_get_default_generation_params", None)
        if value is None and callable(defaults):
            value = defaults().get(name)
        return value
    
    def response(self, 
                   image_paths: List[str],
                   prompt: str,
                   system_prompt: str = None,
                   max_new_tokens: int = 2048,
                   **kwargs) -> str:
        """
        Send a series of images with a prompt to the local model.

        Args:
            image_paths: List of paths to images to send
            prompt: Text prompt to send with the images
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters for generation

        Returns:
            LLM response text
        """
        
        existing_image_paths = [
            image_path for image_path in image_paths if os.path.exists(image_path)
        ]
        messages = self._messages(existing_image_paths, prompt, system_prompt)
        generation_kwargs = self._generation_kwargs(kwargs)

        print("=" * 80)
        print("[HF INPUT] images:", existing_image_paths)
        if system_prompt:
            print("[HF INPUT] system prompt:\n" + system_prompt)
        print("[HF INPUT] user prompt:\n" + prompt)
        print("-" * 80, flush=True)

        # Apply chat template
        chat_template_kwargs = {}
        if self._is_glm_vision:
            # Official GLM template switch: emit a direct answer instead of a
            # thinking trace, without changing the experiment prompt/response
            # contract or adding a second generation stage.
            chat_template_kwargs["enable_thinking"] = False

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **chat_template_kwargs,
        )
        inputs.pop("token_type_ids", None)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        
        print(
            "[HF GENERATION] "
            f"model={self.model_name}, "
            f"temperature={self._effective_generation_value(generation_kwargs, 'temperature')}, "
            f"do_sample={self._effective_generation_value(generation_kwargs, 'do_sample')}, "
            f"top_k={self._effective_generation_value(generation_kwargs, 'top_k')}, "
            f"top_p={self._effective_generation_value(generation_kwargs, 'top_p')}, "
            f"repetition_penalty={self._effective_generation_value(generation_kwargs, 'repetition_penalty')}, "
            f"max_new_tokens={max_new_tokens}",
            flush=True,
        )

        # Trim generated ids
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # The Instruct checkpoints do not emit reasoning, but a stray <think> block
        # would otherwise reach the action parser.
        answer = re.sub(r"<think>.*?</think>", "", output_text[0], flags=re.DOTALL).strip()

        print("[HF OUTPUT]\n" + answer)
        print("=" * 80, flush=True)

        return answer


def _is_deepseek_vl2_small(model: Optional[str]) -> bool:
    if not isinstance(model, str):
        return False
    return "deepseek-vl2-small" in model.replace("\\", "/").lower()


def _positive_int_from_env(
    name: str, default: int, maximum: Optional[int] = None
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


def _fraction_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


class DeepSeekVL2Client:
    """Dedicated vLLM path for DeepSeek-VL2-Small.

    The checkpoint is not registered with AutoModelForImageTextToText in the
    shared Transformers stack. vLLM has a maintained native implementation and
    keeps DeepSeek's older dependency requirements out of the environment used
    by Qwen, Llama, and GLM.
    """

    def __init__(self, model: str = DEEPSEEK_VL2_SMALL_MODEL):
        if not isinstance(model, str):
            raise TypeError(
                "model must be a Hugging Face id or path, got "
                f"{type(model).__name__}"
            )
        self.model = model
        load_env_file()

        self.max_model_len = _positive_int_from_env(
            "DEEPSEEK_VL2_MAX_MODEL_LEN", 4096, maximum=4096
        )
        self.max_images = _positive_int_from_env("DEEPSEEK_VL2_MAX_IMAGES", 1)

        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "DeepSeek-VL2-Small uses vLLM. Install vLLM in a dedicated "
                "Linux GPU environment (see experiment/train/README.md)."
            ) from exc

        self.engine = LLM(
            model=self.model,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            max_num_seqs=1,
            tensor_parallel_size=_positive_int_from_env(
                "DEEPSEEK_VL2_TENSOR_PARALLEL_SIZE", 1
            ),
            gpu_memory_utilization=_fraction_from_env(
                "DEEPSEEK_VL2_GPU_MEMORY_UTILIZATION", 0.92
            ),
            generation_config="auto",
            hf_overrides=DEEPSEEK_VL2_SMALL_HF_OVERRIDES,
            limit_mm_per_prompt={"image": self.max_images},
        )

    @staticmethod
    def _prompt(prompt: str, system_prompt: Optional[str], image_count: int) -> str:
        instruction = prompt
        if system_prompt:
            instruction = f"{system_prompt}\n\n{prompt}"

        if image_count == 1:
            image_prefix = "<image>\n"
        elif image_count > 1:
            image_prefix = "".join(
                f"image_{index}:<image>\n"
                for index in range(1, image_count + 1)
            )
        else:
            image_prefix = ""
        return f"<|User|>: {image_prefix}{instruction}\n\n<|Assistant|>:"

    def response(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        **kwargs,
    ) -> str:
        from PIL import Image

        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Multimodal input image(s) not found: {missing}"
            )
        if len(image_paths) > self.max_images:
            raise ValueError(
                f"DeepSeek-VL2 client allows {self.max_images} image(s), "
                f"got {len(image_paths)}."
            )

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())

        request: Dict[str, Any] = {
            "prompt": self._prompt(prompt, system_prompt, len(images))
        }
        if images:
            request["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }

        # Start with vLLM's checkpoint-derived defaults, then change only the
        # experiment-wide temperature/output ceiling (plus explicit caller
        # overrides, if ever supplied). In particular, this does not impose a
        # shared top_k or top_p on DeepSeek.
        sampling = self.engine.get_default_sampling_params()
        temperature = float(kwargs.get("temperature", 0.7))
        if temperature <= 0:
            raise ValueError("DeepSeek generation temperature must be positive")
        sampling.temperature = temperature
        sampling.max_tokens = max(1, int(max_new_tokens))
        for key in ("seed", "top_k", "top_p", "repetition_penalty", "stop"):
            value = kwargs.get(key)
            if value is not None:
                setattr(sampling, key, value)

        print("=" * 80)
        print("[DEEPSEEK INPUT] images:", image_paths)
        if system_prompt:
            print("[DEEPSEEK INPUT] system prompt:\n" + system_prompt)
        print("[DEEPSEEK INPUT] user prompt:\n" + prompt)
        print(
            "[DEEPSEEK GENERATION] "
            f"temperature={sampling.temperature}, "
            f"top_k={sampling.top_k}, top_p={sampling.top_p}, "
            f"repetition_penalty={sampling.repetition_penalty}",
            flush=True,
        )

        try:
            outputs = self.engine.generate(request, sampling, use_tqdm=False)
        finally:
            for image in images:
                image.close()

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("DeepSeek-VL2 returned no generation candidates.")
        answer = outputs[0].outputs[0].text.strip()
        print("[DEEPSEEK OUTPUT]\n" + answer)
        print("=" * 80, flush=True)
        return answer


def create_llm_client(provider: str = "openrouter", model: Optional[str] = None):
    """
    Create an LLM client for active runners.

    'openrouter' (the default) calls remote API models; 'hf' runs an
    open-weight model locally via Hugging Face.
    """
    provider = (provider or "openrouter").lower()
    if provider == "openrouter":
        return OpenRouterClient(model=model)
    if provider == "hf":
        if _is_deepseek_vl2_small(model):
            return DeepSeekVL2Client(model=model)
        return HFClient(model=model) if model else HFClient()
    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Expected one of: "
        "openrouter, hf."
    )

