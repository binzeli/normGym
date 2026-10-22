"""Multi-process rollout driver backed by a vLLM server.

Why processes and not threads: the pipeline seeds and draws from Python's
*global* ``random`` module while building an episode's agents
(``run_interactive_stage`` -> ``build_initial_agents`` -> ``random.choice`` /
``random.randint``), so concurrent threads would corrupt each other's episode
setup and the rollouts of a group would no longer share one starting state.
Separate processes each keep their own RNG.

Generation is delegated to a vLLM server, so workers hold no model weights and
use no GPU — the pool can be far larger than the GPU count, and the server's
continuous batching is where the speedup comes from. Worker count is therefore
bounded by CPU (env stepping + frame rendering), not by VRAM.

Weights are kept in sync by saving the training LoRA to disk after each update
and hot-swapping it on the server (``sync_lora``); the server must be started
with ``--enable-lora`` and ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``.
"""
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Set in each worker by _init_worker; never touched in the parent.
_CLIENT = None


def _init_worker(rollout_url: str, model_name: str, temperature: float,
                 max_new_tokens: int, top_p: float) -> None:
    """Per-worker one-time setup: no GPU, headless rendering, one HTTP client."""
    # Workers only render frames and speak HTTP. Hiding the GPUs keeps them
    # from allocating VRAM the trainer needs. torch's CUDA init is lazy and
    # nothing in the rollout path touches it, so setting this here is enough.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        pass

    from experiment.train.policy_client import OpenAIBackend, RecordingClient

    global _CLIENT
    _CLIENT = RecordingClient(
        OpenAIBackend(rollout_url, model_name, top_p=top_p),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def _run_task(task: Dict[str, Any]):
    """Run one episode. Returns None on failure so one bad episode can't kill
    the update — the caller filters them out and logs the count."""
    from experiment.train.rollout import run_episode
    try:
        return run_episode(
            _CLIENT,
            norm_type=task['norm'],
            world_type=task['world'],
            seed=task['seed'],
            workdir=task['workdir'],
            capacity=task['capacity'],
            max_steps=task['max_steps'],
            gamma=task['gamma'],
            format_bonus=task['format_bonus'],
            violation_penalty_scale=task.get('violation_penalty_scale', 1.0),
            violation_budget=task.get('violation_budget', 0.0),
            # collision_budget was silently dropped here until 08-24: pooled
            # rollouts (every vLLM run) trained against the flat -5 while the
            # trainer logged/assumed the geometric budget.
            collision_budget=task.get('collision_budget', 0.0),
            progress_shaping_coef=task.get('progress_shaping_coef', 0.0),
            parse_penalty=task.get('parse_penalty', 0.0),
            truncation_penalty=task.get('truncation_penalty', 0.0),
            subject_kind=task.get('subject_kind', 'memory_llm'),
        )
    except Exception:
        sys.stderr.write(
            f"[rollout-worker] episode failed "
            f"({task['norm']} seed={task['seed']}):\n{traceback.format_exc()}\n"
        )
        return None


class RolloutPool:
    """A persistent pool of episode workers.

    Uses the 'spawn' start method: the parent process holds an initialized CUDA
    context for training, and forking that is unsafe.
    """

    def __init__(self, num_workers: int, rollout_url: str, model_name: str,
                 temperature: float, max_new_tokens: int, top_p: float):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.num_workers = num_workers
        self.pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(rollout_url, model_name, temperature, max_new_tokens, top_p),
        )

    def run_batch(self, tasks: List[Dict[str, Any]]) -> List[Optional[Any]]:
        """Run every task concurrently; results come back in task order."""
        return self.pool.map(_run_task, tasks, chunksize=1)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


# ---------------------------------------------------------------------------
# vLLM weight sync
# ---------------------------------------------------------------------------

def sync_lora(rollout_url: str, adapter_path: str, lora_name: str,
              timeout: int = 600, attempts: int = 4, retry_delay: int = 30) -> None:
    """Hot-swap the server's LoRA to the adapter now on disk.

    Unload-then-load under the same name, so rollout workers can keep asking
    for a fixed model name across updates. The unload is best-effort: on the
    first update the name may not be registered yet. The whole exchange is
    retried a few times because a node's own vLLM port has been seen to time
    out for minutes (job 338142 died at update 45 on a 400 "already loaded"
    right after such a timeout, 2026-09-11).

    NEVER send ``load_inplace`` on a normal load: vLLM stores that flag on the
    served LoRARequest and the worker then re-reads the adapter from disk on
    EVERY model step (vllm/lora/worker_manager.py, ``or lora_request.load_inplace``),
    which collapsed generation to ~6 tokens/s and stalled jobs 378147/378138
    (2026-09-11). It is used only as a last resort when the unload keeps
    failing and the server still reports the adapter as loaded.
    """
    import time
    import requests

    base = rollout_url.rstrip('/')
    payload = {'lora_name': lora_name, 'lora_path': os.path.abspath(adapter_path)}
    last_error = 'no attempt made'
    for attempt in range(1, attempts + 1):
        try:
            requests.post(f'{base}/unload_lora_adapter',
                          json={'lora_name': lora_name}, timeout=timeout)
        except Exception as exc:
            print(f'[vllm-sync] unload skipped ({exc})', flush=True)
        body = dict(payload)
        if attempt == attempts and 'already been loaded' in last_error:
            body['load_inplace'] = True   # last resort only, see docstring
            print('[vllm-sync] WARNING: falling back to load_inplace (slower generation until the next sync)', flush=True)
        try:
            response = requests.post(f'{base}/load_lora_adapter', json=body, timeout=timeout)
        except Exception as exc:
            last_error = f'load request failed: {exc}'
        else:
            if response.ok:
                if attempt > 1:
                    print(f'[vllm-sync] load succeeded on attempt {attempt}', flush=True)
                return
            last_error = f'load_lora_adapter returned {response.status_code}: {response.text[:500]}'
            # "already been loaded" means the unload above never reached the server -> retry;
            # any other 4xx is a real request error -> no retry
            if response.status_code != 400 or 'already been loaded' not in response.text:
                if 400 <= response.status_code < 500:
                    raise RuntimeError(f'[vllm-sync] {last_error}')
        if attempt < attempts:
            print(f'[vllm-sync] attempt {attempt}/{attempts} failed ({last_error}); '
                  f'retrying in {retry_delay}s', flush=True)
            time.sleep(retry_delay)
    raise RuntimeError(f'[vllm-sync] gave up after {attempts} attempts: {last_error}')


def wait_for_server(rollout_url: str, attempts: int = 120, delay: int = 10) -> None:
    """Block until the vLLM server answers, or give up with a clear error."""
    import time

    import requests

    health = rollout_url.rstrip('/').rsplit('/v1', 1)[0] + '/health'
    for attempt in range(1, attempts + 1):
        try:
            if requests.get(health, timeout=5).ok:
                print(f'[vllm-sync] server ready after {attempt} check(s)', flush=True)
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(f'[vllm-sync] server never became ready at {health}')
