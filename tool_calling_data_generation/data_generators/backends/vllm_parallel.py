"""
Data-parallel vLLM generation backend.

Spawns one independent vLLM engine per GPU (each in its own OS process),
shards the prompt batch across them, and merges the outputs back in the
original order.  For small dense models (Gemma 4 E2B, Qwen3-0.6B, Granite,
...) this gives near-linear throughput scaling with the number of GPUs.

Why not ``LLM(data_parallel_size=N)``?
    vLLM 0.22.x rejects ``data_parallel_size > 1`` in a single process
    (ValueError) unless you use the ``external_launcher`` executor with one
    OS process per DP rank, and that offline DP path is really aimed at MoE
    models — vLLM itself raises *"Offline data parallel mode is not
    supported/useful for dense models"* in the env-var fallback path (see
    ``vllm/config/parallel.py``).  For dense models the vLLM docs explicitly
    recommend independent instances:

        docs.vllm.ai/en/latest/serving/data_parallel_deployment/
        (Data Parallel Deployment → External Load Balancing):
        "This can already be done trivially for non-MoE models, since each
         deployed server is fully independent."

    Independent engines also have zero cross-engine communication, so there
    is no ZMQ/NCCL coordinator and no load-balancing overhead, and one
    crashed engine does not take the whole generation down.

How it works
------------
* ``initialize_backend()`` spawns ``num_gpus`` worker processes (spawn
  context).  Each worker pins itself to its GPU(s) via
  ``CUDA_VISIBLE_DEVICES`` *before* touching CUDA, then builds a plain
  ``vLLM(..., tensor_parallel_size=...)`` engine.
* ``generate()`` splits ``prompts`` into one contiguous chunk per worker,
  fans the chunks out over per-worker task queues, and collects results
  from a shared result queue.  ``LLM.generate()`` returns outputs in input
  order (vLLM sorts finished requests by request id), so the chunk results
  are concatenated in rank order to reconstruct the original ordering.
* Progress: each worker renders its own ``tqdm`` bar (``position=rank``)
  through vLLM's ``use_tqdm`` callable hook, so the bars stack vertically
  and update independently per GPU.  The bars share a cross-process write
  lock and a fixed bounded width, so they cannot interleave or wrap (which
  previously left frozen "ghost" bars on screen).  They auto-disable when
  stderr is not a TTY (logs / notebooks) or when ``VLLM_PARALLEL_TQDM=0``
  is set.
* ``unload_backend()`` asks every worker to shut down (vLLM
  ``sleep(level=2)`` to drop weights + KV cache, then exit) and joins /
  terminates the worker processes.

Usage
-----
    from tool_calling_data_generation.data_generators.backends import vLLMParallelGenerator

    backend = vLLMParallelGenerator("ibm-granite/granite-4.1-3b", num_gpus=8)
    backend.initialize_backend(
        gpu_memory_utilization=0.85, model_ctx=16384, turbo_mode="True"
    )
    outputs = backend.generate(prompts, temperature=0.7, max_tokens=512)
    backend.unload_backend()
"""

import atexit
import gc
import logging
import multiprocessing as mp
import os
import queue as _queue
import sys
import traceback
from typing import Any

import torch
from tqdm import tqdm

from tool_calling_data_generation.data_generators.backends.base import BaseGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

logging.getLogger("vllm").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker <-> parent protocol
# ---------------------------------------------------------------------------
_TAG_READY = "ready"  # (READY, rank, None) — engine loaded on this rank
_TAG_RESULT = "result"  # (RESULT, rank, job_id, List[str])
_TAG_ERROR = "error"  # (ERROR, rank, job_id|None, traceback_str)
_TAG_SHUTDOWN = "shutdown"  # task sentinel → worker exits

_SHUTDOWN_TIMEOUT_S = 60.0  # grace period for a worker to exit cleanly
_RESULT_WATCHDOG_S = 5.0  # how often the parent re-checks worker liveness
_INIT_WATCHDOG_S = 30.0  # how often the parent re-checks worker liveness at init

#: vLLM ``SamplingParams`` keys we accept as pass-through in ``generate()``.
_SAMPLING_PARAM_KEYS = (
    "temperature",
    "max_tokens",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stop",
    "stop_token_ids",
)

#: Currently-initialized instances (at most one at a time — see
#: ``initialize_backend``).  Guards against GPU oversubscription when the
#: same process (e.g. a notebook kernel) creates a second generator while a
#: previous one is still loaded.
_ACTIVE_INSTANCES: list["vLLMParallelGenerator"] = []

#: ``LLM`` engine kwargs we accept in ``initialize_backend(**kwargs)``.
_ENGINE_KWARG_KEYS = (
    "dtype",
    "quantization",
    "trust_remote_code",
    "revision",
    "tokenizer",
    "tokenizer_mode",
    "hf_token",
    "seed",
    "disable_custom_all_reduce",
    "download_dir",
    "swap_space",
    "cpu_offload_gb",
    "max_model_len",
    "enforce_eager",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
)


def _split_ranges(n: int, k: int) -> list[tuple[int, int]]:
    """Split ``n`` items into ``k`` contiguous ``(start, end)`` ranges.

    Sizes differ by at most one, so every worker gets a balanced slice.
    """
    base, extra = divmod(n, k)
    out: list[tuple[int, int]] = []
    idx = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append((idx, idx + size))
        idx += size
    return out


def _teardown_engine(llm: Any) -> None:
    """Free GPU/CPU resources of one vLLM engine (runs inside the worker)."""
    if llm is None:
        return
    try:
        llm.sleep(level=2)  # discard weights + KV cache across all workers
    except Exception:
        logger.warning("vLLM sleep(level=2) failed during teardown", exc_info=True)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#: Colours cycled across worker ranks so each bar is visually distinct.
#: Hex values (``#rrggbb``) only: tqdm 4.70's ``Bar.COLOURS`` accepts just
#: the 8 basic ANSI names plus hex, and any other name triggers a
#: "Unknown colour (...); valid choices: ..." warning.  Only applied on a
#: real TTY; hex degrades gracefully on terminals without truecolor.
_RANK_TQDM_COLOURS = (
    "#3fb950",  # green
    "#39c5cf",  # cyan
    "#f778ba",  # pink
    "#d29922",  # yellow
    "#58a6ff",  # blue
    "#f47067",  # red
    "#bc8cff",  # purple
    "#e6edf3",  # near-white
)


def _make_rank_progress(rank: int, bar_lock: Any = None) -> Any:
    """Build a per-worker ``tqdm`` factory for ``LLM.generate(use_tqdm=...)``.

    vLLM 0.22+ accepts a callable in place of a bool for ``use_tqdm`` and
    uses it as the ``tqdm`` constructor for its internal progress bar
    (``LLM._run_engine``: ``tqdm_func = use_tqdm if callable(...) else tqdm``).

    The factory pins the bar to this worker's ``rank`` via ``position=rank``,
    so every worker's bar stacks on its own terminal line and advances at
    its own pace, independently of the other processes.  vLLM's tokenization
    path calls the factory with the prompt iterable instead of ``total=``;
    that iterable is handed back untouched (only the generation loop gets a
    bar).

    Robustness details (why these exist):
      * ``ncols`` is pinned to a fixed, bounded width instead of vLLM's
        ``dynamic_ncols=True``.  tqdm only trims the rendered line to the
        terminal width when ``ncols`` is known, and dynamic re-measuring can
        yield ``None`` (flaky ``ioctl``/resize).  An untrimmed line wraps on
        the terminal, which shifts every stacked bar below it by a row and
        leaves frozen "ghost" bars behind.
      * ``bar_lock`` is a shared ``multiprocessing.RLock`` created in the
        parent and injected as the bar's write lock.  tqdm's default lock
        only works with fork; under the spawn context each worker gets its
        own private lock, so concurrent ``\r``/cursor-move writes from
        different processes race and corrupt the stacked layout.
      * Only valid tqdm colours (hex) are used; unknown colour names make
        tqdm emit a "Unknown colour (...); valid choices: ..." warning.

    Bars auto-disable when stderr is not a TTY (redirected logs, notebooks)
    and can be forced off with ``VLLM_PARALLEL_TQDM=0``.
    """
    forced_off = os.environ.get("VLLM_PARALLEL_TQDM", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }

    # Fixed, bounded bar width: measured once per job, never re-measured.
    # Override with VLLM_PARALLEL_TQDM_NCOLS=<int>.
    width = None
    try:
        width = os.get_terminal_size(sys.stderr.fileno()).columns - 1
    except Exception:
        width = None  # non-TTY or undetectable -> fall back to default
    try:
        ncols = int(
            os.environ.get(
                "VLLM_PARALLEL_TQDM_NCOLS", str(min(width, 120) if width else 120)
            )
        )
    except ValueError:
        ncols = 120
    ncols = max(30, min(ncols, 200))

    def _factory(iterable_or_total: Any = None, **kwargs: Any) -> Any:
        if "total" not in kwargs:
            # Tokenization path: vLLM wraps the prompt iterable here (called
            # as ``tqdm_func(iterable)``, no ``total`` kwarg); we do not want
            # a second bar per worker, so pass the iterable through untouched.
            if not isinstance(iterable_or_total, int):
                return iterable_or_total
            kwargs["total"] = iterable_or_total
        kwargs.pop("desc", None)  # vLLM passes its own desc; we override it.
        kwargs.pop("dynamic_ncols", None)  # pinned width (see docstring)
        bar = tqdm(
            desc=f"[vllm rank {rank}]",
            position=rank,
            leave=False,
            ncols=ncols,
            disable=True if forced_off else None,  # None -> auto (non-TTY)
            colour=_RANK_TQDM_COLOURS[rank % len(_RANK_TQDM_COLOURS)]
            if (not forced_off and sys.stderr.isatty())
            else None,
            **kwargs,
        )
        if bar_lock is not None:
            # Serialize all bar writes across worker processes (docstring).
            bar._lock = bar_lock
        return bar

    return _factory


def _worker_main(
    model_id: str,
    engine_kwargs: dict[str, Any],
    rank: int,
    gpu_ids: list[int],
    task_queue: "mp.Queue",
    result_queue: "mp.Queue",
    bar_lock: Any = None,
) -> None:
    """Run one vLLM engine pinned to ``gpu_ids`` and serve generation jobs.

    Lives in its own process (spawn context).  Protocol:
      * emits ``(READY, rank, None)`` or ``(ERROR, rank, None, tb)`` once,
      * consumes ``(job_id, prompts, sampling_params_dict)`` tasks,
      * emits ``(RESULT, rank, job_id, texts)`` or
        ``(ERROR, rank, job_id, tb)`` per task,
      * stops on ``None`` / ``(SHUTDOWN,)``.
    """
    # Pin this worker to its GPU(s) *before* anything initializes CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    try:
        from vllm import LLM, SamplingParams
    except Exception:  # pragma: no cover - import failure in child
        result_queue.put((_TAG_ERROR, rank, None, traceback.format_exc()))
        return

    llm: LLM | None = None
    try:
        llm = LLM(model_id, **engine_kwargs)
        result_queue.put((_TAG_READY, rank, None))
    except Exception:
        result_queue.put((_TAG_ERROR, rank, None, traceback.format_exc()))
        return

    try:
        while True:
            msg = task_queue.get()
            if msg is None or (
                isinstance(msg, tuple) and msg and msg[0] == _TAG_SHUTDOWN
            ):
                break
            job_id, prompts, sp_dict = msg
            try:
                sampling_params = SamplingParams(**sp_dict)
                outputs = llm.generate(
                    prompts,
                    sampling_params,
                    use_tqdm=_make_rank_progress(rank, bar_lock),
                )
                texts = [o.outputs[0].text.strip() for o in outputs]
                result_queue.put((_TAG_RESULT, rank, job_id, texts))
            except Exception:
                result_queue.put((_TAG_ERROR, rank, job_id, traceback.format_exc()))
    finally:
        _teardown_engine(llm)


class vLLMParallelGenerator(BaseGenerator):
    """Data-parallel vLLM backend: one engine per GPU, batch sharded across them.

    Args:
        model_id: HuggingFace model ID (or local path) to generate with.
        num_gpus: Number of GPUs to use.  ``None`` (default) uses every GPU
            visible to the process (``torch.cuda.device_count()``).
    """

    def __init__(
        self,
        model_id: str,
        num_gpus: int | None = None,
    ) -> None:
        super().__init__(model_id)
        self.num_gpus = (
            int(num_gpus) if num_gpus is not None else torch.cuda.device_count()
        )
        if self.num_gpus < 1:
            raise ValueError(
                "vLLMParallelGenerator requires at least one GPU, but "
                f"num_gpus={self.num_gpus}. Pass num_gpus=<N> explicitly "
                "or unset CUDA_VISIBLE_DEVICES."
            )
        self._ctx = mp.get_context("spawn")
        self._processes: list[mp.Process] = []
        self._task_queues: list[mp.Queue] = []
        self._result_queue: mp.Queue | None = None
        self._job_counter = 0
        self._atexit_registered = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize_backend(self, **kwargs) -> None:
        """Spawn one vLLM engine per GPU and wait until all are ready.

        Accepts the same kwargs as ``vLLMGenerator`` plus a few extras:

        - ``gpu_memory_utilization`` (float, default 0.95 turbo / 0.8 else)
        - ``model_ctx`` (int) → ``max_model_len``
        - ``turbo_mode`` (bool|str) → aggressive batching settings
        - ``tensor_parallel_size`` (int, default 1) → GPUs per engine
        - any key in ``_ENGINE_KWARG_KEYS`` is forwarded to ``LLM``
        """
        if self.is_backend_loaded:
            logger.info("Backend already loaded; skipping initialization.")
            return

        available_gpus = torch.cuda.device_count()
        if self.num_gpus > available_gpus:
            raise RuntimeError(
                f"Requested num_gpus={self.num_gpus} but only {available_gpus} "
                "GPU(s) are visible to this process. Pass a smaller num_gpus "
                "or unset CUDA_VISIBLE_DEVICES."
            )

        turbo_mode = str(kwargs.get("turbo_mode", "False")).lower() == "true"
        if os.getenv("GPU_GO_BRRR", "False").lower() == "true":
            turbo_mode = True
        tensor_parallel_size = int(kwargs.get("tensor_parallel_size", 1))
        if self.num_gpus % tensor_parallel_size != 0:
            raise ValueError(
                f"num_gpus={self.num_gpus} is not divisible by "
                f"tensor_parallel_size={tensor_parallel_size}."
            )
        n_replicas = self.num_gpus // tensor_parallel_size

        engine_kwargs: dict[str, Any] = {
            "gpu_memory_utilization": kwargs.get(
                "gpu_memory_utilization", 0.95 if turbo_mode else 0.8
            ),
            "max_model_len": int(
                kwargs.get("model_ctx", kwargs.get("max_model_len", 16384))
            ),
            "tensor_parallel_size": tensor_parallel_size,
        }
        if turbo_mode:
            engine_kwargs.update(
                enforce_eager=True,
                max_num_batched_tokens=131072,
                enable_chunked_prefill=False,
                max_num_seqs=2048,
            )
        for key in _ENGINE_KWARG_KEYS:
            if key in kwargs and key not in (
                "max_model_len",
                "enforce_eager",
                "max_num_seqs",
                "max_num_batched_tokens",
                "enable_chunked_prefill",
            ):
                engine_kwargs[key] = kwargs[key]
        # explicit per-key overrides (non-turbo path, or user-tuned)
        for key in (
            "enforce_eager",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
        ):
            if key in kwargs:
                engine_kwargs[key] = kwargs[key]

        gpu_groups = [
            list(range(r * tensor_parallel_size, (r + 1) * tensor_parallel_size))
            for r in range(n_replicas)
        ]

        logger.info(
            "vLLMParallelGenerator: %d replica(s), tensor_parallel_size=%d, "
            "GPUs=%s, model=%s",
            n_replicas,
            tensor_parallel_size,
            gpu_groups,
            self.model_id,
        )

        # Load the tokenizer in the parent (CPU-only) so that the DPO
        # pipeline can use ``backend.tokenizer`` after init.
        self._load_tokenizer()

        self._result_queue = self._ctx.Queue()
        # Shared lock that serializes progress-bar writes across worker
        # processes (tqdm's default write lock is fork-only; see
        # ``_make_rank_progress`` for why this prevents bar corruption).
        bar_lock = self._ctx.RLock()
        for rank, gpu_group in enumerate(gpu_groups):
            task_q = self._ctx.Queue()
            proc = self._ctx.Process(
                target=_worker_main,
                args=(
                    self.model_id,
                    engine_kwargs,
                    rank,
                    gpu_group,
                    task_q,
                    self._result_queue,
                    bar_lock,
                ),
                name=f"vllm-parallel-rank{rank}",
                daemon=False,
            )
            proc.start()
            self._processes.append(proc)
            self._task_queues.append(task_q)

        # Only one parallel backend may own the GPUs at a time; unload any
        # previous instance first (matters when re-running notebook cells).
        # (Iterate over a copy: unload_backend() mutates _ACTIVE_INSTANCES.)
        for inst in _ACTIVE_INSTANCES[:]:
            if inst is not self:
                inst.unload_backend()
        _ACTIVE_INSTANCES.append(self)

        self._wait_until_ready(n_replicas)

        if not self._atexit_registered:
            atexit.register(self.unload_backend)
            self._atexit_registered = True

        self.is_backend_loaded = True
        logger.info("All %d vLLM workers are ready.", n_replicas)

    def _load_tokenizer(self) -> None:
        """Populate ``self.tokenizer`` with an HF tokenizer (best effort)."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        except Exception:
            logger.warning(
                "Could not load an HF tokenizer for %s in the parent process; "
                "backend.tokenizer will be None.",
                self.model_id,
                exc_info=True,
            )
            self.tokenizer = None

    def _wait_until_ready(self, n_replicas: int) -> None:
        """Block until every worker reports READY (or an ERROR)."""
        ready = 0
        errors: list[str] = []
        while ready < n_replicas:
            try:
                msg = self._result_queue.get(timeout=_INIT_WATCHDOG_S)
            except _queue.Empty:
                if any(not p.is_alive() for p in self._processes):
                    break  # a worker died before reporting
                continue
            tag = msg[0]
            if tag == _TAG_READY:
                ready += 1
            elif tag == _TAG_ERROR:
                errors.append(f"rank {msg[1]}:\n{msg[3]}")
        if errors or ready < n_replicas:
            self._abort_initialization()
            detail = (
                "\n".join(errors)
                if errors
                else (f"only {ready}/{n_replicas} workers became ready.")
            )
            raise RuntimeError(
                f"Failed to initialize vLLMParallelGenerator workers. {detail}"
            )

    def _abort_initialization(self) -> None:
        """Shut down partially-started workers after a failed init."""
        try:
            _ACTIVE_INSTANCES.remove(self)
        except ValueError:
            pass
        for q in self._task_queues:
            try:
                q.put(None, timeout=5)
            except Exception:
                pass
        for p in self._processes:
            try:
                p.join(timeout=10)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=5)
                except Exception:
                    pass
        self._processes.clear()
        self._task_queues.clear()
        self._result_queue = None
        self.is_backend_loaded = False

    def unload_backend(self) -> None:
        """Shut down all worker engines and free GPU memory."""
        if not self.is_backend_loaded and not self._processes:
            return
        try:
            _ACTIVE_INSTANCES.remove(self)
        except ValueError:
            pass
        logger.info(
            "Unloading vLLMParallelGenerator (%d workers) ...", len(self._processes)
        )
        for q in self._task_queues:
            try:
                q.put(None, timeout=5)
            except Exception:
                pass
        for i, p in enumerate(self._processes):
            p.join(timeout=_SHUTDOWN_TIMEOUT_S)
            if p.is_alive():
                logger.warning("Worker %d did not exit; terminating.", i)
                try:
                    p.terminate()
                    p.join(timeout=10)
                except Exception:
                    pass
                if p.is_alive():
                    try:
                        p.kill()
                        p.join(timeout=5)
                    except Exception:
                        pass
        for q in self._task_queues + (
            [self._result_queue] if self._result_queue else []
        ):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
        self._processes.clear()
        self._task_queues.clear()
        self._result_queue = None
        self.is_backend_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("vLLMParallelGenerator unloaded.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate one completion per prompt across all GPUs in parallel.

        Returns a ``List[str]`` in the same order as *prompts*.
        """
        if not self.is_backend_loaded:
            raise RuntimeError(
                "vLLMParallelGenerator.generate() called before initialize_backend()."
            )
        if not prompts:
            return []
        n_workers = len(self._processes)
        if n_workers == 0:
            raise RuntimeError("No vLLM workers available; call initialize_backend().")

        sp_dict: dict[str, Any] = {}
        for key in _SAMPLING_PARAM_KEYS:
            if key in kwargs:
                sp_dict[key] = kwargs[key]
        sp_dict.setdefault("temperature", 0.7)
        sp_dict.setdefault("max_tokens", 512)

        chunk_plan = _split_ranges(len(prompts), n_workers)
        job_id = self._job_counter
        self._job_counter += 1

        pending: set = set()
        for rank, (start, end) in enumerate(chunk_plan):
            if end > start:
                self._task_queues[rank].put((job_id, prompts[start:end], sp_dict))
                pending.add(rank)

        results: dict[int, list[str]] = {}
        errors: list[str] = []
        while pending:
            try:
                msg = self._result_queue.get(timeout=_RESULT_WATCHDOG_S)
            except _queue.Empty:
                dead = [r for r in pending if not self._processes[r].is_alive()]
                if dead:
                    raise RuntimeError(
                        "vLLM worker process(es) died during generation: "
                        f"ranks {dead}. Check the worker logs / GPU memory."
                    )
                continue
            tag = msg[0]
            if tag == _TAG_RESULT:
                _, rank, got_job_id, texts = msg
                if got_job_id != job_id:
                    continue  # stale message from an earlier job
                results[rank] = texts
                pending.discard(rank)
            elif tag == _TAG_ERROR:
                _, rank, got_job_id, tb = msg
                if got_job_id is not None and got_job_id != job_id:
                    continue
                errors.append(f"rank {rank}:\n{tb}")
                pending.discard(rank)

        if errors:
            raise RuntimeError(
                "vLLM workers reported errors during generation:\n" + "\n".join(errors)
            )

        merged: list[str] = []
        for rank, (start, end) in enumerate(chunk_plan):
            if end > start:
                merged.extend(results.get(rank, []))
        if len(merged) != len(prompts):
            raise RuntimeError(
                f"Output count mismatch: got {len(merged)} outputs for "
                f"{len(prompts)} prompts. Something went wrong in a worker."
            )
        return merged

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.unload_backend()
        except Exception:
            pass


__all__ = ["vLLMParallelGenerator"]


