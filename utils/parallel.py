"""Process-level multi-GPU work dispatch.

The pipeline is a set of *independent* GPU jobs — one LoRA fine-tune per expert,
one SLG run per ablation, one run per scalability size. Each job already fits on
a single GPU, so the fastest way to use N GPUs is to run N jobs at once, one per
GPU, rather than sharding a single model across devices.

``run_parallel`` takes a list of picklable *tasks* and a top-level worker
function, and runs each task exactly once, each pinned to one GPU via
``CUDA_VISIBLE_DEVICES``. The number of workers auto-scales to however many GPUs
are visible (``CUDA_VISIBLE_DEVICES`` under SLURM, else ``torch.cuda`` count), so
requesting more GPUs in ``job.sh`` speeds things up with no code change.

Consistency guarantee: a task is the atomic unit of work and is never split, so
its result is identical to a single-GPU run — only the *scheduling* changes.
With <=1 visible GPU (or ``SLG_DISABLE_PARALLEL=1``) it degrades to plain
in-process sequential execution, i.e. the original behaviour.

Uses the ``spawn`` start method so it is safe even after the parent has
initialised CUDA; module-level imports here stay torch-free so a worker can set
its device before torch is imported.
"""

import multiprocessing as mp
import os
import queue as _queue
from typing import Any, List, Sequence, Tuple

WorkerRef = Tuple[str, str]  # (module_name, function_name)


def visible_gpu_ids() -> List[str]:
    """Tokens for every GPU this process may use, verbatim from the environment.

    Passing these tokens through to a child's ``CUDA_VISIBLE_DEVICES`` keeps the
    selection correct whether SLURM exposes cgroup-local indices ("0,1,2,3") or
    physical ids / MIG UUIDs.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [tok for tok in cvd.split(",") if tok.strip() != ""]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _disabled() -> bool:
    return os.environ.get("SLG_DISABLE_PARALLEL", "").strip() not in ("", "0", "false", "False")


def _worker(gpu_id: str, worker_ref: WorkerRef, task_q, result_q) -> None:
    # Pin the device BEFORE importing torch (via the worker module), so this
    # process only ever sees its one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import importlib

    module_name, fn_name = worker_ref
    fn = getattr(importlib.import_module(module_name), fn_name)
    while True:
        item = task_q.get()
        if item is None:  # sentinel: no more work for this worker
            return
        idx, task = item
        try:
            result_q.put((idx, None, fn(task)))
        except Exception:
            import traceback

            result_q.put((idx, traceback.format_exc(), None))


def _run_sequential(worker_ref: WorkerRef, tasks: Sequence[Any], label: str) -> List[Any]:
    import importlib

    from logging_config import logger

    module_name, fn_name = worker_ref
    fn = getattr(importlib.import_module(module_name), fn_name)
    results: List[Any] = []
    for i, task in enumerate(tasks):
        logger.info("[%s] task %d/%d (sequential, 1 GPU)", label, i + 1, len(tasks))
        results.append(fn(task))
    return results


def _run_pool(worker_ref: WorkerRef, tasks: Sequence[Any], label: str, gpus: List[str]) -> List[Any]:
    from logging_config import logger

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    for i, task in enumerate(tasks):
        task_q.put((i, task))
    for _ in gpus:
        task_q.put(None)  # one sentinel per worker

    procs = [
        ctx.Process(target=_worker, args=(gpu, worker_ref, task_q, result_q), daemon=False)
        for gpu in gpus
    ]
    for p in procs:
        p.start()
    logger.info("[%s] dispatching %d task(s) across %d GPU(s): %s",
                label, len(tasks), len(gpus), ",".join(gpus))

    results: List[Any] = [None] * len(tasks)
    failures = {}
    done = 0
    while done < len(tasks):
        try:
            idx, err, res = result_q.get(timeout=2)
        except _queue.Empty:
            # Guard against a worker dying without reporting (e.g. OOM kill /
            # segfault) so we do not block forever.
            if all(not p.is_alive() for p in procs):
                while True:
                    try:
                        idx, err, res = result_q.get_nowait()
                    except _queue.Empty:
                        break
                    done += 1
                    (failures.__setitem__(idx, err) if err else results.__setitem__(idx, res))
                if done < len(tasks):
                    raise RuntimeError(
                        f"[{label}] workers exited early; only {done}/{len(tasks)} "
                        "task(s) reported. Check the log above for the crash."
                    )
            continue
        done += 1
        if err:
            failures[idx] = err
            logger.error("[%s] task %d FAILED:\n%s", label, idx, err)
        else:
            results[idx] = res
            logger.info("[%s] completed %d/%d", label, done, len(tasks))

    for p in procs:
        p.join()
    if failures:
        first = next(iter(failures.values()))
        raise RuntimeError(
            f"[{label}] {len(failures)}/{len(tasks)} task(s) failed. First traceback:\n{first}"
        )
    return results


def run_parallel(worker_ref: WorkerRef, tasks: Sequence[Any], label: str = "tasks") -> List[Any]:
    """Run ``tasks`` across every visible GPU and return results in task order.

    ``worker_ref`` is ``(module_name, function_name)`` of a top-level callable
    taking a single picklable ``task`` and returning a picklable result. Raises
    ``RuntimeError`` if any task fails (after all others have been attempted).
    """
    tasks = list(tasks)
    if not tasks:
        return []
    gpus = visible_gpu_ids()
    if _disabled() or len(gpus) <= 1:
        return _run_sequential(worker_ref, tasks, label)
    return _run_pool(worker_ref, tasks, label, gpus)
