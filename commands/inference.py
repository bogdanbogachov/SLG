import os
from openai import OpenAI
from logging_config import logger
from config import CONFIG
from utils.parallel import run_parallel
from utils.path_utils import ensure_dir, get_answers_root


def run_baseline(experiment: str):
    from inference.baseline import ask_baseline
    client = OpenAI(api_key=CONFIG['open_ai_api_key'])
    ensure_dir(os.path.join(get_answers_root(experiment), experiment))
    files_config = CONFIG['files']
    model = CONFIG['models']['gpt_4_1']
    ask_baseline(file=files_config['qa_test'], model=model, experiment=experiment, client=client)


def run_rag(experiment: str):
    from inference.baseline import AskRag
    client = OpenAI(api_key=CONFIG['open_ai_api_key'])
    files_config = CONFIG['files']
    rag = AskRag(
        documents_file=files_config['qa_train'],
        questions_file=files_config['qa_test'],
        experiment=experiment,
        client=client
    )
    rag.generate_responses()


def _finetuned_infer_worker(task: dict) -> None:
    """Run one fine-tuned-baseline inference pass (picklable for the GPU pool)."""
    from inference.baseline import ask_finetuned
    ask_finetuned(**task)


def run_finetuned(experiment: str, output_suffix: str = ""):
    paths_config = CONFIG['paths']
    models_paths = paths_config['models']
    adapters_config = CONFIG['adapters']
    experiments_dir = paths_config['experiments']
    downloaded_models_dir = paths_config['downloaded_models']

    ensure_dir(os.path.join(get_answers_root(experiment), experiment))
    files_config = CONFIG['files']

    base_model_3_2_1b = os.path.join(downloaded_models_dir, models_paths['3_2_1b'])
    base_model_3_1_8b = os.path.join(downloaded_models_dir, models_paths['3_1_8b'])

    # A quick-check (--limit) run writes to the sibling answers/<exp>/<exp>__limitN/
    # folder so it never overwrites the full baseline outputs.
    output_label = experiment + output_suffix

    training_components = CONFIG['training_components']
    tasks = []
    if training_components.get('train_3_2_1b', False):
        tasks.append({
            "file": files_config['qa_test'],
            "base_model": base_model_3_2_1b,
            "adapter": os.path.join(experiments_dir, experiment, adapters_config['finetuned_3_2_1b']),
            "experiment": experiment,
            "output_label": output_label,
        })
    if training_components.get('train_3_1_8b', False):
        tasks.append({
            "file": files_config['qa_test'],
            "base_model": base_model_3_1_8b,
            "adapter": os.path.join(experiments_dir, experiment, adapters_config['finetuned_3_1_8b']),
            "experiment": experiment,
            "output_label": output_label,
        })

    if not tasks:
        logger.info("No fine-tuned baselines selected; nothing to do.")
        return
    run_parallel(("commands.inference", "_finetuned_infer_worker"), tasks, label="infer_finetuned")


def run_slg(experiment: str, ablation: str = "full", output_suffix: str = ""):
    """Batch SLG inference. ``ablation`` selects a leave-one-out preset (#2);
    non-full runs write to answers/<experiment>/<experiment>__<ablation>/.
    ``output_suffix`` (e.g. "__limit50") isolates a quick-check run's outputs.

    Speed path: a ``full`` run's **round 1** (route+answer+verify over the whole
    test set — the dominant cost) is sharded data-parallel across every visible
    GPU, then A/C are replayed in canonical order and the small reroute rounds run
    sequentially (:meth:`SmallLanguageRouter.finish_from_round1`). This is
    equivalent to the single-stream run because round-1 routing is A-independent
    (empty competence on the first pass); see ``answer_shard_round1``. Falls back
    to the single-GPU stream for ablations, quick-checks, 1 GPU, or
    ``SLG_DISABLE_PARALLEL``."""
    import os as _os
    from inference.slg import SmallLanguageRouter
    from inference.slg.ablation import get_ablation
    from utils.parallel import visible_gpu_ids
    ensure_dir(os.path.join(get_answers_root(experiment), experiment))
    files_config = CONFIG['files']
    qa_file = files_config['qa_test']

    use_sharded_round1 = (
        ablation == "full" and not output_suffix
        and len(visible_gpu_ids()) > 1
        and not _os.environ.get("SLG_DISABLE_PARALLEL")
    )
    if use_sharded_round1:
        _run_full_round1_sharded(experiment, qa_file, output_suffix)
        return

    router = SmallLanguageRouter(
        experts_location=experiment, experiment=experiment,
        ablation=get_ablation(ablation), output_suffix=output_suffix,
    )
    router.ask(file=qa_file)


def _slg_full_round1_worker(task: dict) -> dict:
    """Run round 1 of a ``full`` run over one contiguous shard (picklable for the
    GPU pool); returns raw per-global-index results with no A/C applied."""
    from inference.slg import SmallLanguageRouter
    from inference.slg.ablation import get_ablation
    router = SmallLanguageRouter(
        experts_location=task["experiment"], experiment=task["experiment"],
        ablation=get_ablation("full"),
    )
    return router.answer_shard_round1(task["file"], task["shard_index"], task["num_shards"])


def _run_full_round1_sharded(experiment: str, qa_file: str, output_suffix: str = ""):
    """Shard round 1 of a full run across GPUs, then replay A/C + reroutes on one GPU."""
    import json
    from inference.slg import SmallLanguageRouter
    from inference.slg.ablation import get_ablation
    from utils.parallel import visible_gpu_ids

    from inference.slg.session import SessionState

    label = experiment + output_suffix
    out_path = os.path.join(get_answers_root(experiment), label, "slg.json")
    with open(qa_file, "r", encoding="utf-8") as f:
        n = len(json.load(f))
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            if len(json.load(f)) == n:
                logger.info("SLG full run already complete (%d/%d); skipping.", n, n)
                return

    finish_router = SmallLanguageRouter(
        experts_location=experiment, experiment=experiment,
        ablation=get_ablation("full"), output_suffix=output_suffix,
    )

    # Resume an interrupted reroute tail WITHOUT re-sharding round 1: round 1 is
    # not checkpointed, but once the (sequential) tail has started, its checkpoint
    # carries the replayed round-1 A/C state + statuses, so we continue from there.
    ckpt = finish_router._checkpoint_path()
    probe = SessionState(ablation=get_ablation("full"))
    state, start_attempt, round_progress = finish_router._load_checkpoint(ckpt, probe, n)
    if state is not None:
        logger.info("SLG full: resuming reroute tail from checkpoint; skipping round-1 sharding.")
        finish_router._finish_tail(qa_file, state, probe, start_attempt, round_progress, ckpt)
        return

    num_shards = max(1, len(visible_gpu_ids()))
    _warm_expert_cache(experiment)
    tasks = [
        {"experiment": experiment, "file": qa_file, "shard_index": k, "num_shards": num_shards}
        for k in range(num_shards)
    ]
    logger.info("SLG full: sharding round 1 of %d question(s) across %d GPU(s).", n, num_shards)
    results = run_parallel(("commands.inference", "_slg_full_round1_worker"),
                           tasks, label="slg_full_round1")

    round1 = {}
    for r in results:
        if r:
            round1.update(r["results"])

    if len(round1) != n:
        logger.warning("Round-1 shards returned %d/%d results; falling back to single-stream.",
                       len(round1), n)
        finish_router.ask(file=qa_file)
        return
    finish_router.finish_from_round1(qa_file, round1)


def _warm_expert_cache(experiment: str) -> None:
    """Build the shared expert-embedding cache once, before any parallel SLG
    runs, so concurrent workers only ever *read* it (no write race). Cheap no-op
    if the cache already exists."""
    from inference.slg.retriever import ExpertRetriever
    ExpertRetriever(experiment)  # builds experiments/<exp>/slg_index/ if missing


def _slg_ablation_worker(task: dict) -> None:
    """Run one SLG ablation preset (picklable for the GPU pool)."""
    run_slg(task["experiment"], ablation=task["ablation"])


def _slg_base_shard_worker(task: dict) -> dict:
    """Answer one contiguous shard of the test set under the ``base`` ablation
    (picklable for the GPU pool); returns picklable partial results by global
    index. Safe to shard only because base has no online A/C coupling."""
    from inference.slg import SmallLanguageRouter
    from inference.slg.ablation import get_ablation
    router = SmallLanguageRouter(
        experts_location=task["experiment"], experiment=task["experiment"],
        ablation=get_ablation("base"),
    )
    return router.answer_shard(task["file"], task["shard_index"], task["num_shards"])


def run_slg_ablations(experiment: str):
    """Run the full leave-one-out ablation suite (#2): full, -A, -B, -C, base.

    The four *coupled* presets (full, -A, -B, -C) carry online competence (A)
    and/or calibration (C) state that evolves across the question stream, so
    each must run as a single ordered process — they are dispatched one-per-GPU
    (result identical to a single-GPU run; only which GPU runs which preset
    changes). ``base`` turns A and C off, so its questions are independent: it is
    instead **sharded data-parallel across all GPUs** and merged in order
    (bit-identical answers), filling the GPUs that would otherwise sit idle while
    the odd preset out runs solo."""
    from inference.slg.ablation import PRESETS
    _warm_expert_cache(experiment)

    coupled = [name for name in PRESETS if name != "base"]
    tasks = [{"experiment": experiment, "ablation": name} for name in coupled]
    run_parallel(("commands.inference", "_slg_ablation_worker"), tasks, label="slg_ablations")

    if "base" in PRESETS:
        _run_base_sharded(experiment)


def _run_base_sharded(experiment: str):
    """Run the ``base`` ablation sharded across every visible GPU, then merge."""
    import json
    from inference.slg.pipeline import merge_sharded_base
    from utils.parallel import visible_gpu_ids

    qa_file = CONFIG['files']['qa_test']
    label = f"{experiment}__base"
    out_path = os.path.join(get_answers_root(experiment), label, "slg.json")

    with open(qa_file, "r", encoding="utf-8") as f:
        n = len(json.load(f))
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            if len(json.load(f)) == n:
                logger.info("Base ablation already complete (%d/%d); skipping.", n, n)
                return

    num_shards = max(1, len(visible_gpu_ids()))
    tasks = [
        {"experiment": experiment, "file": qa_file, "shard_index": k, "num_shards": num_shards}
        for k in range(num_shards)
    ]
    logger.info("Base ablation: sharding %d question(s) across %d GPU(s).", n, num_shards)
    results = run_parallel(("commands.inference", "_slg_base_shard_worker"), tasks, label="slg_base_shards")
    merge_sharded_base(experiment, qa_file, [r for r in results if r])


def _scalability_size_worker(task: dict) -> dict:
    """Run the fixed question set at one pool size and return its metrics
    (picklable for the GPU pool). Each size gets a dedicated GPU, so latency is
    measured without cross-size contention."""
    import time
    from inference.slg import SmallLanguageRouter
    from evaluate.slg_metrics import compute

    router = SmallLanguageRouter(
        experts_location=task["experiment"], experiment=task["experiment"],
        expert_subset=task["subset"],
    )
    router._output_label = task["label"]  # each size keeps its own answers
    start = time.perf_counter()
    router.ask(file=task["qa_file"])
    elapsed = time.perf_counter() - start
    metrics = compute(os.path.join(get_answers_root(task["experiment"]), task["label"]))
    return {
        "n_experts": task["k"],
        "n_core": task["n_core"],
        "n_distractors": task["n_distractors"],
        "latency_s": round(elapsed, 2),
        "latency_per_q_s": round(elapsed / max(metrics["summary"]["n"], 1), 3),
        "routing_accuracy": metrics["summary"]["routing_accuracy_overall"],
        "coverage": metrics["summary"]["coverage"],
    }


def run_slg_scalability(experiment: str):
    """Scalability sweep (#5): distractor scaling on a fixed question set.

    The task is held constant: the same question set is asked at every pool
    size. The *core* experts (those that actually answer the questions) are
    always present; the pool is grown by adding **distractor** experts the
    questions never need. This isolates the effect of a larger pool — does
    latency stay flat and can the router still pick the right expert as
    irrelevant competitors are added — instead of penalising small pools for
    missing experts. Intended for the synthetic set (files.qa_scalability),
    which supplies the distractor experts.

    ``scalability_sizes`` are interpreted as *total* pool sizes (core +
    distractors); sizes below the core size or beyond the available distractors
    are skipped.
    """
    import json
    from inference.slg.pipeline import list_valid_experts
    from evaluate.slg_metrics import slug_title

    files_config = CONFIG['files']
    routing_cfg = CONFIG['routing']
    answers_root = get_answers_root(experiment)

    qa_file = files_config.get('qa_scalability', files_config['qa_test'])
    sizes = routing_cfg.get('scalability_sizes', [5, 10, 20])

    # Ground-truth experts the fixed question set actually needs (CPU-only plan).
    with open(qa_file, "r", encoding="utf-8") as f:
        qa = json.load(f)
    needed = {slug_title(item.get("title")) for item in qa}

    valid = set(list_valid_experts(experiment))
    core = sorted(needed & valid)                 # always present (answer the task)
    distractors = sorted(valid - needed)          # added to grow the pool
    if not core:
        logger.warning("Scalability: no routable expert covers %s; nothing to do.", qa_file)
        return
    missing = needed - valid
    if missing:
        logger.warning(
            "Scalability: %d question topics have no expert (e.g. %s); those questions "
            "cannot be routed correctly at any size.", len(missing), sorted(missing)[:5],
        )

    # Build one task per valid size; sizes are independent runs (each holds the
    # task constant), dispatched one-per-GPU. Each size run always has a whole
    # GPU to itself, so its latency is measured without contention — consistent
    # with a single-GPU sweep.
    tasks = []
    for k in sizes:
        n_distractors = k - len(core)
        if n_distractors < 0:
            logger.info("Skipping size %d (< core size %d; task can't be held constant).", k, len(core))
            continue
        if n_distractors > len(distractors):
            logger.info("Skipping size %d (need %d distractors, only %d available).",
                        k, n_distractors, len(distractors))
            continue
        tasks.append({
            "experiment": experiment,
            "qa_file": qa_file,
            "subset": core + distractors[:n_distractors],
            "label": f"{experiment}__scale{k}",
            "k": k,
            "n_core": len(core),
            "n_distractors": n_distractors,
        })

    if not tasks:
        logger.warning("Scalability: no valid pool sizes to run for %s.", qa_file)
        return

    _warm_expert_cache(experiment)
    results = run_parallel(("commands.inference", "_scalability_size_worker"),
                           tasks, label="slg_scalability")
    results = sorted((r for r in results if r), key=lambda r: r["n_experts"])

    out_dir = os.path.join(answers_root, experiment, "slg_diagnostics")
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "scalability.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Scalability sweep complete: %s", results)


def run_slg_metrics(experiment: str):
    """Compute selective-prediction + routing-curve metrics (#3, #4) for every
    finished SLG run found for this experiment (full + any ablations)."""
    import os as _os
    from inference.slg.ablation import PRESETS
    from evaluate.slg_metrics import run as run_metrics

    answers_root = get_answers_root(experiment)
    labels = [experiment] + [
        f"{experiment}{p.suffix}" for p in PRESETS.values() if p.suffix
    ]
    for label in labels:
        if not _os.path.isfile(_os.path.join(answers_root, label, "slg.json")):
            continue
        m = run_metrics(label)
        logger.info("Metrics for %s: %s", label, m["summary"])


def run_slg_all(experiment: str):
    """Run the whole SLG experiment suite in order, for a single batch job.

    1) leave-one-out ablations (#2)  ->  2) scalability sweep (#5, if a large
    enough expert pool exists)  ->  3) routing-curve + selective metrics (#3, #4).
    Each step is guarded so one failure does not abort the rest of the job.
    """
    steps = [
        ("ablations", lambda: run_slg_ablations(experiment)),
        ("scalability", lambda: run_slg_scalability(experiment)),
        ("metrics", lambda: run_slg_metrics(experiment)),
    ]
    outcomes = {}
    for name, fn in steps:
        try:
            logger.info("########## SLG suite: %s ##########", name)
            fn()
            outcomes[name] = "OK"
        except Exception:
            logger.exception("SLG suite step '%s' failed; continuing.", name)
            outcomes[name] = "FAILED"

    # Each step is guarded so one failure does not abort the rest — which means
    # the Slurm job can 'succeed' with steps missing. Surface a clear summary so
    # a failure is not silent; check output files if anything is not OK.
    summary = " ".join(f"{name}={outcomes[name]}" for name, _ in steps)
    if all(v == "OK" for v in outcomes.values()):
        logger.info("########## SLG suite summary: %s ##########", summary)
    else:
        logger.warning("########## SLG suite summary (INCOMPLETE): %s ##########", summary)


def run_paper_assets(experiment: str):
    """Aggregate all results (quality, ablation behaviour, scalability) into
    paper-ready LaTeX tables + figures under paper_assets/<experiment>/.

    Pure CPU post-processing; every source is optional. Run this last — after
    ``--slg_all`` and ``--evaluate`` — so it captures both the ablation/scalability
    behaviour and the answer-quality metrics."""
    from evaluate.paper_assets import build
    out_dir = build(experiment)
    logger.info("Paper assets written to %s", out_dir)


def run_slg_chat(experiment: str):
    from inference.slg import SmallLanguageRouter
    router = SmallLanguageRouter(experts_location=experiment, experiment=experiment)
    router.chat()
