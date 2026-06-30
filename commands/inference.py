import os
from openai import OpenAI
from logging_config import logger
from config import CONFIG
from utils.path_utils import ensure_dir


def run_baseline(experiment: str):
    from inference.baseline import ask_baseline
    client = OpenAI(api_key=CONFIG['open_ai_api_key'])
    paths_config = CONFIG['paths']
    answers_dir = paths_config['answers']
    ensure_dir(os.path.join(answers_dir, experiment))
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


def run_finetuned(experiment: str):
    from inference.baseline import ask_finetuned
    paths_config = CONFIG['paths']
    models_paths = paths_config['models']
    adapters_config = CONFIG['adapters']
    answers_dir = paths_config['answers']
    experiments_dir = paths_config['experiments']
    downloaded_models_dir = paths_config['downloaded_models']
    
    ensure_dir(os.path.join(answers_dir, experiment))
    files_config = CONFIG['files']
    
    base_model_3_2_1b = os.path.join(downloaded_models_dir, models_paths['3_2_1b'])
    base_model_3_1_8b = os.path.join(downloaded_models_dir, models_paths['3_1_8b'])
    
    training_components = CONFIG['training_components']
    if training_components.get('train_3_2_1b', False):
        ask_finetuned(file=files_config['qa_test'],
                      base_model=base_model_3_2_1b,
                      adapter=os.path.join(experiments_dir, experiment, adapters_config['finetuned_3_2_1b']),
                      experiment=experiment)
    
    if training_components.get('train_3_1_8b', False):
        ask_finetuned(file=files_config['qa_test'],
                      base_model=base_model_3_1_8b,
                      adapter=os.path.join(experiments_dir, experiment, adapters_config['finetuned_3_1_8b']),
                      experiment=experiment)


def run_slg(experiment: str, ablation: str = "full"):
    """Batch SLG inference. ``ablation`` selects a leave-one-out preset (#2);
    non-full runs write to answers/<experiment>__<ablation>/."""
    from inference.slg import SmallLanguageRouter
    from inference.slg.ablation import get_ablation
    paths_config = CONFIG['paths']
    ensure_dir(os.path.join(paths_config['answers'], experiment))
    files_config = CONFIG['files']
    router = SmallLanguageRouter(
        experts_location=experiment, experiment=experiment,
        ablation=get_ablation(ablation),
    )
    router.ask(file=files_config['qa_test'])


def run_slg_ablations(experiment: str):
    """Run the full leave-one-out ablation suite (#2): full, -A, -B, -C, base."""
    from inference.slg.ablation import PRESETS
    for name in PRESETS:
        logger.info("=== SLG ablation run: %s ===", name)
        run_slg(experiment, ablation=name)


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
    import time
    from inference.slg import SmallLanguageRouter
    from evaluate.slg_metrics import compute, slug_title

    paths_config = CONFIG['paths']
    files_config = CONFIG['files']
    routing_cfg = CONFIG['routing']
    answers_dir = paths_config['answers']

    qa_file = files_config.get('qa_scalability', files_config['qa_test'])
    sizes = routing_cfg.get('scalability_sizes', [5, 10, 20])

    # Ground-truth experts the fixed question set actually needs.
    with open(qa_file, "r", encoding="utf-8") as f:
        qa = json.load(f)
    needed = {slug_title(item.get("title")) for item in qa}

    probe = SmallLanguageRouter(experts_location=experiment, experiment=experiment)
    valid = set(probe._valid_experts)
    del probe

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

    results = []
    for k in sizes:
        n_distractors = k - len(core)
        if n_distractors < 0:
            logger.info("Skipping size %d (< core size %d; task can't be held constant).", k, len(core))
            continue
        if n_distractors > len(distractors):
            logger.info("Skipping size %d (need %d distractors, only %d available).",
                        k, n_distractors, len(distractors))
            continue
        subset = core + distractors[:n_distractors]
        label = f"{experiment}__scale{k}"
        logger.info("=== SLG scalability: %d experts (%d core + %d distractors) ===",
                    k, len(core), n_distractors)
        router = SmallLanguageRouter(
            experts_location=experiment, experiment=experiment, expert_subset=subset,
        )
        # Override the output label so each size keeps its own answers.
        router._output_label = label
        start = time.perf_counter()
        router.ask(file=qa_file)
        elapsed = time.perf_counter() - start
        metrics = compute(os.path.join(answers_dir, label))
        results.append({
            "n_experts": k,
            "n_core": len(core),
            "n_distractors": n_distractors,
            "latency_s": round(elapsed, 2),
            "latency_per_q_s": round(elapsed / max(metrics["summary"]["n"], 1), 3),
            "routing_accuracy": metrics["summary"]["routing_accuracy_overall"],
            "coverage": metrics["summary"]["coverage"],
        })

    out_dir = os.path.join(answers_dir, experiment, "slg_diagnostics")
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "scalability.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Scalability sweep complete: %s", results)
    logger.info("Scalability sweep complete: %s", results)


def run_slg_metrics(experiment: str):
    """Compute selective-prediction + routing-curve metrics (#3, #4) for every
    finished SLG run found for this experiment (full + any ablations)."""
    import os as _os
    from inference.slg.ablation import PRESETS
    from evaluate.slg_metrics import run as run_metrics

    answers_dir = CONFIG['paths']['answers']
    labels = [experiment] + [
        f"{experiment}{p.suffix}" for p in PRESETS.values() if p.suffix
    ]
    for label in labels:
        if not _os.path.isfile(_os.path.join(answers_dir, label, "slg.json")):
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
    for name, fn in steps:
        try:
            logger.info("########## SLG suite: %s ##########", name)
            fn()
        except Exception:
            logger.exception("SLG suite step '%s' failed; continuing.", name)


def run_slg_chat(experiment: str):
    from inference.slg import SmallLanguageRouter
    router = SmallLanguageRouter(experts_location=experiment, experiment=experiment)
    router.chat()
