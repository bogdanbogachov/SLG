"""Layerwise residual-stream activation patching for selected SLG wins."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS_TO_PATCH = tuple(range(16))


def _read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return {r["qid"]: r for line in f if line.strip() for r in [json.loads(line)] if r.get("qid")}


def _model_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for candidate in (getattr(base, "model", None), base):
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError("Could not locate transformer residual layers on this model.")


def _capture_prompt_residuals(model, layers, ids, prompt_length):
    import torch
    captured = {}
    handles = []
    for idx, layer in enumerate(layers):
        def hook(_module, _inputs, output, layer_idx=idx):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[:, :prompt_length].detach().clone()
        handles.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _patched_logprob(model, layers, ids, prompt_length, answer_count, replacement):
    import torch
    from torch.nn import functional as F
    layer_idx, residual = replacement

    def patch(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, :prompt_length] = residual.to(device=hidden.device, dtype=hidden.dtype)
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, :prompt_length] = residual.to(device=hidden.device, dtype=hidden.dtype)
        return hidden

    handle = layers[layer_idx].register_forward_hook(patch)
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits
    finally:
        handle.remove()
    target = ids[:, prompt_length:prompt_length + answer_count]
    prediction_logits = logits[:, prompt_length - 1:prompt_length + answer_count - 1]
    return float(F.log_softmax(prediction_logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).sum().item())


def _logprob(model, ids, prompt_length, answer_count):
    import torch
    from torch.nn import functional as F
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits
    target = ids[:, prompt_length:prompt_length + answer_count]
    prediction_logits = logits[:, prompt_length - 1:prompt_length + answer_count - 1]
    return float(F.log_softmax(prediction_logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).sum().item())


def _chat_ids(tokenizer, question, answer=None):
    from utils.prompt_utils import apply_chat_template, create_assistant_message, create_user_message
    prompt_text = apply_chat_template([create_user_message(question)], tokenizer, add_generation_prompt=True)
    messages = [create_user_message(question)]
    if answer is not None:
        messages.append(create_assistant_message(answer))
    full_text = apply_chat_template(messages, tokenizer, add_generation_prompt=answer is None)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids
    if full_ids.shape[1] < prompt_ids.shape[1] or not full_ids[0, :prompt_ids.shape[1]].equal(prompt_ids[0]):
        raise ValueError("Chat template prompt tokens do not prefix the teacher-forced sequence.")
    return prompt_ids, full_ids


def _select_qids(dataset, seed, results_root=None):
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    score_dir = results_root / dataset / f"seed_{seed}" / "scores"
    base = _read_jsonl(score_dir / "finetuned_3_2_1b.jsonl")
    slg = _read_jsonl(score_dir / "slg_finetuned_router.jsonl")
    candidates = []
    for qid in sorted(set(base) & set(slg)):
        if float(base[qid].get("ROUGE-L", 1.0)) <= 0.3 and float(slg[qid].get("ROUGE-L", 0.0)) >= 0.8:
            candidates.append(qid)
    if not candidates:
        return []
    return candidates[:200]


def _one_dataset(dataset, qids, limit, seed, completed_keys, partial_path,
                 results_root=None, data_root=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from inference.slg import SmallLanguageGraph
    from main import configure_bundle

    configure_bundle(dataset, seed)
    from config import CONFIG
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    cell = results_root / dataset / f"seed_{seed}"
    split_dir = data_root / dataset / "splits" / f"seed_{seed}"
    CONFIG["paths"].update({
        "experiments": str(cell), "answers": str(cell / "answers"),
        "checkpoints": str(cell / "checkpoints"),
        "split_by_title": str(split_dir / "by_title"),
    })
    CONFIG["files"].update({
        "qa": str(data_root / dataset / "source.json"),
        "qa_train": str(split_dir / "train.json"),
        "qa_test": str(split_dir / "test.json"),
    })
    test_rows = _read_json(split_dir / "test.json")
    by_qid = {r["qid"]: r for r in test_rows}
    work = [by_qid[qid] for qid in qids if qid in by_qid and (dataset, seed, qid) not in completed_keys]
    if not work:
        print(f"{dataset} seed {seed}: no selected activation-patching examples; skipping")
        return []
    base_path = Path(CONFIG["paths"]["downloaded_models"]) / CONFIG["paths"]["models"]["3_2_1b"]
    baseline_adapter = cell / CONFIG["adapters"]["finetuned_3_2_1b"]
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float16, local_files_only=True).to("cuda")
    model = PeftModel.from_pretrained(model, str(baseline_adapter), adapter_name="baseline", is_trainable=False)
    model.set_adapter("baseline")
    model.eval()
    layers = _model_layers(model)
    if len(layers) < len(LAYERS_TO_PATCH):
        raise ValueError(f"Expected at least {len(LAYERS_TO_PATCH)} layers, found {len(layers)}")

    # Cache tokenized records and expert paths, then select unrelated controls
    # with a matching prompt length so residual positions line up exactly.
    tokenized = {}
    title_to_expert = {}
    for row in test_rows:
        prompt, full = _chat_ids(tokenizer, row["question"], row["answer"])
        tokenized[row["qid"]] = (prompt, full)
        title_to_expert[row["title"]] = SmallLanguageGraph._expected_expert_id_from_title(row["title"])
    rng = random.Random(seed)
    control_for = {}
    for row in work:
        prompt_len = tokenized[row["qid"]][0].shape[1]
        choices = [r for r in test_rows if r["qid"] != row["qid"] and tokenized[r["qid"]][0].shape[1] == prompt_len]
        if not choices:
            # The position-aligned control can also use the common prefix.
            choices = [r for r in test_rows if r["qid"] != row["qid"]]
        control_for[row["qid"]] = rng.choice(choices) if choices else None

    records = []
    try:
        for index, row in enumerate(work[:limit], 1):
            qid = row["qid"]
            prompt, full = tokenized[qid]
            prompt_len = prompt.shape[1]
            n_answer = min(16, max(0, full.shape[1] - prompt_len))
            if prompt_len < 1 or n_answer < 1:
                continue
            ids = full.to("cuda")
            expert = title_to_expert[row["title"]]
            expert_path = cell / CONFIG["slg_formation"]["slg_dir"] / expert
            model.load_adapter(str(expert_path), adapter_name="slg_expert", is_trainable=False)
            model.set_adapter("slg_expert")
            slg_lp = _logprob(model, ids, prompt_len, n_answer)
            slg_residuals = _capture_prompt_residuals(model, _model_layers(model), ids, prompt_len)

            control = control_for[qid]
            if control is not None:
                control_prompt, _ = tokenized[control["qid"]]
                model.set_adapter("slg_expert")
                control_residuals = _capture_prompt_residuals(model, _model_layers(model), control_prompt.to("cuda"), control_prompt.shape[1])
            else:
                control_residuals = None
            model.set_adapter("baseline")
            base_lp = _logprob(model, ids, prompt_len, n_answer)
            denom = slg_lp - base_lp
            if abs(denom) < 1e-8:
                model.delete_adapter("slg_expert")
                continue
            record_start = len(records)
            for layer_idx in LAYERS_TO_PATCH:
                patched_lp = _patched_logprob(model, _model_layers(model), ids, prompt_len, n_answer, (layer_idx, slg_residuals[layer_idx]))
                control_lp = None
                if control_residuals is not None:
                    residual = control_residuals[layer_idx]
                    if residual.shape[1] != prompt_len:
                        if residual.shape[1] > prompt_len:
                            residual = residual[:, :prompt_len]
                        else:
                            pad = residual[:, -1:, :].expand(-1, prompt_len - residual.shape[1], -1)
                            residual = torch.cat((residual, pad), dim=1)
                    control_lp = _patched_logprob(model, _model_layers(model), ids, prompt_len, n_answer, (layer_idx, residual))
                records.append({"dataset": dataset, "seed": seed, "qid": qid, "layer": layer_idx,
                                "restoration": (patched_lp - base_lp) / denom,
                                "control_restoration": None if control_lp is None else (control_lp - base_lp) / denom,
                                "lp_base": base_lp, "lp_slg": slg_lp, "lp_patched": patched_lp,
                                "control_qid": None if control is None else control["qid"]})
            model.delete_adapter("slg_expert")
            with partial_path.open("a", encoding="utf-8") as f:
                for record in records[record_start:]:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            completed_keys.add((dataset, seed, qid))
            if index % 10 == 0:
                print(f"{dataset}: processed {index}/{min(limit, len(work))} selected qids")
    finally:
        del model
        torch.cuda.empty_cache()
    return records


def run(limit=200, seed=None, datasets=None, seeds=None, results_root=None,
        data_root=None, output_dir=None, qids_by_cell=None):
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    out = Path(output_dir) if output_dir is not None else ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "activation_patching.partial.jsonl"
    all_records = []
    if partial_path.exists():
        with partial_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip():
                        all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    group_counts = {}
    for record in all_records:
        key = (record["dataset"], int(record["seed"]), record["qid"])
        group_counts[key] = group_counts.get(key, 0) + 1
    complete_keys = {key for key, count in group_counts.items() if count == len(LAYERS_TO_PATCH)}
    all_records = [r for r in all_records if (r["dataset"], int(r["seed"]), r["qid"]) in complete_keys]
    # Remove partial/truncated question groups before resuming so the progress
    # file remains valid JSONL and completed qids are never duplicated.
    with partial_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    seeds = tuple(seeds) if seeds is not None else ((seed,) if seed is not None else (42, 53, 64))
    datasets = tuple(datasets) if datasets is not None else ("l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100", "cqad20")
    for current_seed in seeds:
        # Select successful SLG recoveries at 100% inflation, then reuse the
        # exact qids at every inflation level for the paired layerwise contrast.
        avia_qids = [] if qids_by_cell else _select_qids("s_avia_100", current_seed, results_root)
        if not avia_qids and not qids_by_cell:
            print(f"seed {current_seed}: no 100%-inflation qids meet the selection thresholds; skipping its avia inflation contrast")
        for dataset in datasets:
            qids = (qids_by_cell or {}).get((dataset, current_seed))
            if qids is None:
                qids = avia_qids if dataset in {"s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100"} else _select_qids(dataset, current_seed, results_root)
            if qids:
                all_records.extend(_one_dataset(dataset, qids, limit, current_seed, complete_keys, partial_path, results_root, data_root))
    if not all_records:
        print("No selected 1B-fails/SLG-succeeds examples were available; skipping the activation-patching figure.")
        return None
    csv_path = out / "activation_patching.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0]))
        writer.writeheader(); writer.writerows(all_records)
    _plot(all_records, out)
    return csv_path


def _plot(records, out):
    import matplotlib.pyplot as plt
    import numpy as np
    rng = random.Random(20261015)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    specs = (
        ("s_avia_25", "25% inflation", "restoration", "o-"),
        ("s_avia_100", "100% inflation", "restoration", "s-"),
        ("s_avia_100", "Unrelated-question control", "control_restoration", "x--"),
    )
    if not {"s_avia_25", "s_avia_100"}.issubset({r.get("dataset") for r in records}):
        dataset = records[0]["dataset"]
        specs = ((dataset, "SLG expert patch", "restoration", "o-"),
                 (dataset, "Unrelated-question control", "control_restoration", "x--"))
    for dataset, label, key, style in specs:
        xs, means, lows, highs = [], [], [], []
        for layer in LAYERS_TO_PATCH:
            values = [float(r[key]) for r in records if r["dataset"] == dataset and r["layer"] == layer and r.get(key) is not None]
            if not values:
                continue
            boots = []
            for _ in range(2000):
                boots.append(float(np.mean([rng.choice(values) for _ in values])))
            boots.sort()
            xs.append(layer); means.append(float(np.mean(values)))
            lows.append(boots[49]); highs.append(boots[1949])
        ax.plot(xs, means, style, label=label)
        ax.fill_between(xs, lows, highs, alpha=.15)
    ax.axhline(0, color="black", linewidth=.8, alpha=.5)
    ax.set(xlabel="Llama layer", ylabel="Restoration of first-16-token log probability")
    ax.set_xticks(LAYERS_TO_PATCH); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "activation_patching.pdf")
    fig.savefig(out / "activation_patching.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(run(args.limit, args.seed))


if __name__ == "__main__":
    main()
