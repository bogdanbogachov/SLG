"""Fine-tuning module for language models."""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import EarlyStoppingCallback

from logging_config import logger
from config import CONFIG
from utils.path_utils import ensure_dir
from utils.model_loader import load_base_model_and_tokenizer
from utils.prompt_utils import apply_chat_template


def finetune(
    model_to_tune: str,
    adapter_name: str,
    data: str,
    experiment_number: str,
    slg: bool = False,
    train_limit: int = 0,
) -> None:
    """
    Fine-tune a language model with LoRA.

    Args:
        model_to_tune: Path to the base model directory
        adapter_name: Name for the adapter
        data: Path to JSON data file
        experiment_number: Experiment identifier
        slg: Whether this is for SLG (Small Language Router)
        train_limit: If >0, fine-tune on only this many examples (seeded random
            subset) for a quick smoke test. 0 = use the full training set.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found! Please ensure you have a CUDA-compatible GPU.")

    # Load model and tokenizer using utility function
    model, tokenizer = load_base_model_and_tokenizer(model_to_tune)

    # Load dataset
    dataset = load_dataset("json", data_files=data, split="train")
    if train_limit and train_limit > 0 and train_limit < len(dataset):
        # Quick-check subset: seeded so it is reproducible; shuffled first so a
        # pooled multi-expert file (qa_train) stays roughly balanced.
        full_n = len(dataset)
        dataset = dataset.shuffle(seed=int(CONFIG['seed'])).select(range(train_limit))
        logger.info("train_limit: fine-tuning '%s' on %d/%d examples.",
                    adapter_name, train_limit, full_n)
    logger.debug(f"Dataset after loading: {dataset}")
    logger.debug(f"Dataset shape: {dataset.shape}")

    # Get training config from CONFIG
    training_config = CONFIG['training']
    data_config = CONFIG['data']
    test_split_ratio = data_config['test_split_ratio']
    max_length = data_config['max_length']

    # Define a function to apply the chat template
    def apply_chat_template_to_example(example):
        """Apply chat template to a dataset example."""
        from utils.prompt_utils import create_user_message, create_assistant_message
        
        messages = [
            create_user_message(example['question']),
            create_assistant_message(example['answer'])
        ]
        prompt = apply_chat_template(messages, tokenizer, add_generation_prompt=False)
        return {"prompt": prompt}

    # Apply the chat template function to the dataset
    new_dataset = dataset.map(apply_chat_template_to_example)
    new_dataset = new_dataset.train_test_split(test_split_ratio)
    logger.debug(f"Dataset after splitting: {new_dataset}")

    if tokenizer.pad_token is None:
        # Use an existing special token as the pad token to avoid resizing the
        # embeddings. Llama exposes reserved tokens; other families (e.g. Qwen)
        # do not, so fall back to eos there.
        if "<|reserved_special_token_15|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|reserved_special_token_15|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the data
    def tokenize_function(example):
        """Tokenize example with proper label handling."""
        tokens = tokenizer(
            example['prompt'],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        # Set padding token labels to -100 to ignore them in loss calculation
        tokens['labels'] = [
            -100 if token == tokenizer.pad_token_id else token
            for token in tokens['input_ids']
        ]
        return tokens

    # Apply tokenize_function to each row
    tokenized_dataset = new_dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'prompt'])

    # Get LoRA config from CONFIG
    lora_config = training_config['lora']
    peft_params = LoraConfig(
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        r=lora_config['r'],
        task_type='CAUSAL_LM'
    )

    # Get learning rate and label smoothing from config
    learning_rate = training_config['learning_rate']
    label_smoothing_factor = training_config['label_smoothing_factor']

    # Model-size-aware batch sizes: a bigger model fills an 80GB GPU sooner, so it
    # takes a smaller per-device batch. Keys are chosen from the model dir name:
    # …_14b (Qwen-14B), …_8b (Llama-8B), …_3b (Qwen-3B), else the default (1B).
    # Size is matched before family: the bare "qwen" fallback below would otherwise
    # hand a 14B model the 3B batch and OOM.
    name = os.path.basename(os.path.normpath(model_to_tune)).lower()
    if "14b" in name:
        suffix = "_14b"
    elif "8b" in name:
        suffix = "_8b"
    elif "3b" in name or "qwen" in name:
        suffix = "_3b"
    else:
        suffix = ""
    per_device_train_batch_size = training_config[f'per_device_train_batch_size{suffix}']
    per_device_eval_batch_size = training_config[f'per_device_eval_batch_size{suffix}']
    logger.info(
        "Fine-tune batch sizes for %s: train=%d eval=%d (grad_accum=%d)",
        adapter_name, per_device_train_batch_size, per_device_eval_batch_size,
        training_config['gradient_accumulation_steps'],
    )

    # Create checkpoint directory
    paths_config = CONFIG['paths']
    checkpoints_dir = paths_config['checkpoints']
    checkpoint_dir = os.path.join(checkpoints_dir, experiment_number, adapter_name)
    ensure_dir(checkpoint_dir)

    # Get logging directory from config
    log_dir = CONFIG['logging']['log_dir']
    logging_dir = os.path.join(log_dir, experiment_number)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=training_config['num_epochs'],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=training_config['logging_steps'],
        seed=int(CONFIG['seed']),
        fp16=True,
        use_cpu=False,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        log_level="info",
        logging_dir=logging_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=training_config['weight_decay'],
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=training_config['max_grad_norm'],
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type='cosine',
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        optim='adamw_torch',
        label_smoothing_factor=label_smoothing_factor,
        load_best_model_at_end=True,
        save_total_limit=training_config['save_total_limit']
    )

    # Initialize Trainer
    early_stopping_patience = training_config['early_stopping_patience']
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_params,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    # Train the model (keep full model on CUDA; load_best_model_at_end reload can leave some weights on CPU)
    trainer.train()
    trainer.model.to(torch.device("cuda"))
    trainer.evaluate()

    # Save the model and tokenizer
    experiments_dir = CONFIG['paths']['experiments']
    
    if slg:
        slg_subdir = CONFIG.get('slg', {}).get('slg_dir', 'slg')
        slg_dir = os.path.join(experiments_dir, experiment_number, slg_subdir)
        ensure_dir(slg_dir)
        save_path = os.path.join(slg_dir, adapter_name)
        
        trainer.model.save_pretrained(save_path, save_adapter=True)
        tokenizer.save_pretrained(save_path)
        
        training_log_path = os.path.join(save_path, 'training_log.txt')
        with open(training_log_path, "a") as log_file:
            log_file.write(str(trainer.state.log_history))
    else:
        experiment_dir = os.path.join(experiments_dir, experiment_number)
        ensure_dir(experiment_dir)
        save_path = os.path.join(experiment_dir, adapter_name)
        
        trainer.model.save_pretrained(save_path, save_adapter=True)
        tokenizer.save_pretrained(save_path)
        
        training_log_path = os.path.join(save_path, 'training_log.txt')
        with open(training_log_path, "a") as log_file:
            log_file.write(str(trainer.state.log_history))

    return None
