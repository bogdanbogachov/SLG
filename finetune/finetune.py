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
    orchestrator: bool = False
) -> None:
    """
    Fine-tune a language model with LoRA.
    
    Args:
        model_to_tune: Path to the base model directory
        adapter_name: Name for the adapter
        data: Path to JSON data file
        experiment_number: Experiment identifier
        slg: Whether this is for SLG (Small Language Graph)
        orchestrator: Whether this is an orchestrator model
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found! Please ensure you have a CUDA-compatible GPU.")

    # Load model and tokenizer using utility function
    model, tokenizer = load_base_model_and_tokenizer(model_to_tune)

    # Load dataset
    dataset = load_dataset("json", data_files=data, split="train")
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
        
        if orchestrator:
            messages = [
                create_user_message(
                    f"Analyze this question and find an appropriate expert who can answer it: {example['question']}"
                ),
                create_assistant_message(example['title'])
            ]
            prompt = apply_chat_template(messages, tokenizer, add_generation_prompt=False)
            return {"prompt": prompt}
        else:
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
        # Set an existing special token as a padding token, this way we can avoid model resizing
        tokenizer.pad_token = "<|reserved_special_token_15|>"

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
    if orchestrator:
        orchestrator_config = training_config['orchestrator']
        learning_rate = orchestrator_config['learning_rate']
        label_smoothing_factor = orchestrator_config['label_smoothing_factor']
    else:
        learning_rate = training_config['learning_rate']
        label_smoothing_factor = training_config['label_smoothing_factor']

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
        fp16=True,
        report_to="tensorboard",
        log_level="info",
        logging_dir=logging_dir,
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
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

    # Train the model
    trainer.train()
    trainer.evaluate()

    # Save the model and tokenizer
    experiments_dir = CONFIG['paths']['experiments']
    
    if slg:
        slg_dir = os.path.join(experiments_dir, experiment_number, 'slg')
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
