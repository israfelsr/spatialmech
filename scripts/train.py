import argparse
import os
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def freeze_vision_encoder(model):
    """
    Freeze the vision encoder parameters, only train the LLM.

    Args:
        model: Qwen2VLForConditionalGeneration model
    """
    # Freeze vision tower (visual encoder)
    if hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = False
        logger.info("Frozen vision encoder (visual)")

    # Freeze any vision-related modules
    for name, param in model.named_parameters():
        if 'visual' in name.lower() or 'vision' in name.lower():
            param.requires_grad = False
            logger.info(f"Frozen parameter: {name}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")


def setup_lora(model, use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    """
    Setup LoRA for efficient fine-tuning of the LLM.

    Args:
        model: Base model
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        Model with LoRA applied (if use_lora=True)
    """
    if not use_lora:
        return model

    # LoRA configuration targeting LLM attention layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_model_and_processor(model_path, use_lora=True, lora_r=8, lora_alpha=16):
    """
    Load Qwen2.5-VL model and processor from disk.

    Args:
        model_path: Path to the model directory
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha

    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading model from {model_path}")

    # Load processor (handles both text and images)
    processor = AutoProcessor.from_pretrained(model_path)

    # Load model with bfloat16 for efficiency
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Freeze vision encoder
    freeze_vision_encoder(model)

    # Setup LoRA if requested
    if use_lora:
        model = setup_lora(model, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)

    return model, processor


def preprocess_function(examples, processor, max_length=512):
    """
    Preprocess examples for training.

    This is a placeholder - you'll need to adapt this based on your actual dataset format.
    Assumes your dataset has 'image', 'question', and 'answer' fields.

    Args:
        examples: Batch of examples from dataset
        processor: Qwen2VL processor
        max_length: Maximum sequence length

    Returns:
        Processed batch with input_ids, attention_mask, labels
    """
    # This is a dummy implementation - adapt to your dataset structure
    images = examples.get("image", [])
    questions = examples.get("question", [])
    answers = examples.get("answer", [])

    # Format as conversation
    conversations = []
    for q, a in zip(questions, answers):
        # Qwen2.5-VL chat format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": a}]
            }
        ]
        conversations.append(conversation)

    # Process with the processor
    texts = processor.apply_chat_template(conversations, tokenize=False)

    # Tokenize
    batch = processor(
        text=texts,
        images=images if images else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Create labels (same as input_ids for causal LM)
    batch["labels"] = batch["input_ids"].clone()

    return batch


def main(args):
    # Load model and processor
    model, processor = load_model_and_processor(
        args.model_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Load dataset from disk
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = load_from_disk(args.data_path)

    # Split into train/eval if not already split
    if "train" not in dataset:
        logger.info("Splitting dataset into train/eval")
        dataset = dataset.train_test_split(test_size=args.eval_split)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else dataset.get("validation", None)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=args.save_total_limit,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=args.report_to,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
    )

    # Create preprocessing function with processor
    def preprocess_fn(examples):
        return preprocess_function(examples, processor, max_length=args.max_length)

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train dataset",
    )

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Preprocessing eval dataset",
        )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Save LoRA adapters separately if using LoRA
    if args.use_lora:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        logger.info(f"Saving LoRA adapters to {lora_output_dir}")
        model.save_pretrained(lora_output_dir)

    logger.info("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL model (LLM only)")

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="/leonardo_work/EUHPC_D27_102/compmech/models/Qwen2.5-VL-3B-Instruct",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the dataset directory (load_from_disk format)"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation if train/test split doesn't exist"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/qwen2.5vl_finetuned",
        help="Output directory for checkpoints and final model"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "none"],
        help="Reporting tool"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
