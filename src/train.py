"""
QLoRA SFT training for logistics CX extraction (Qwen2.5-3B + Unsloth).

Refactored for TRL v0.25+ and current Transformers:
- TRL: Use SFTConfig (not TrainingArguments) for SFTTrainer; SFT-specific options
  (max_length, dataset_text_field, packing, dataset_num_proc) live in SFTConfig.
- Transformers: SFTConfig subclasses TrainingArguments, so all standard training
  args (learning_rate, max_steps, bf16, etc.) are passed there.
- SFTTrainer accepts args=SFTConfig | TrainingArguments; processing_class is the
  tokenizer (replaces deprecated tokenizer= for consistency with processor-based VLMs).

Unsloth compatibility (Transformers >= 5.0): Unsloth's patched SFTTrainer re-builds
SFTConfig from args.to_dict(). Their SFTConfig __init__ does not accept push_to_hub_token
(Unsloth only strips it when Transformers < 5.0). We use UnslothSafeSFTConfig so
to_dict() omits Hub-related keys and the re-instantiation succeeds.
"""

# Unsloth must be imported before trl/transformers/peft so its patches apply.
import unsloth  # noqa: F401

import os
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Keys to strip from to_dict() so Unsloth's SFTConfig(**dict_args) accepts the result
# (Unsloth only pops push_to_hub_token when Transformers < 5.0; we support 5.x)
_UNSLOTH_STRIP_KEYS = frozenset({"push_to_hub_token"})


class UnslothSafeSFTConfig(SFTConfig):
    """SFTConfig that omits Hub-related keys in to_dict() for Unsloth + Transformers 5.x."""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        for k in _UNSLOTH_STRIP_KEYS:
            d.pop(k, None)
        return d

REPO_ROOT = Path(__file__).resolve().parent.parent


def train_logistics_model(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_save_dir: Optional[Path] = None,
):
    print("🚀 Initializing Unsloth for 8GB VRAM QLoRA Training...")

    # Paths (work from any CWD)
    data_path = data_path or REPO_ROOT / "data" / "processed" / "golden_training_dataset.jsonl"
    output_dir = output_dir or REPO_ROOT / "outputs"
    model_save_dir = model_save_dir or REPO_ROOT / "models" / "qwen-logistics-lora"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # -------------------------------------------------------------------------
    # 1. Model config (Unsloth 4-bit for 8GB VRAM)
    # -------------------------------------------------------------------------
    max_seq_length = 2048  # Enough for 10–14 turn transcripts
    dtype = None  # Auto-detects fp16 for e.g. RTX 3070 Ti
    load_in_4bit = True  # Critical for fitting 3B in 8GB VRAM

    # -------------------------------------------------------------------------
    # 2. Load base model and tokenizer (Unsloth)
    # -------------------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Apply ChatML so our {"role": "user"/"assistant"} JSON matches Qwen's format.
    # TRL can also apply chat templates from a conversational "messages" column;
    # we pre-format to "text" here so we control the exact template (Unsloth chatml).
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    # -------------------------------------------------------------------------
    # 3. LoRA adapters (train a small subset of params)
    # -------------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # -------------------------------------------------------------------------
    # 4. Dataset: load JSONL and format as single "text" per example
    # -------------------------------------------------------------------------
    # Our JSONL has "messages" (system / user / assistant). We convert to
    # a single string via the tokenizer's chat template so SFTTrainer can use
    # dataset_text_field="text" (SFTConfig). Alternatively we could pass
    # raw "messages" and let TRL apply the template; we pre-format for explicit
    # control and compatibility with this tokenizer's chatml.
    print("📦 Loading the Golden Dataset...")
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # -------------------------------------------------------------------------
    # 5. SFTConfig (TRL v0.25+): training + SFT-specific options
    # -------------------------------------------------------------------------
    # Use UnslothSafeSFTConfig so to_dict() strips push_to_hub_token (and hub_token);
    # Unsloth's SFTConfig re-built from that dict then accepts all keys.
    sft_config = UnslothSafeSFTConfig(
        # --- SFT-specific (TRL) ---
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=2,
        # --- Standard training (Transformers TrainingArguments) ---
        output_dir=str(output_dir),
        eval_strategy="no",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=150,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",  # 8-bit Adam (bitsandbytes) for VRAM savings
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
    )

    # -------------------------------------------------------------------------
    # 6. SFTTrainer (TRL): model + config + dataset + processing_class
    # -------------------------------------------------------------------------
    # processing_class is the tokenizer (TRL naming for tokenizer/processor).
    # Do not pass dataset_text_field/max_seq_length to the trainer; they belong in SFTConfig.
    print("🔥 Starting the Training Loop...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # -------------------------------------------------------------------------
    # 7. Train and save LoRA adapters
    # -------------------------------------------------------------------------
    trainer_stats = trainer.train()

    print("\n✅ Training Complete! Saving LoRA adapters...")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))
    print(f"🎉 Done! Model saved in {model_save_dir}/")
    print(f"Training stats: {trainer_stats}")


if __name__ == "__main__":
    train_logistics_model()
