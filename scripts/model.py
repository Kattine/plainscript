"""Model helpers for QLoRA training and inference.

This file handles dtype/device choices and adapter loading for before/after tests.
"""

import os
import inspect

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_DIR = os.path.join("models", "qlora-adapter")
PROCESSED_DIR = os.path.join("data", "processed")

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Rewrite the following medical text "
    "into plain language that a patient with no medical background can "
    "understand. Preserve all key findings and conclusions. Do not add "
    "information not present in the original text."
)


def _use_bf16() -> bool:
    """Use bf16 only on Ampere+ GPUs; otherwise stay in fp32."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def get_compute_dtype() -> torch.dtype:
    """Return bf16 on Ampere+, else fp32."""
    return torch.bfloat16 if _use_bf16() else torch.float32


def get_bnb_config() -> BitsAndBytesConfig:
    """Build the 4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_compute_dtype(),
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    """Build the LoRA config for attention projection layers."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )


def train_model(
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1536,
    output_dir: str = ADAPTER_DIR,
):
    """Fine-tune Qwen2-1.5B-Instruct with QLoRA and save the adapter."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        dtype=get_compute_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())

    # Keep trainable weights in fp32 for stable training on T4.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(PROCESSED_DIR, "train.jsonl"),
            "validation": os.path.join(PROCESSED_DIR, "val.jsonl"),
        },
    )

    # Newer TRL uses max_length; older versions still use max_seq_length.
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    seq_len_key = "max_length" if "max_length" in sft_params else "max_seq_length"

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=_use_bf16(),
        fp16=False,
        report_to="none",
        seed=42,
        **{seq_len_key: max_seq_length},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


def _get_device() -> str:
    """Pick the best available inference device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model():
    """Load the base model and tokenizer for baseline outputs."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = _get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


def load_finetuned_model(adapter_path: str = ADAPTER_DIR):
    """Load the model with the trained LoRA adapter for inference."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = _get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_plain_summary(
    text: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    """Generate a plain-language rewrite for one medical input text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    train_model()
