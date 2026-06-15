#!/usr/bin/env python3

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from transformers import Trainer as HFTrainer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.nn as nn

# ====== CONFIG ======

BASE_MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"

DATA_DIR = "intrigue_clean"
OUTPUT_DIR = "intrigue-ss-model"

MAX_SEQ_LEN = 512        # shorter to keep memory + speed reasonable
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 5
LEARNING_RATE = 5e-6
IGNORE_INDEX = -100
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ====== LOAD DATA ======

def load_corpus(data_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    texts = []
    for path in tqdm(paths, desc="Loading newsletters"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        lines = txt.splitlines()
        drop = 0
        cleaned_lines = []
        for ln in lines:
            if drop < 3 and ln.strip():
                drop += 1
                continue
            cleaned_lines.append(ln)
        cleaned = "\n".join(cleaned_lines).strip()
        if cleaned:
            texts.append(cleaned)
    return texts


print(f"[INFO] Loading corpus from {DATA_DIR}...")
all_texts = load_corpus(DATA_DIR)
print(f"[INFO] Loaded {len(all_texts)} documents.")

if not all_texts:
    raise RuntimeError("No documents loaded. Check DATA_DIR and file paths.")

# ====== TOKENIZER & CHUNKING ======

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    use_fast=True,
    fix_mistral_regex=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def chunk_texts(texts: List[str], max_seq_len: int) -> List[str]:
    joined = "\n\n" + ("\n\n".join(texts)) + "\n\n"
    enc = tokenizer(
        joined,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"][0]

    chunks = []
    for i in range(0, len(input_ids), max_seq_len):
        chunk_ids = input_ids[i : i + max_seq_len]
        if len(chunk_ids) < 64:
            break
        text_chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(text_chunk)

    return chunks

print("[INFO] Tokenizing and chunking corpus...")
chunks = chunk_texts(all_texts, MAX_SEQ_LEN)
print(f"[INFO] Created {len(chunks)} training chunks of up to {MAX_SEQ_LEN} tokens.")

dataset = Dataset.from_list([{"text": c} for c in chunks])

def tokenize_fn(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )

    input_ids = enc["input_ids"]

    labels = []
    for seq in input_ids:
        seq_labels = []
        for tid in seq:
            if tid == tokenizer.pad_token_id:
                seq_labels.append(IGNORE_INDEX)
            else:
                seq_labels.append(tid)
        labels.append(seq_labels)

    enc["labels"] = labels
    return enc

print("[INFO] Tokenizing dataset...")
tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"],
)

print("[INFO] Vocab size:", tokenizer.vocab_size)
for i in range(min(3, len(tokenized))):
    ex = tokenized[i]
    labels = ex["labels"]
    non_ignored = [t for t in labels if t != IGNORE_INDEX]
    if non_ignored:
        print(
            f"[DEBUG] Example {i} labels: min={min(non_ignored)}, "
            f"max={max(non_ignored)}, len={len(labels)}"
        )

# ====== LOAD MODEL + LORA (Accelerate will handle multi-GPU) ======

print(f"[INFO] Loading base model {BASE_MODEL_NAME} ...")

if torch.cuda.is_available():
    dtype = torch.bfloat16
else:
    dtype = torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=dtype,
)

base_model.config.use_cache = False
if hasattr(base_model, "gradient_checkpointing_enable"):
    base_model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.config.ignore_index = IGNORE_INDEX

print("[INFO] Trainable parameters:")
model.print_trainable_parameters()

# ====== CUSTOM TRAINER WITH MANUAL LOSS ======

class DebugTrainer(HFTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if return_outputs:
            return loss, outputs
        return loss

# ====== TRAINING ARGS & TRAINER ======

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=False,
    bf16=torch.cuda.is_available(),
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    report_to="none",
)

trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

print("[INFO] Starting training with Accelerate (2 GPUs)...")
trainer.train()
print("[INFO] Training complete. Saving LoRA adapter + tokenizer...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("[INFO] Saved Intrigue NeMo adapter to", OUTPUT_DIR)
