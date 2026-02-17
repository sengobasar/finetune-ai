# src/train.py

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ----------------------------
# Config
# ----------------------------

MODEL_NAME = "microsoft/DialoGPT-small"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "model/finetuned_model"

EPOCHS = 3
BATCH_SIZE = 2
LR = 3e-5
MAX_LEN = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Load Model & Tokenizer
# ----------------------------

print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-style models need pad token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# ----------------------------
# Load Dataset
# ----------------------------

print("Loading dataset...")

dataset = load_dataset("json", data_files=DATA_PATH)


# ----------------------------
# Format Dataset
# ----------------------------

def format_example(example):
    text = (
        "User: " + example["prompt"] +
        "\nBot: " + example["response"] +
        tokenizer.eos_token
    )
    return {"text": text}


dataset = dataset.map(format_example)


# ----------------------------
# Tokenization
# ----------------------------

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )


tokenized = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names
)


# ----------------------------
# Data Collator (IMPORTANT)
# ----------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ----------------------------
# Training Arguments
# ----------------------------

print("Setting training arguments...")

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,

    learning_rate=LR,
    warmup_steps=50,
    weight_decay=0.01,

    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,

    fp16=torch.cuda.is_available(),

    report_to="none"
)


# ----------------------------
# Trainer
# ----------------------------

print("Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)


# ----------------------------
# Train
# ----------------------------

print("Starting training...")
trainer.train()


# ----------------------------
# Save
# ----------------------------

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete.")
