# Import libraries
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


# Load existing dataset from Hugging Face
dataset = load_dataset("dprashar/npc_dialogue_rpg_quests", split="train")

# Print dataset structure
print("Dataset columns:", dataset.column_names)
print("Sample data:", dataset[0])

# Preprocess the data - adjust based on actual column names
def preprocess_function(examples):
    # Check what columns are actually available and adjust accordingly
    # For example, if columns are "npc_type" and "text" instead:
    if "npc_type" in examples and "text" in examples:
        return {
            "text": [f"Character Type: {npc_type} Dialogue: {text}"
                    for npc_type, text in zip(examples["npc_type"], examples["text"])]
        }
    # If there's just "text" with dialogue content
    elif "text" in examples:
        return {"text": examples["text"]}
    # Fallback option
    else:
        # Use whatever columns are available
        # You'll need to adapt this based on the actual structure
        print("Available columns:", examples.keys())
        return {"text": examples[list(examples.keys())[0]]}

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]


# Initialize tokenizer and model
model_name = "distilgpt2"  # Small enough to run on Colab's free GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if GPU is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Set tokenizer parameters
tokenizer.pad_token = tokenizer.eos_token
max_length = 128

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments - Updated to use GPU
training_args = TrainingArguments(
    output_dir="./results",
    eval_steps=500,
    save_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    # Add the following parameters for GPU training
    fp16=True,  # Use mixed precision training
    no_cuda=False,  # Ensure CUDA is used if available
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model_path = "./fine_tuned_dialogue_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

