import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, get_peft_model
import os

# --- Configuration ---
# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf" # You can switch this to other models like "google/gemma-2b-it"

# Fine-tuned model name
new_model = "llama-2-7b-lab-notes"

# Dataset path
dataset_path = "../data/finetuning_data/dataset.json"

# Output directory for the fine-tuned model
output_dir = f"../models/output/{new_model}"

# --- Main Fine-Tuning Logic ---

def format_prompt(sample):
    """Formats the data sample into a single string."""
    return f"{sample['prompt']}{sample['completion']}"

def main():
    print("Starting the fine-tuning process...")

    # 1. Load the dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 2. Configure Quantization (for memory efficiency)
    # Using 4-bit quantization to fit larger models into memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Load the pre-trained model
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" # Automatically distribute the model across available GPUs
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 4. Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Configure PEFT (Parameter-Efficient Fine-Tuning) with LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Get the PEFT model
    peft_model = get_peft_model(model, lora_config)

    # 6. Set up Training Arguments
    print("Setting up training arguments...")
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1, # A single epoch is often enough for fine-tuning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # 7. Initialize the Trainer
    # Note: We are using a custom data collator that handles formatting.
    # For simplicity in this script, we'll use a lambda function.
    # In a more complex setup, you'd use a dedicated DataCollator.
    from trl import SFTTrainer

    print("Initializing the SFTTrainer...")
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="prompt", # This is a simplification; we format the whole text
        formatting_func=format_prompt, # Use our formatting function
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # 8. Start Training
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    # 9. Save the fine-tuned model
    print(f"Saving the fine-tuned model to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")

    print("\n--- Testing the fine-tuned model ---")
    prompt_text = "### Instruction:\nExtract the parameters for a projectile motion problem from the text and provide them in JSON format.\n\n### Input:\nThe ball was thrown at 10 m/s from a 2m ledge, at an angle of 15 degrees.\n\n### Response:"
    
    pipe = pipeline(task="text-generation", model=peft_model, tokenizer=tokenizer, max_length=200)
    result = pipe(prompt_text)
    print(result[0]['generated_text'])
    print("-------------------------------------\n")


if __name__ == "__main__":
    main()
