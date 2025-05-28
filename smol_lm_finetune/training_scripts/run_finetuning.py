# training_scripts/run_finetuning.py
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration ---
MODEL_ID = "HuggingFaceTB/SmolLM-360M-Instruct"

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_datasets/tokenizable_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
GGUF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "gguf_model")

# Training parameters - TUNE THESE CAREFULLY FOR 4GB VRAM
MICRO_BATCH_SIZE = 1 # Try 1 or 2.
GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch size = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-4 # Common for LoRA
NUM_EPOCHS = 1 # Start with 1 epoch, increase if needed
MAX_SEQ_LENGTH = 512 # Max sequence length. Reduce if OOM. SmolLM might have a smaller context window. Check its config.
                     # For SmolLM-360M, context length is typically 2048.
                     # If 512 causes OOM with QLoRA, try 256.
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 50 # How often to evaluate (if you have an eval set)
SAVE_STEPS = 100 # How often to save checkpoints

# QLoRA config
USE_QLORA = True # Set to True to enable QLoRA
LORA_R = 16 # LoRA attention dimension (rank)
LORA_ALPHA = 32 # LoRA alpha
LORA_DROPOUT = 0.05
# SmolLM specific: target modules might need adjustment.
# Common ones: "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
# You may need to inspect the model structure to find suitable Linear layers.
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    # "gate_proj", # Add if present and makes sense
    # "up_proj",   # Add if present and makes sense
    # "down_proj"  # Add if present and makes sense
    # "lm_head" # Sometimes targeted, but often not for LoRA on CausalLMs unless specifically needed.
]

def main(resume_from_checkpoint=None):
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # --- 1. Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Set padding token if not set. For GPT-like models, eos_token is often used.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    quantization_config = None
    if USE_QLORA:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16 if bf16 not supported well
            bnb_4bit_use_double_quant=True,
        )
        print("Using QLoRA with 4-bit quantization.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config if USE_QLORA else None, # Only apply if USE_QLORA
        device_map="auto", # Automatically distribute layers. For single GPU, effectively "cuda:0"
                           # device_map={"": 0} # Explicitly for single GPU
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16 # If not using bitsandbytes for loading, but want bfloat16. Not for 4GB.
    )
    model.config.use_cache = False # Important for training, disable for PEFT

    # --- 2. PEFT Configuration (LoRA/QLoRA) ---
    if USE_QLORA:
        print("Preparing model for k-bit training and applying PEFT...")
        # model = prepare_model_for_kbit_training(model) # Already handled by bitsandbytes if loading in 4/8bit
        model = prepare_model_for_kbit_training(model)

        # Filter target modules to only those present in the model
        model_modules = [name for name, module in model.named_modules()]
        actual_target_modules = [m for m in LORA_TARGET_MODULES if any(m in layer_name for layer_name in model_modules)]
        if not actual_target_modules:
            print(f"Warning: None of the LORA_TARGET_MODULES ({LORA_TARGET_MODULES}) were found in the model. PEFT may not be effective.")
            print("Available Linear layers in the model (first few):")
            count = 0
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    print(name)
                    count += 1
                    if count > 10: break
        else:
            print(f"Applying LoRA to modules: {actual_target_modules}")


        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=actual_target_modules,
            bias="none", # or "all" or "lora_only"
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif not USE_QLORA: # If not QLoRA, but still want to try fitting full model (unlikely on 4GB)
         # If you were to fine-tune all parameters (not recommended for 4GB VRAM):
         print("Attempting full model fine-tuning (requires significant VRAM).")
         pass # No PEFT model wrapping


    # --- 3. Load and Preprocess Dataset ---
    print(f"Loading processed dataset from {PROCESSED_DATA_PATH}")
    tokenized_dataset = load_from_disk(PROCESSED_DATA_PATH)

    # Ensure it's a Dataset, not DatasetDict, if preprocess_data.py saved it directly as Dataset
    if isinstance(tokenized_dataset, dict): # e.g. if you had 'train' and 'validation' splits
        train_dataset = tokenized_dataset['train']
        # eval_dataset = tokenized_dataset.get('validation') # Optional
    else: # If it's a single dataset (as per current preprocess_data.py)
        train_dataset = tokenized_dataset
        # You might want to split it here if you didn't before
        # train_test_split = train_dataset.train_test_split(test_size=0.1)
        # train_dataset = train_test_split['train']
        # eval_dataset = train_test_split['test']


    def tokenize_function(examples):
        # The preprocessor already created the full text string with prompt + response.
        # We just need to tokenize it.
        # The model will learn to predict the next token, so the 'labels' will be the input_ids shifted.
        # The DataCollatorForLanguageModeling will handle this shifting.
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length", # or False, DataCollator will handle padding
            max_length=MAX_SEQ_LENGTH,
            # return_overflowing_tokens=True, # Can be useful for very long texts
            # return_length=True,
        )
        return tokenized_output

    print("Tokenizing dataset...")
    # Important: ensure your "text" column (or whatever you named it) is used.
    # Adjust `batched` and `num_proc` based on your CPU cores and RAM.
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=max(1, os.cpu_count() // 2), # Use half of CPU cores
        remove_columns=train_dataset.column_names # Remove original text column
    )
    print(f"Tokenized training dataset features: {tokenized_train_dataset.features}")
    print(f"Sample tokenized input: {tokenized_train_dataset[0]['input_ids'][:50]}")

    # Data collator will dynamically pad batches and create labels for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # --- 4. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch", # Paged optimizer for QLoRA
        optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch", # paged_adamw_8bit can be tried if adamw_torch OOMs with QLoRA
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        fp16=True,  # Use mixed precision if GPU supports it (most NVIDIA GPUs from past ~5 years do)
                    # bf16=True if your GPU supports bfloat16 (Ampere and newer)
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3, # Keep only the last 3 checkpoints
        # evaluation_strategy="steps" if eval_dataset else "no",
        # eval_steps=EVAL_STEPS if eval_dataset else None,
        # load_best_model_at_end=True if eval_dataset else False,
        report_to="tensorboard",
        # ddp_find_unused_parameters=False, # Set if using DDP and encountering issues
        dataloader_num_workers = 2, # Or 1, adjust based on system
        gradient_checkpointing=True, # Saves memory at the cost of a bit of slower training. VERY helpful.
        # resume_from_checkpoint=resume_from_checkpoint # Handled by Trainer if path is provided
    )

    # --- 5. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        # eval_dataset=tokenized_eval_dataset if eval_dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 6. Train ---
    print("Starting training...")
    # The `resume_from_checkpoint` argument to `trainer.train()` can be a boolean (True to use latest in output_dir)
    # or a path to a specific checkpoint.
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # --- 7. Save Final Model & Metrics ---
    print("Training finished. Saving final model and metrics.")
    trainer.save_model(FINAL_MODEL_DIR) # Saves LoRA adapters and tokenizer
    # To save the full model (if not using PEFT or after merging PEFT):
    # model.save_pretrained(FINAL_MODEL_DIR)
    # tokenizer.save_pretrained(FINAL_MODEL_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Saves optimizer, scheduler, RNG states etc.

    print(f"Final PEFT adapters (if used) saved to: {FINAL_MODEL_DIR}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"Logs saved in: {LOG_DIR}")

    # --- 8. (Optional but Recommended) Merge LoRA adapters and save full model if using PEFT ---
    if USE_QLORA:
        print("Merging LoRA adapters with base model and saving...")
        # Reload the base model in full precision (or bf16/fp16 if desired)
        # Important: For merging, the base model should not be quantized initially for best results
        # However, if you loaded with quantization_config, you might need to handle this carefully.
        # Let's try to merge onto the quantized model first, then explore full precision merge if issues.

        # Deactivate adapter, load base model, then load adapter and merge
        # This part can be tricky with models loaded in 4-bit.
        # The PEFT library's `merge_and_unload` is the way to go.
        try:
            merged_model = model.merge_and_unload()
            merged_model_dir = os.path.join(OUTPUT_DIR, "final_merged_model")
            merged_model.save_pretrained(merged_model_dir)
            tokenizer.save_pretrained(merged_model_dir)
            print(f"Merged model saved to {merged_model_dir}")
            # Now, this `merged_model_dir` is what you'd convert to GGUF
            return merged_model_dir # Return path to merged model for GGUF conversion
        except Exception as e:
            print(f"Could not merge LoRA adapters directly: {e}")
            print("GGUF conversion will need to use the base model + LoRA adapters separately if supported by the conversion script,")
            print(f"or you can try to perform merging manually. For now, GGUF step will target {FINAL_MODEL_DIR} (adapters).")
            return FINAL_MODEL_DIR # Fallback to adapter directory

    return FINAL_MODEL_DIR # If not using QLoRA

def convert_to_gguf(model_path, output_dir):
    """
    Converts the fine-tuned model (ideally merged) to GGUF format using llama.cpp.
    """
    print(f"\n--- Attempting GGUF Conversion for model at: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Skipping GGUF conversion.")
        return

    # llama.cpp path (assuming it's cloned sibling to smol_lm_finetune)
    LLAMA_CPP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp"))
    CONVERT_SCRIPT = os.path.join(LLAMA_CPP_PATH, "convert-hf-to-gguf.py") # Or convert.py for older llama.cpp

    if not os.path.exists(CONVERT_SCRIPT):
        # Fallback to older script name if the new one isn't found
        CONVERT_SCRIPT = os.path.join(LLAMA_CPP_PATH, "convert.py")
        if not os.path.exists(CONVERT_SCRIPT):
            print(f"llama.cpp conversion script not found at {CONVERT_SCRIPT} or its alternative. Please ensure llama.cpp is compiled.")
            print("Skipping GGUF conversion.")
            return

    os.makedirs(output_dir, exist_ok=True)
    output_gguf_file = os.path.join(output_dir, "smollm_finetuned.gguf")

    # Common quantization types for GGUF: q4_0, q4_k_m, q5_k_m, q8_0
    # q4_k_m is often a good balance.
    # The script might have a --outfile argument or write to a fixed name.
    # `convert-hf-to-gguf.py` usage: python convert-hf-to-gguf.py <model_dir> --outfile <output.gguf> --outtype <type>
    # `convert.py` (older) usage might differ, often: python convert.py <model_dir> --outtype f16 (then quantize with ./quantize)

    print(f"Using llama.cpp convert script: {CONVERT_SCRIPT}")
    # Prioritize convert-hf-to-gguf.py
    if "convert-hf-to-gguf.py" in CONVERT_SCRIPT:
        # Try f16 first for broadest compatibility, then quantize later with main llama.cpp binary if needed
        # Or directly try a quantized type like Q4_K_M if the script supports it well.
        # For simplicity, let's try f16 conversion here, then you can quantize it using the `quantize` tool in llama.cpp
        # command = [
        #     "python", CONVERT_SCRIPT,
        #     model_path,
        #     "--outfile", output_gguf_file,
        #     "--outtype", "f16" # Common options: f16, f32, q8_0, q4_0, q4_1, q5_0, q5_1
        # ]
        # A more robust approach for `convert-hf-to-gguf.py` for QLoRA models:
        # It often needs the original model path if adapters are separate.
        # If we have a merged model, it's simpler.
        command = [
            "python", CONVERT_SCRIPT,
            model_path, # Path to the merged model directory
            "--outfile", output_gguf_file,
            "--outtype", "f16" # Start with f16, then quantize
        ]
        print(f"Executing GGUF conversion (f16): {' '.join(command)}")
        try:
            import subprocess
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            print("GGUF conversion (f16) successful!")
            print(process.stdout)
            print(f"GGUF f16 model saved to: {output_gguf_file}")

            # Now, attempt quantization using ./quantize from llama.cpp
            quantized_gguf_file = os.path.join(output_dir, "smollm_finetuned_q4_K_M.gguf")
            quantize_executable = os.path.join(LLAMA_CPP_PATH, "quantize")
            if os.path.exists(quantize_executable):
                quantize_command = [
                    quantize_executable,
                    output_gguf_file, # Input f16 GGUF
                    quantized_gguf_file, # Output quantized GGUF
                    "Q4_K_M" # Common quantization type
                ]
                print(f"Executing GGUF quantization: {' '.join(quantize_command)}")
                process_quant = subprocess.run(quantize_command, check=True, capture_output=True, text=True)
                print("GGUF quantization successful!")
                print(process_quant.stdout)
                print(f"Quantized GGUF (Q4_K_M) model saved to: {quantized_gguf_file}")
            else:
                print(f"llama.cpp quantize executable not found at {quantize_executable}. Skipping quantization step.")
                print(f"You can manually quantize {output_gguf_file} later using llama.cpp.")

        except subprocess.CalledProcessError as e:
            print("Error during GGUF conversion or quantization:")
            print(e.stderr)
            print(e.stdout)
            print("GGUF conversion/quantization failed.")
        except FileNotFoundError:
             print(f"Python executable or script not found. Ensure Python is in PATH and script exists.")

    elif "convert.py" in CONVERT_SCRIPT: # Older script
        # The older convert.py usually converts to an intermediate GGUF (often FP16 or FP32)
        # and then you use the 'quantize' binary from llama.cpp.
        # python convert.py models/mymodel/ --outtype f16 --outfile models/mymodel/ggml-model-f16.gguf
        command = [
            "python", CONVERT_SCRIPT,
            model_path,
            "--outtype", "f16", # f16, f32 are common starting points
            "--outfile", output_gguf_file
        ]
        print(f"Executing GGUF conversion (older script): {' '.join(command)}")
        # Run command (omitted for brevity, similar to above subprocess call)
        print("GGUF conversion with older script might require manual quantization step using `llama.cpp/quantize`.")

    else:
        print("Unknown GGUF conversion script. Please check llama.cpp.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM model.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from, or 'auto' to resume from latest.")
    args = parser.parse_args()

    resume_path = None
    if args.resume:
        if args.resume.lower() == 'auto':
            # Try to find the latest checkpoint
            if os.path.exists(CHECKPOINT_DIR) and any(os.listdir(CHECKPOINT_DIR)):
                # `get_last_checkpoint` can find the latest checkpoint if using Trainer's default naming
                from transformers.trainer_utils import get_last_checkpoint
                latest_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)
                if latest_checkpoint:
                    resume_path = latest_checkpoint
                    print(f"Auto-resuming from latest checkpoint: {resume_path}")
                else:
                    print(f"No checkpoint found in {CHECKPOINT_DIR} for auto-resume.")
            else:
                print(f"Checkpoint directory {CHECKPOINT_DIR} is empty or doesn't exist. Starting fresh.")
        else:
            resume_path = args.resume
            print(f"Resuming from specified checkpoint: {resume_path}")


    final_model_directory = main(resume_from_checkpoint=resume_path)

    if final_model_directory:
        convert_to_gguf(final_model_directory, GGUF_OUTPUT_DIR)
    else:
        print("Main training function did not return a valid model directory. Skipping GGUF conversion.")