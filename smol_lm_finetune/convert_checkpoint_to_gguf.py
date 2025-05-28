#!/usr/bin/env python3
import os
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Parse arguments
parser = argparse.ArgumentParser(description="Convert checkpoint to GGUF")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
parser.add_argument("--gguf_dir", type=str, required=True, help="Output directory for GGUF model")
args = parser.parse_args()

# Get checkpoint path
checkpoint_path = args.checkpoint
output_dir = args.output_dir
gguf_dir = args.gguf_dir

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(gguf_dir, exist_ok=True)

# Load base model and tokenizer
print("Loading base model...")
base_model_id = "HuggingFaceTB/SmolLM-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    device_map="auto"
)

# Load and apply the LoRA adapter weights
print(f"Loading adapter from {checkpoint_path}...")
model = PeftModel.from_pretrained(model, checkpoint_path)

# Merge LoRA weights with base model
print("Merging weights...")
merged_model = model.merge_and_unload()

# Save the merged model
print(f"Saving merged model to {output_dir}...")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Convert to GGUF using llama.cpp
print("Converting to GGUF...")
# Adjust the path to your llama.cpp directory
LLAMA_CPP_PATH = "../../llama.cpp"  # Adjust this path as needed
CONVERT_SCRIPT = os.path.join(LLAMA_CPP_PATH, "convert-hf-to-gguf.py")

if not os.path.exists(CONVERT_SCRIPT):
    print(f"Error: {CONVERT_SCRIPT} not found. Please check the path.")
    sys.exit(1)

output_gguf_file = os.path.join(gguf_dir, "smollm_finetuned_interim.gguf")
import subprocess
command = [
    "python", CONVERT_SCRIPT,
    output_dir,
    "--outfile", output_gguf_file,
    "--outtype", "f16"
]
print(f"Running command: {' '.join(command)}")
subprocess.run(command, check=True)

# Quantize the model (optional)
quantized_gguf_file = os.path.join(gguf_dir, "smollm_finetuned_interim_q4_k_m.gguf")
quantize_executable = os.path.join(LLAMA_CPP_PATH, "quantize")
if os.path.exists(quantize_executable):
    print("Quantizing model...")
    quantize_command = [
        quantize_executable,
        output_gguf_file,
        quantized_gguf_file,
        "q4_k_m"
    ]
    subprocess.run(quantize_command, check=True)
    print(f"Quantized GGUF saved to: {quantized_gguf_file}")
else:
    print(f"Warning: Quantize executable not found at {quantize_executable}")
    print(f"GGUF model saved to: {output_gguf_file}")

print("Conversion complete!")
