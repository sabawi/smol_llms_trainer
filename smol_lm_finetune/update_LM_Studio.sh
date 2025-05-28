#!/bin/bash
# Create a dedicated directory for your fine-tuned model
MODEL_DIR="/home/sabawi/.lmstudio/models/lmstudio-community/SmolLM2-360M-Finetuned-GGUF"
mkdir -p "$MODEL_DIR"
# Copy the GGUF model to the new directory
cp "training_output/interim_gguf/smollm_finetuned_interim.gguf" "$MODEL_DIR/smollm_finetuned_interim.gguf"
# Create a model card
cat > "$MODEL_DIR/README.md" << 'EOF'
SmolLM2-360M-Finetuned
This is a fine-tuned version of HuggingFaceTB/SmolLM2-360M-Instruct, trained on custom dataset.
Training details

Base model: SmolLM2-360M-Instruct
Training method: QLoRA
Training progress: ~0.5 epochs (interim checkpoint)
Date: May 12, 2025

Usage
This model responds best to the prompt format:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
[Your instruction here]

### Response:
For instructions with additional context:
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
[Your instruction here]

### Input:
[Your input text here]

### Response:
EOF
echo "Model copied to LM Studio community models directory at: $MODEL_DIR"
# Optionally, create a JSON config file for the model
cat > "$MODEL_DIR/config.json" << 'EOF'
{
"name": "SmolLM2-360M-Finetuned",
"modelType": "llama",
"contextSize": 2048,
"promptTemplate": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
}
EOF
echo "Setup complete. Restart LM Studio to see the new model."
