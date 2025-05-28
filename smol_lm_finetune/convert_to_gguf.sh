#!/bin/bash

# Set paths
LLAMA_CPP_PATH="../../llama.cpp"
CONVERT_SCRIPT="${LLAMA_CPP_PATH}/convert_hf_to_gguf.py"
MERGED_MODEL_DIR="training_output/interim_merged_model"
GGUF_DIR="training_output/interim_gguf"
OUTPUT_GGUF_FILE="${GGUF_DIR}/smollm_finetuned_interim.gguf"

# Check if convert script exists
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "Error: $CONVERT_SCRIPT not found. Please check the path."
    exit 1
fi

# Create output directory
mkdir -p "$GGUF_DIR"

# Run conversion
echo "Running conversion..."
python3 "$CONVERT_SCRIPT" \
    "$MERGED_MODEL_DIR" \
    --outfile "$OUTPUT_GGUF_FILE" \
    --outtype f16

# Check if conversion was successful
if [ ! -f "$OUTPUT_GGUF_FILE" ]; then
    echo "Error: Conversion failed. Output file not found."
    exit 1
fi

echo "GGUF model saved to: $OUTPUT_GGUF_FILE"

# Optionally quantize the model
QUANTIZE_EXECUTABLE="${LLAMA_CPP_PATH}/quantize"
if [ -f "$QUANTIZE_EXECUTABLE" ]; then
    echo "Quantizing model..."
    QUANTIZED_GGUF_FILE="${GGUF_DIR}/smollm_finetuned_interim_q4_k_m.gguf"
    
    "$QUANTIZE_EXECUTABLE" \
        "$OUTPUT_GGUF_FILE" \
        "$QUANTIZED_GGUF_FILE" \
        "q4_k_m"
    
    echo "Quantized GGUF model saved to: $QUANTIZED_GGUF_FILE"
else
    echo "Warning: Quantize executable not found at $QUANTIZE_EXECUTABLE"
    echo "You may need to compile llama.cpp first with 'make' command"
fi

echo "Conversion complete!"
