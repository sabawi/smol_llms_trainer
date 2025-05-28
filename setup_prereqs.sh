pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX # Replace cuXXX with your CUDA version (e.g., cu118, cu121)
pip install transformers datasets accelerate bitsandbytes peft # For training
pip install sentencepiece # Often needed by tokenizers
pip install huggingface_hub # For model download/upload
pip install tensorboard # For logging
# For GGUF conversion later (llama.cpp is the common tool)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
/usr/bin/cmake # This might require g++, make, etc. sudo apt install build-essential
cd ..
