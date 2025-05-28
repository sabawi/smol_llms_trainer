# NEW INSTRUCTION - USING CMAKE
# Ensure you have cmake and a C++ compiler (like g++) installed
sudo apt update
sudo apt install cmake build-essential

# If you haven't cloned llama.cpp yet:
# git clone https://github.com/ggerganov/llama.cpp.git

# Navigate to the llama.cpp directory
cd llama.cpp  # Make sure you are in the root of the llama.cpp repository

# Create a build directory and navigate into it
mkdir build
cd build

# Configure the build (this generates the build files, e.g., Makefiles for 'make' or Ninja files)
# You can add options here, e.g., -DLLAMA_CUBLAS=ON for NVIDIA GPU support for inference,
# though for conversion/quantization, CPU is often fine.
# For just the conversion tools, basic cmake should be enough.
cmake ..

# Build the project (this will compile the executables like 'quantize', 'main', etc.)
# The -j flag allows parallel compilation, using number of available cores.
cmake --build . --config Release -j

# The compiled binaries (like 'quantize') will now typically be in this 'build/bin' directory
# or directly in the 'build' directory depending on llama.cpp's CMake setup.
# Common location: ./bin/quantize or ./quantize from within the 'build' directory.

# Go back to your project's root or the parent directory of llama.cpp
cd ../.. # If you were in smol_lm_finetune/llama.cpp/build, this takes you to smol_lm_finetune
