git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3
git submodule update --init --recursive

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# This depends on your GPU architecture ? you can ask GPT what to set here
export TORCH_CUDA_ARCH_LIST="8.6"

export CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17"
export MAX_JOBS=16

pip uninstall -y flash-attn flash_attn
pip install --no-build-isolation .
