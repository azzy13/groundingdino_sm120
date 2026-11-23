#!/usr/bin/env bash
# Create a conda env and build GroundingDINO's CUDA extension for Ampere/Ada/Blackwell (sm_80/86/89/120 + PTX).
# Usage: bash setup_env.sh [env_name]      (default: dino_sm120)
set -euo pipefail
ENV=${1:-dino_sm120}
HERE=$(cd "$(dirname "$0")" && pwd)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV" python=3.10
conda env config vars set -n "$ENV" PYTHONNOUSERSITE=1   # keep ~/.local packages out of this env
conda activate "$ENV"

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c nvidia cuda-version=12.8 cuda-nvcc cuda-cudart-dev cuda-cccl \
    libcusparse-dev libcublas-dev libcusolver-dev libcurand-dev cuda-profiler-api
pip install numpy==1.26.4 transformers==4.49.0 opencv-python "supervision>=0.22.0" \
    pycocotools addict yapf timm "setuptools>=82,<83" wheel setuptools_scm ninja

# conda's CUDA headers live under targets/, which g++ does not search by default
T=$CONDA_PREFIX/targets/x86_64-linux
cd "$HERE/src"
rm -rf build ./*.egg-info
CUDA_HOME=$CONDA_PREFIX CPATH=$T/include LIBRARY_PATH=$T/lib \
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;12.0+PTX" MAX_JOBS=8 \
    pip install . --no-deps --no-build-isolation

cd /tmp
SO=$(python -c "import torch, groundingdino._C as c; print(c.__file__)")
echo "Built: $SO"
cuobjdump --list-elf "$SO" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' '; echo "(SASS)"
cuobjdump --list-ptx "$SO" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' '; echo "(PTX)"
python "$HERE/tests/test_msda.py"
