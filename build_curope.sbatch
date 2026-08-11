#!/bin/bash
#SBATCH --job-name=build_curope
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_curope_%A.out
#SBATCH --error=logs/build_curope_%A.err
#
# Compile the CroCo CUDA RoPE2D kernel (src/croco/models/curope).
#
# Without it, croco/models/pos_embed.py falls back to the pure-PyTorch RoPE2D
# ("cannot find cuda-compiled version of RoPE2D, using a slow pytorch version
# instead") on EVERY forward pass of every training/eval run.
#
# Must run on a GPU node: curope/setup.py calls torch.cuda.get_gencode_flags(),
# which needs a visible device to pick the arch (sm_90 on MN5's H100s).
#
#   sbatch build_curope.sbatch
#
set -eo pipefail

# --- self-locating worktree -------------------------------------------------
# Use the directory this script was submitted from, NOT a hardcoded path, so the
# job always builds the checkout you actually submitted. Fail loudly otherwise.
WT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -d "$WT/src/CUT3R/src/croco/models/curope" ] || {
  echo "ERROR: \$WT does not look like a my-da3 checkout: $WT" >&2; exit 1; }
cd "$WT"
mkdir -p logs

echo "worktree : $WT"
echo "node     : $(hostname)"

# --- toolchain --------------------------------------------------------------
# nvidia-hpc-sdk/25.3 ships nvcc 12.8.93, an exact match for torch 2.8.0+cu128
# (a CUDA MAJOR-version mismatch makes torch.utils.cpp_extension hard-fail).
#
# Do NOT `module load nvidia-hpc-sdk/25.3`: it setenv's CC/CXX to nvc/nvc++ and
# prepends compilers/bin ahead of cuda/bin. That breaks the build two ways --
#   nvc++-Error-Unknown switch: -fwrapv        (Python's CFLAGS assume GCC)
#   nvcc fatal: Unsupported NVHPC compiler found. nvc++ is the only NVHPC
#               compiler that is supported.    (-ccbin resolved to nvc)
# and it shadows nvcc 12.8 with an 11.8 one. Reference the CUDA tree directly
# and keep the GNU toolchain instead.
export CUDA_HOME=/apps/ACC/NVIDIA-HPC-SDK/25.3/Linux_x86_64/25.3/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=gcc
export CXX=g++

source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

echo "nvcc     : $(nvcc --version | grep release)"
echo "g++      : $(g++ --version | head -1)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# --- build ------------------------------------------------------------------
cd "$WT/src/CUT3R/src/croco/models/curope"
python setup.py build_ext --inplace

# --- verify -----------------------------------------------------------------
# Not just "does it import": the kernel rejects plain-contiguous tensors, so the
# only thing that matters is whether the layouts blocks.py actually produces are
# accepted, and whether the result matches the pure-PyTorch RoPE it replaces.
cd "$WT"
python verify_curope.py

echo "BUILD OK"
