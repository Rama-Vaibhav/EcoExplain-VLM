#!/bin/bash
# Run ONCE on Ada (login node or short interactive GPU session) after cloning repo.
#   bash scripts/ada_env_bootstrap.sh
#
# Then install PyTorch with CUDA from https://pytorch.org for your module CUDA version.

set -euo pipefail

VENV="${VENV:-$HOME/venvs/ecoexplain}"
REPO="${REPO:-$HOME/EcoExplain-VLM}"

echo "Creating venv at $VENV"
python3 -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -U pip wheel

echo ""
echo "Install PyTorch + CUDA next (example — pick wheels matching Ada modules):"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
echo ""
read -r -p "Install default cu121 torch+vision now? [y/N] " ans
if [[ "${ans,,}" == "y" ]]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

pip install transformers accelerate peft bitsandbytes tqdm Pillow pandas

cd "$REPO"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "OK. Submit jobs from repo root:"
echo "  cd $REPO && mkdir -p results && sbatch scripts/ada_slurm_qwen09_smoke.sh"
echo "  cd $REPO && sbatch scripts/ada_slurm_qwen09.sh"
