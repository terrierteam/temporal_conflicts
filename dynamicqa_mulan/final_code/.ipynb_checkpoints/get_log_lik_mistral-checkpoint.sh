#!/bin/bash

# ================================
#   Experiment Runner Script
# ================================

# Load conda (adjust this path if needed)
source /opt/miniconda3/etc/profile.d/conda.sh

# Root projects directory
PROJECTS_DIR="$HOME/nfs/projects"

# Define environments + working directories
ENV1="dqa_312_hf_latest"       # for experiments
DIR1="$PROJECTS_DIR/dynamicqa_MULAN_1/final_code/"    # directory where experiment scripts live

ENV2="fact_mutability"         # for long program
DIR2="$PROJECTS_DIR/fact_mutability/"   # directory where long program lives

# ------------------------
# First environment (experiments)
# ------------------------
echo ">>> Activating $ENV1"
conda activate "$ENV1"
echo "Active env: $CONDA_DEFAULT_ENV"

cd "$DIR1" || exit 1
echo "Working dir: $(pwd)"

# === Run your experiment programs here ===
python get_likelihoods_MULAN.py --evaluation_model mistralai/Mistral-7B-Instruct-v0.1 --dataset mutable
python get_likelihoods_MULAN.py --evaluation_model mistralai/Mistral-7B-Instruct-v0.1 --dataset immutable
python get_likelihoods_MULAN.py --evaluation_model mistralai/Mistral-7B-Instruct-v0.1 --dataset immutable_n
python get_likelihoods_MULAN.py --evaluation_model mistralai/Mistral-7B-Instruct-v0.1 --dataset mutable --add_context True
python get_likelihoods_MULAN.py --evaluation_model mistralai/Mistral-7B-Instruct-v0.1 --dataset immutable --add_context True
python get_likelihoods_MULAN.py --evaluation_model mistralai/Mistral-7B-Instruct-v0.1 --dataset immutable_n --add_context True
conda deactivate

# ------------------------
# Second environment (long program)
# ------------------------
echo ">>> Activating $ENV2"
conda activate "$ENV2"
echo "Active env: $CONDA_DEFAULT_ENV"

cd "$DIR2" || exit 1
echo "Working dir: $(pwd)"

# Force wandb into offline mode ONLY for this program
export WANDB_MODE=offline

# === Run your long program here ===
# Example:
python inference.py --queries_path dataset/data/fm_queries_0.txt --model_name_or_path huggyllama/llama-7b --exp_name very_important3

conda deactivate

echo ">>> Done!"