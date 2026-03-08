#!/bin/bash
echo "Creating conda environment 'georesist_env' with pygimli..."
conda create -y -n georesist_env -c conda-forge python=3.10 pygimli
echo "Activating environment and installing georesistpy..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate georesist_env
pip install -e ".[all]"
echo "Environment created successfully."
echo "To use it, run: conda activate georesist_env"
