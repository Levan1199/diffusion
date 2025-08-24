#!/bin/bash
#SBATCH --job-name=diffusion_lt             # Job name
#SBATCH --output=diffusion_lt%j.log      # Standard output and error log
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --mem=24G                          # Total memory required (adjust as necessary)
#SBATCH --gpus=1                            # Number of GPUs
#SBATCH --time=4:30:00                     # Time limit hrs:min:sec
#SBATCH --mail-user=hungle99@yorku.ca
#SBATCH --mail-type=ALL


# Activate your environment
module load python/3.13.2
source /home/$USER/scratch/hungProject/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision

pip install einops tqdm numpy argparse matplotlib # insntal packages


# run LLM
cd /home/$USER/scratch/diffusion
export PYTHONPATH=$(pwd):$PYTHONPATH
python pipeline/train_dif2.py --epochs 500 --batch-size 512 --num-workers 8 --cond-class 10 --swin-path "outputs\swin_ae\ckpt_03_130.pt"
