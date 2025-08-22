#!/bin/bash
#SBATCH --job-name=swin_ae             # Job name
#SBATCH --output=swin_ae_%j.log      # Standard output and error log
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
python pipeline/train_swin_ae.py --epochs 500 --batch-size 256 --load-ckpt outputs/swin_ae/ckpt_00_170.pt --num-workers 8
