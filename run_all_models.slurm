#!/bin/bash

#SBATCH --job-name=generation_models
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition=gpu
#SBATCH --qos=gpu
##SBATCH --gres=gpu:p100:1
##SBATCH --gres=gpu:a100_80g:1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=24
#SBATCH --time=0-04:00:00

# Carica il modulo Miniconda3
module load miniconda3

# Nome dell'ambiente Conda
ENV_NAME=diffusion-env

# Se l'ambiente non esiste, crealo e installa i pacchetti
if ! conda env list | grep -q "$ENV_NAME"; then
  echo "L'ambiente Conda non esiste. Lo creo..."
  conda create -y -n $ENV_NAME python=3.9
  source activate $ENV_NAME
  echo "Installo i pacchetti necessari..."
  conda install -y pytorch torchvision matplotlib imageio -c pytorch
else
  echo "L'ambiente Conda $ENV_NAME esiste già."
fi

# Attiva l'ambiente Conda
source activate $ENV_NAME


# Verifica e installa 'denoising-diffusion-pytorch' se necessario
if ! python -c "import denoising_diffusion_pytorch" &> /dev/null; then
  echo "denoising-diffusion-pytorch non trovato. Procedo con l'installazione..."
  pip install denoising-diffusion-pytorch
else
  echo "denoising-diffusion-pytorch già installato."
fi


# Esegui gli script Python
# python run_vae.py -cats
# python run_gan.py -cats
python run_diffusion.py -cats
