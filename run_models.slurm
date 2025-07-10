#!/bin/bash

#SBATCH --job-name=vae_vs_gan_vs_diffusion
#SBATCH --output="outputs/%j/%x.o%j"
#SBATCH --error="outputs/%j/%x.e%j"
#SBATCH --partition=gpu
#SBATCH --qos=gpu
##SBATCH --gres=gpu:p100:1
#SBATCH --gres=gpu:a100_40g:1
##SBATCH --gres=gpu:a100_80g:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=24
#SBATCH --time=0-02:00:00

# --- Preparazione della directory di output ---

# Directory principale per gli output
OUTPUTS_DIR="outputs"
# Crea la directory principale se non esiste
mkdir -p "$OUTPUTS_DIR"

# Nome della sottocartella basata sull'ID del job
# Slurm fornisce l'ID del job nella variabile d'ambiente $SLURM_JOB_ID
JOB_OUTPUT_DIR="$OUTPUTS_DIR/$SLURM_JOB_ID"
# Crea la sottocartella specifica per questo job
mkdir -p "$JOB_OUTPUT_DIR"

# Spostati nella cartella di output specifica del job
# Tutti gli output successivi (es. file .gif) verranno salvati qui
cd "$JOB_OUTPUT_DIR"

# --- Stampa delle risorse richieste ---
echo "--- Informazioni sul Job Slurm ---"
echo "Nome Job: $SLURM_JOB_NAME"
echo "ID Job: $SLURM_JOB_ID"
echo "Partizione: $SLURM_JOB_PARTITION"
echo "QoS: $SLURM_JOB_QOS"
echo "GPU Richieste: $SLURM_JOB_GPUS"
echo "Memoria Richiesta: $SLURM_MEM_PER_NODE"
echo "CPU per Task: $SLURM_CPUS_PER_TASK"
echo "Data e Ora Inizio Esecuzione: $(date)"
echo "-----------------------------------"

# --- Parsing degli argomenti ---
RUN_VAE=false
RUN_GAN=false
RUN_DIFFUSION=false
ARGS_PROVIDED=false # Flag per controllare se sono stati forniti argomenti

for arg in "$@"; do
    ARGS_PROVIDED=true # Almeno un argomento è stato fornito
    case "$arg" in
        -vae)
            RUN_VAE=true
            ;;
        -gan)
            RUN_GAN=true
            ;;
        -diffusion)
            RUN_DIFFUSION=true
            ;;
        *)
            echo "Avviso: Parametro non riconosciuto: $arg"
            ;;
    esac
done

# Se nessun parametro specifico è stato fornito, imposta tutti i flag su true
if ! $ARGS_PROVIDED; then
    echo "Nessun modello specificato. Eseguo tutti i modelli di default."
    RUN_VAE=true
    RUN_GAN=true
    RUN_DIFFUSION=true
fi

# --- Configurazione dell'ambiente Conda ---
module load miniconda3

# Nome dell'ambiente Conda
ENV_NAME=diffusion-env

# Se l'ambiente Conda non esiste, crealo e installa i pacchetti
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "L'ambiente Conda non esiste. Lo creo..."
    conda create -y -n "$ENV_NAME" python=3.9
    source activate "$ENV_NAME"
    echo "Installo i pacchetti necessari..."
    conda install -y pytorch torchvision matplotlib imageio -c pytorch
    conda config --add channels defaults
else
    echo "L'ambiente Conda $ENV_NAME esiste già."
fi

# Attiva l'ambiente Conda
source activate "$ENV_NAME"

# Verifica e installa 'denoising-diffusion-pytorch' se necessario
if ! python -c "import denoising_diffusion_pytorch" &> /dev/null; then
    echo "denoising-diffusion-pytorch non trovato. Procedo con l'installazione..."
    pip install denoising-diffusion-pytorch
else
    echo "denoising-diffusion-pytorch già installato."
fi

# --- Esecuzione degli script Python ---

# Funzione per calcolare e stampare il tempo di esecuzione di uno script
calculate_and_print_time() {
    local start_time=$1
    local script_name=$2
    local end_time=$(date +%s)
    local total_seconds=$((end_time - start_time))
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "Tempo impiegato per %s: %02d:%02d:%02d\n" "$script_name" "$hours" "$minutes" "$seconds"
}

# Registra il tempo di inizio per calcolare la durata totale
START_TIME_TOTAL=$(date +%s)

# Esegui gli script Python specificati (o tutti di default)
if $RUN_VAE; then
    echo "Esecuzione di run_vae.py..."
    START_TIME_VAE=$(date +%s)
    python ../../run_vae.py -cats
    calculate_and_print_time "$START_TIME_VAE" "run_vae.py"
fi

if $RUN_GAN; then
    echo "Esecuzione di run_gan.py..."
    START_TIME_GAN=$(date +%s)
    python ../../run_gan.py -cats
    calculate_and_print_time "$START_TIME_GAN" "run_gan.py"
fi

if $RUN_DIFFUSION; then
    echo "Esecuzione di run_diffusion.py..."
    START_TIME_DIFFUSION=$(date +%s)
    python ../../run_diffusion.py -cats
    calculate_and_print_time "$START_TIME_DIFFUSION" "run_diffusion.py"
fi

# --- Calcolo e stampa del tempo totale impiegato del Job ---
END_TIME_TOTAL=$(date +%s)
TOTAL_SECONDS_JOB=$((END_TIME_TOTAL - START_TIME_TOTAL))

# Formatta il tempo in HH:MM:SS
HOURS_JOB=$((TOTAL_SECONDS_JOB / 3600))
MINUTES_JOB=$(((TOTAL_SECONDS_JOB % 3600) / 60))
SECONDS_JOB=$((TOTAL_SECONDS_JOB % 60))

printf "\n--- Riepilogo Esecuzione Job ---\n"
printf "Data e Ora Fine Esecuzione: $(date)\n"
printf "Tempo Totale Impiegato per il Job: %02d:%02d:%02d\n" "$HOURS_JOB" "$MINUTES_JOB" "$SECONDS_JOB"
echo "------------------------------"

# Torna alla directory originale
cd ../../
