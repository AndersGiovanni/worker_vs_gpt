#!/bin/bash

#SBATCH --job-name=experiments    # Job name
#SBATCH --output=run_outputs/convert_vicuna.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=16       # Schedule one core
#SBATCH --mem-per-cpu=7G 
#SBATCH --time=24:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red,brown    # Run on either the Red or Brown queue
#SBATCH --mail-type=FAIL,END    # Send an email when the job finishes or fails

hostname

# nvidia-smi

module load poetry


poetry shell


# poetry update

# poetry install

python3 -W ignore -m src.worker_vs_gpt.data_processing.apply_delta --base decapoda-research/llama-13b-hf --target ./models/vicuna/vicuna-13b --delta lmsys/vicuna-13b-delta
