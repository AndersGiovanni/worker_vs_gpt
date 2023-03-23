#!/bin/bash

#SBATCH --job-name=install_environment    # Job name
#SBATCH --output=run_outputs/poetry_installer.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=16       # Schedule one core
#SBATCH --time=02:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=brown,red    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END    # Send an email when the job finishes or fails

hostname

module load poetry

poetry --version

poetry shell

poetry update

poetry install