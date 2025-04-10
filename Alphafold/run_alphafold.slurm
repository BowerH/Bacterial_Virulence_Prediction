#!/bin/bash
#SBATCH --job-name=virulence_af
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:2
#SBATCH --time=48:00:00
#SBATCH --mem=120G

module load alphafold/3.0-singularity

run_alphafold.sh \
  --fasta_paths=data/raw/virulence_targets.fasta \
  --max_template_date=2025-04-01 \
  --db_preset=full_dbs \
  --model_preset=multimer_v3 \
  --output_dir=data/processed/structures/