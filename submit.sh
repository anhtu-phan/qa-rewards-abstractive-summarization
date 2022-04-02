#!/bin/bash
#
#SBATCH --job-name=qa_training
#SBATCH --output=qa_training_output.txt
#SBATCH --ntasks=1
#SBATCH --time=1500:00
#SBATCH --mem=8192
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun /home/students/anhtu/qa-rewards-abstractive-summarization/run_finetune_gpt2.sh
