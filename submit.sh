#!/bin/bash
#
#SBATCH --job-name=qa_training
#SBATCH --output=qa_training_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun /home/students/anhtu/qa-rewards-abstractive-summarization/run_finetune_gpt2.sh
