#!/bin/bash
#
#SBATCH --job-name=qa_training
#SBATCH --output=qa_training_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --mem=32000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun /home/students/anhtu/qa-rewards-abstractive-summarization/run_training.sh