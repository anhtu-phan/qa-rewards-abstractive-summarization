#!/bin/bash
#
#SBATCH --job-name=qa_training
#SBATCH --output=qa_training_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun /home/students/anhtu/qa-rewards-abstractive-summarization/bash/run_training.sh