#!/bin/bash
#
#SBATCH --job-name=qa_eval
#SBATCH --output=qa_eval_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

source /home/students/anhtu/qa-rewards-abstractive-summarization/venv/bin/activate
python /home/students/anhtu/qa-rewards-abstractive-summarization/eval.py
deactivate