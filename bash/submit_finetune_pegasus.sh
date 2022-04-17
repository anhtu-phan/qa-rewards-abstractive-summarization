#!/bin/bash
#
#SBATCH --job-name=qa_pegasus
#SBATCH --output=qa_finetune_pegasus_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:mem11g:1
#SBATCH --mem=32000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

source /home/students/anhtu/qa-rewards-abstractive-summarization/venv/bin/activate
python /home/students/anhtu/qa-rewards-abstractive-summarization/finetuning/run_summarization.py --model_name_or_path google/pegasus-xsum --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 3e-5 --num_train_epochs 10 --max_source_length 512 --max_target_length 80 --do_train --do_eval --dataset_name xsum --output_dir "/home/students/anhtu/qa-rewards-abstractive-summarization/finetuning/output_pegasus/"
deactivate
