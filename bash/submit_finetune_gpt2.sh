#!/bin/bash
#
#SBATCH --job-name=qa_fine_tuning
#SBATCH --output=qa_finetune_gpt2_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

source /home/students/anhtu/qa-rewards-abstractive-summarization/venv/bin/activate
python /home/students/anhtu/qa-rewards-abstractive-summarization/finetuning/run_clm.py --model_name_or_path gpt2 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 --learning_rate 6.25e-5 --adam_epsilon 1e-8 --max_grad_norm 1.0 --num_train_epochs 10 --warmup_steps 500 --block_size 512 --do_train --do_eval --dataset_name xsum --output_dir "/home/students/anhtu/qa-rewards-abstractive-summarization/finetuning/output/"
deactivate
