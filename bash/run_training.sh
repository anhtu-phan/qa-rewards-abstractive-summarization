#!/bin/bash

if [ -d "/home/students/anhtu/qa-rewards-abstractive-summarization/venv/" ]; then
  source /home/students/anhtu/qa-rewards-abstractive-summarization/venv/bin/activate
else
  python3 -m venv /home/students/anhtu/qa-rewards-abstractive-summarization/venv
  source /home/students/anhtu/qa-rewards-abstractive-summarization/venv/bin/activate
  pip install -r /home/students/anhtu/qa-rewards-abstractive-summarization/requirements.txt
fi
python /home/students/anhtu/qa-rewards-abstractive-summarization/training.py --pretrained_model_path "./finetuning/output/checkpoint-last" --summary_model_name "gpt2"
deactivate