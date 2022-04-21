#!/bin/bash

if [ -d "/home/students/anhtu/qa-rewards-abstractive-summarization/venv/" ]; then
  source /home/students/anhtu/qa-rewards-abstractive-summarization/venv/bin/activate
else
  python3 -m venv /home/students/anhtu/qa-rewards-abstractive-summarization/venv
  source /home/students/anhtu/qa-rewards-abstractive-summarization/venv/bin/activate
  pip install -r /home/students/anhtu/qa-rewards-abstractive-summarization/requirements.txt
fi
python /home/students/anhtu/qa-rewards-abstractive-summarization/training.py --pretrained_model_path ./finetuning/output_pegasus/checkpoint-2500 --summary_model_name "google_pegasus_xsum" --max_token_len 128 --max_sum_token_len 40
deactivate