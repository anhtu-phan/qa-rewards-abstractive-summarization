#!/usr/bin/env python3

from model import init_summary_model
from ppo.utils import respond_to_batch

import torch
from flask import Flask, request, render_template
import argparse

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', doc_input=None, model_type=model_type)


@app.route('/', methods=['POST'])
def index_post():
    doc_input = request.form['input_doc']
    encoding = summary_tokenizer.encode_plus(doc_input, return_tensors="pt", truncation=True, padding="max_length", max_length=max_token_len).to(device)
    response_ids = respond_to_batch(summary_model, encoding['input_ids'].to(device), txt_len=80)
    response_ids_ref = respond_to_batch(summary_model_ref, encoding['input_ids'].to(device), txt_len=80)

    response = summary_tokenizer.decode(response_ids[0])
    response_ref = summary_tokenizer.decode(response_ids_ref[0])

    return render_template('index.html', doc_input=doc_input.strip(), model_type=model_type, model_result=response, model_ref_result=response_ref)


def run(port):
    app.debug = False  # change this to True if you want to debug
    app.run('0.0.0.0', port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_ref_path', type=str)
    parser.add_argument('--max_token_len', default=80, type=int)
    args = parser.parse_args()
    model_type = args.model_type
    model_path = args.model_path
    model_ref_path = args.model_ref_path
    max_token_len = args.max_token_len

    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary_model, summary_model_ref, summary_tokenizer = init_summary_model(model_type, model_path, model_ref_path, device)

    run(args.port)

