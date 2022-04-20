#!/usr/bin/env python3

import time
from flask import Flask, request, render_template
import argparse

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    time.sleep(5)
    result = request.form['input_doc']
    print(f"result ===>>>>{result}")
    return render_template('index.html', model_result=result, model_ref_result=result)


def run(port):
    app.debug = False  # change this to True if you want to debug
    app.run('0.0.0.0', port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    args = parser.parse_args()
    # model_type = args.model_type

    # if model_type == "google_pegasus_xsum":
    #     summary_model = PegasusForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path).to(
    #         device)
    #     summary_model_ref = PegasusForConditionalGeneration.from_pretrained(
    #         pretrained_model_name_or_path=model_ref_path).to(device)
    #     summary_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    # elif model_type == "gpt2":
    #     summary_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #     summary_model = GPT2Model.from_pretrained(pretrained_model_name_or_path=model_path,
    #                                               max_length=max_token_len).to(device)
    #     summary_model_ref = GPT2Model.from_pretrained(pretrained_model_name_or_path=model_ref_path,
    #                                                   max_length=max_token_len).to(device)
    # else:
    #     raise NotImplementedError

    run(args.port)

