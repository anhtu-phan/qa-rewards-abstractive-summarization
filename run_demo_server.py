#!/usr/bin/env python3

import os

import time
import datetime
import numpy as np
import uuid
import json

import functools
import logging
import collections

### the webserver
from flask import Flask, request, render_template
import argparse

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    result = request.form['input_doc']
    print(f"result ===>>>>{result}")
    return render_template('index.html', model_result=result, model_ref_result=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    args = parser.parse_args()

    app.debug = False  # change this to True if you want to debug
    app.run('0.0.0.0', args.port)


if __name__ == '__main__':
    main()

