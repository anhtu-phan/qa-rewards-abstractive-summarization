#!/bin/bash

pip install virtualenv
virtualenv venv
source venv/bin/activate
wget https://github.com/deepmind/rc-data/raw/master/requirements.txt
pip install -r requirements.txt
python download-bbc-articles.py [--timestamp_exactness 14]