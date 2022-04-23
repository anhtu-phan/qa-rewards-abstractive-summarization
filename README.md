# Improve Abstractive Summarization by using Question Answering Rewards 
<p align = "center">
<img src = "https://i.imgur.com/jMh44dK.png">
</p>
<p align = "center">
The training process for the summarization framework with QA rewards <a href="https://aclanthology.org/2021.findings-emnlp.47.pdf">[paper]</a>
</p>
This project will implement the framework proposed in <a href="https://aclanthology.org/2021.findings-emnlp.47.pdf">this paper</a> that provides a general methodology for training abstractive summarization models to address the hallucination problems
such as missing key information in source documents (low recall) or generated summaries be
containing facts that are inconsistent with the source documents (low precision). The framework used question-answering based rewards to further train the pre-trained summarization
models in a Reinforcement Learning (RL) context.

## Features

- [DONE] Experience with GPT-2 and PEGASUS on XSUM dataset
- [TODO] Training with BART 
- [TODO] Experience with SAMSUM dataset

## How to run

- The pre-trained model trains with question-answering based reward can be downloaded from <a href='https://drive.google.com/drive/folders/1-zYEyohanDyyMFZmgnfozLxqwxmN2ydc?usp=sharing'>here</a>
- The pre-trained summarization GPT-2 model can be downloaded from <a href='https://drive.google.com/drive/folders/1i7ZoxNiTyFm6_i7AMLdLK17b-U9muKqd?usp=sharing'>here</a>

### Install 

    #python3.7
    pip install --upgrade pip
    pip install -r requirements.txt

### Demo
    
    python run_demo_server.py --port PORT --model_type TYPE --model_path PATH --model_ref_path PATH

- `PORT`: port to run server (default server will run on http://localhost:8769)
- `model_type`: type of summarization pre-trained model. Chose follow options
  + `gpt2`
  + `google_pegasus_xsum`
- `model_path`: pre-trained model trains with question-answering based reward 
- `model_ref_path`: pre-trained summarization model 

### Training
    
    python training.py --pretrained_model_path PRETRAINED_PATH --summary_model_name MODEL_NAME

- `pretrained_model_path`: pre-trained summarization model
- `summary_model_name`: type of summarization pre-trained model. Chose options
  + `gpt2`
  + `google_pegasus_xsum`
- The training model will be saved to `./checkpoint/{summary_model_name}`

### Eval
    
    python eval.py --model_path MODEL_PATH --model_ref_path MODEL_REF_PATH --model_type MODEL_TYPE

- `model_path`: pre-trained model trains with question-answering based reward
- `model_ref_path`: pre-trained summarization model
- `model_type`: type of summarization pre-trained model. Chose follow options
  + `gpt2`
  + `google_pegasus_xsum`