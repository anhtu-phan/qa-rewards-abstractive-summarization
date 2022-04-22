# qa-rewards-abstractive-summarization
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

## How to run 

### Install 

    #python3.7
    pip install --upgrade pip
    pip install -r requirements.txt

### Demo
    
    python run_demo_server.py --port PORT

### Training
    
    python training.py --pretrained_model_path PRETRAINED_PATH --summary_model_name MODEL_NAME

- `pretrained_model_path`
- `summary_model_name`:
  + `gpt2`
  + `google_pegasus_xsum`

### Eval
    
    python eval.py --model_path MODEL_PATH --model_ref_path MODEL_REF_PATH --model_type MODEL_TYPE

- `model_path`:
- `model_ref_path`:
- `model_type`: