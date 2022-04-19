# qa-rewards-abstractive-summarization
<p align = "center">
<img src = "https://i.imgur.com/jMh44dK.png">
</p>
<p align = "center">
The training process for the summarization framework with QA rewards <a href="https://aclanthology.org/2021.findings-emnlp.47.pdf">[paper]</a>
</p>

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