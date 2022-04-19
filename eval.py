import torch
from transformers import (
    GPT2Tokenizer,
)
from ppo.gpt2 import GPT2HeadWithValueModel
from ppo.utils import respond_to_batch
from datasets import load_dataset, load_metric
from rouge_score import rouge_scorer
from tqdm import tqdm
import nltk
import numpy as np
import time

batch_size = 2
model_type = 'gpt2'
max_token_len = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

summary_model = GPT2HeadWithValueModel.from_pretrained(
    pretrained_model_name_or_path="./checkpoint/checkpoint-80").to(device)
summary_model_ref = GPT2HeadWithValueModel.from_pretrained(
    pretrained_model_name_or_path="./finetuning/output/checkpoint-last").to(device)

metric = load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(preds, labels):

    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


measure = ['rouge1', 'rouge2', 'rougeL']
scorer = rouge_scorer.RougeScorer(measure)


def compute_metric_rouge(response_ids, labels):
    result = {}
    for me in measure:
        result[me] = {'precision': 0, 'recall': 0, 'f_measure': 0}

    response = tokenizer.batch_decode(response_ids)
    for i_r, res in enumerate(response):
        res_score = scorer.score(res, labels[i_r])
        for me in measure:
            result[me]['precision'] += res_score[me].precision
            result[me]['recall'] += res_score[me].recall
            result[me]['f_measure'] += res_score[me].fmeasure

    return result


# dataset_builder = load_dataset_builder("xsum")
# print(dataset_builder.info.splits)
ds = load_dataset("xsum", split="test")
ds.set_format('pandas')
df = ds[:]


for i in tqdm(range(0, df.shape[0], batch_size)):
    print(f"--------------- batch {i} ---------------\n")
    df_batch = df.iloc[i:min(i + batch_size, df.shape[0])]
    df_batch.reset_index(inplace=True)
    print(f"df_batch: {df_batch}\n")
    encoding = tokenizer.batch_encode_plus(df_batch['document'], return_tensors="pt", truncation=True,
                                                   padding="max_length", max_length=max_token_len).to(device)
    response_ids = respond_to_batch(summary_model, encoding['input_ids'].to(device), txt_len=80)
    response_ids_ref = respond_to_batch(summary_model_ref, encoding['input_ids'].to(device), txt_len=80)

    t = time.time()
    result_pytorch = compute_metrics(response_ids, df_batch['summary'].tolist())
    print(f"compute_pytorch_time: {time.time()-t}")
    t = time.time()
    result_rouge = compute_metric_rouge(response_ids, df_batch['summary'].tolist())
    print(f"compute_pytorch_time: {time.time() - t}")
    print(result_pytorch)
    print(result_rouge)

