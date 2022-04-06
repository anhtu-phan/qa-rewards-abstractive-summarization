import torch
from transformers import (
    GPT2Tokenizer,
)
from ppo.gpt2 import GPT2HeadWithValueModel
from ppo.utils import respond_to_batch
from datasets import load_dataset, load_dataset_builder
from rouge_score import rouge_scorer
from tqdm import tqdm

batch_size = 2
model_type = 'gpt2'
max_token_len = 512

# dataset_builder = load_dataset_builder("xsum")
# print(dataset_builder.info.splits)
ds = load_dataset("xsum", split="test")
ds.set_format('pandas')
df = ds[:]

device = "cuda" if torch.cuda.is_available() else "cpu"

summary_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
summary_tokenizer.pad_token = summary_tokenizer.eos_token
summary_model = GPT2HeadWithValueModel.from_pretrained(pretrained_model_name_or_path="./finetuning/output/checkpoint-10000").to(device)
summary_model_ref = GPT2HeadWithValueModel.from_pretrained(pretrained_model_name_or_path="./finetuning/output/checkpoint-10000").to(device)

measure = ['rouge1', 'rouge2', 'rougeL']
scorer = rouge_scorer.RougeScorer(measure)

result = {}
result_ref = {}
for me in measure:
    result[me] = {'precision': 0, 'recall': 0, 'f_measure': 0}
    result_ref[me] = {'precision': 0, 'recall': 0, 'f_measure': 0}

for i in tqdm(range(0, df.shape[0], batch_size)):
    df_batch = df.iloc[i:min(i+batch_size, df.shape[0])]
    encoding = summary_tokenizer.batch_encode_plus(df_batch['document'], return_tensors="pt", truncation=True, padding="max_length", max_length=max_token_len).to(device)
    response_ids = respond_to_batch(summary_model, encoding['input_ids'].to(device), txt_len=80, seq2seq=False)
    response_ids_ref = respond_to_batch(summary_model_ref, encoding['input_ids'].to(device), txt_len=80, seq2seq=False)

    response = summary_tokenizer.batch_decode(response_ids)
    response_ref = summary_tokenizer.batch_decode(response_ids_ref)

    for i_r, res in enumerate(response):
        res_score = scorer.score(res, df_batch['summary'][i_r])
        ref_score = scorer.score(response_ref[i_r], df_batch['summary'][i_r])
        for me in measure:
            result[me]['precision'] += res_score[me].precision
            result[me]['recall'] += res_score[me].recall
            result[me]['f_measure'] += res_score[me].fmeasure

            result_ref[me]['precision'] += ref_score[me].precision
            result_ref[me]['recall'] += ref_score[me].recall
            result_ref[me]['f_measure'] += ref_score[me].fmeasure

for me in measure:
    result[me]['precision'] /= df.shape[0]
    result[me]['recall'] /= df.shape[0]
    result[me]['f_measure'] /= df.shape[0]

    result_ref[me]['precision'] /= df.shape[0]
    result_ref[me]['recall'] /= df.shape[0]
    result_ref[me]['f_measure'] /= df.shape[0]

print(result)
print(result_ref)
