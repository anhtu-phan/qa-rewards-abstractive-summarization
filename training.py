from transformers import (
    PegasusForConditionalGeneration, 
    PegasusTokenizer, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
import torch
from trl.ppo import PPOTrainer
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import time
import os

from qa_generation import QAGeneration

batch_size = 256
steps = 25600
forward_batch_size = 16
xsum_path = "./datasets/xsum.csv"
max_token_len = 512

summary_model_name = "google/pegasus-xsum"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
    
summary_model = PegasusForConditionalGeneration.from_pretrained(summary_model_name).to(device)
summary_model_ref = PegasusForConditionalGeneration.from_pretrained(summary_model_name).to(device)
qa_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl").to(device)
gen_answer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
gen_answer_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad").to(device)


def prepare_data():
    ds = load_dataset("xsum", split="train")
    # print(ds[0])
    # print(ds.shape)
    # print(ds.features)
    ds.set_format('pandas')
    df = ds[:]
    return df


def reward_calculation(generated_summaries, ground_truth_summaries):
    # generated_summaries_tokens = [qa_tokenizer.encode(sum, return_tensors='pt').to(device)[0, :] for sum in generated_summaries]
    # generated_summaries_tokens = t5_tokenizer(generated_summaries).input_ids
    qa_gen = QAGeneration(model=qa_model, tokenizer=qa_tokenizer, ans_model=qa_model, ans_tokenizer=qa_tokenizer)
    print(qa_gen(generated_summaries[1]))    


def main():
    if os.path.exists(xsum_path):
        df = pd.read_csv(xsum_path)
    else:
        df = prepare_data()
        summary_tokenizer = PegasusTokenizer.from_pretrained(summary_model_name)
        #TO-DO increase length of tokens
        df['tokens'] = df['document'].progress_apply(lambda x: summary_tokenizer.encode(x, return_tensors="pt", truncation=True, padding=True).to(device)[0, :])
        print(df.head(1))
        print("-------------------")
        # print(df[0, :])
        # print(df['tokens'][0].length)
        df['query'] = df['tokens'].progress_apply(lambda x: summary_tokenizer.decode(x))
        df.to_csv(xsum_path)
        print(df.head(1))

    ppo_trainer = PPOTrainer(summary_model, summary_model_ref)
    for epoch in tqdm(range(int(steps/batch_size))):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()

        df_batch = df.sample(batch_size)
        game_data['query'] = df_batch['query'].tolist()
        query_tensors = torch.stack(df_batch['tokens'].tolist())
        print(f'query_tensors shape ', query_tensors.shape)

        t = time.time()
        response_tensors = []
        for i in range(int(batch_size/forward_batch_size)):
            response = summary_model.generate(query_tensors)
            response_tensors.append(response)
        
        print(f'response_tensors shape ', response_tensors.shape)
        print(response_tensors)
        response_tensors = torch.cat(response_tensors)
        game_data['response'] = [summary_tokenizer.decode(response_tensors[i, :]) for i in range(batch_size)]
        timing['time/get_response'] = time.time() - t
        
        
if __name__ == "__main__":
    reward_calculation(["summarize: studies have shown that owning a dog is good for you", "42 is the answer to life, universe and everything."], [])