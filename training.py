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
import numpy as np
from tqdm import tqdm
import time
import os

from qa_generation import QAGeneration

tqdm.pandas()

config = {
    "batch_size": 8,
    "steps": 80,
    "forward_batch_size": 4,
    "max_token_len": 512,
    "max_sum_token_len": 128,
    "summary_model_name": "google/pegasus-xsum",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

summary_model = PegasusForConditionalGeneration.from_pretrained(config['summary_model_name']).to(device)
summary_model_ref = PegasusForConditionalGeneration.from_pretrained(config['summary_model_name']).to(device)
summary_tokenizer = PegasusTokenizer.from_pretrained(config['summary_model_name'])

qa_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl").to(device)
ans_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
ans_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl").to(device)
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


def gen_answer(questions, context):
    batch_question_context = []

    for question in questions:
        batch_question_context.append((question, context))

    encoding = gen_answer_tokenizer.batch_encode_plus(batch_question_context, padding=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outputs = gen_answer_model(input_ids, attention_mask=attention_mask)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits

    answers = []
    for (start_score, end_score, input_id) in zip(start_scores, end_scores, input_ids):
        max_startscore = torch.argmax(start_score)
        max_endscore = torch.argmax(end_score)
        ans_tokens = input_id[max_startscore: max_endscore + 1]
        answer_tokens = gen_answer_tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        answer_tokens_string = gen_answer_tokenizer.convert_tokens_to_string(answer_tokens)
        answers.append(answer_tokens_string)

    # print(answers)
    return answers


def get_question_answer_pair(qa_g_a, qa_g_t):
    l_q_g_a = []
    l_a_g_a = []
    l_q_g_t = []
    l_a_g_t = []
    for g_qa, t_qa in zip(qa_g_a, qa_g_t):
        q_g_a = []
        a_g_a = []
        for qa in g_qa:
            q_g_a.append(qa['question'])
            a_g_a.append(qa['answer'])
        l_q_g_a.append(q_g_a)
        l_a_g_a.append(a_g_a)

        q_g_t = []
        a_g_t = []
        for qa in t_qa:
            q_g_t.append(qa['question'])
            a_g_t.append(qa['answer'])
        l_q_g_t.append(q_g_t)
        l_a_g_t.append(a_g_t)

    return l_q_g_a, l_a_g_a, l_q_g_t, l_a_g_t


def norm_levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1]) / max(len(seq1), len(seq2))


def reward_calculation(generated_summaries, ground_truth_summaries):
    qa_gen = QAGeneration(model=qa_model, tokenizer=qa_tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer)

    qa_g_a = [qa_gen(gen_sum) for gen_sum in generated_summaries]
    qa_g_t = [qa_gen(truth_sum) for truth_sum in ground_truth_summaries]

    l_q_g_a, l_a_g_a, l_q_g_t, l_a_g_t = get_question_answer_pair(qa_g_a, qa_g_t)

    reward = []
    for idx, truth_sum in enumerate(ground_truth_summaries):
        g_questions = l_q_g_a[idx]
        t_questions = l_q_g_t[idx]

        a_g_a_ = gen_answer(g_questions, truth_sum)
        r_a = 0
        for a_idx, a_g in enumerate(a_g_a_):
            r_a += norm_levenshtein(a_g, l_a_g_a[idx][a_idx])
        r_a = r_a / len(a_g_a_)

        a_g_t_ = gen_answer(t_questions, generated_summaries[idx])
        r_t = 0
        for t_idx, a_t in enumerate(a_g_t_):
            r_t += norm_levenshtein(a_t, l_a_g_t[idx][t_idx])
        r_t = r_t / len(a_g_t_)
        reward.append((r_a + r_t) / 2)
    return torch.tensor(reward)


def tokenize_document(row):
    encoding = summary_tokenizer.encode_plus(row['document'], return_tensors="pt", truncation=True, padding="max_length").to(device)
    row['tokens'] = encoding["input_ids"][0, :]
    row['attention_mask'] = encoding["attention_mask"][0, :]
    return row


def main():
    x_sum_path = "./datasets/x_sum.pkl"
    if os.path.exists(x_sum_path):
        df = pd.read_pickle(x_sum_path)
    else:
        df = prepare_data()
        # TO-DO increase length of tokens
        df = df.progress_apply(tokenize_document, axis=1)
        df['query'] = df['tokens'].progress_apply(lambda x: summary_tokenizer.decode(x))
        df.to_pickle(x_sum_path)

    ppo_trainer = PPOTrainer(summary_model, summary_model_ref, **config)
    for _ in tqdm(range(int(config['steps'] / config['batch_size']))):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()

        df_batch = df.sample(config['batch_size'])
        game_data['query'] = df_batch['query'].tolist()
        query_tensors = torch.stack(df_batch['tokens'].tolist())
        attention_mask_tensors = torch.stack(df_batch['attention_mask'].tolist())
        ground_truth_sum = df_batch['summary'].tolist()

        t = time.time()

        response_tensors = summary_model.generate(query_tensors)
        game_data['response'] = [summary_tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
        timing['time/get_response'] = time.time() - t

        t = time.time()
        rewards = reward_calculation(game_data['response'], ground_truth_sum)
        timing['time/get_rewards'] = time.time() - t

        t = time.time()
        _ = ppo_trainer.step(query_tensors, response_tensors, rewards, attention_mask_tensors)
        timing['time/optimization'] = time.time() - t

        timing['time/epoch'] = time.time() - t0
        print(str(timing))

        logs['env/reward_mean'] = torch.mean(rewards)
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
        print(str(logs))


if __name__ == "__main__":
    main()