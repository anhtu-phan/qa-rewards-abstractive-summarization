import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    PegasusTokenizer,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)

from datasets import load_dataset
from ppo.gpt2 import GPT2HeadWithValueModel
from ppo.pegasus import PegasusHeadWithValueModel
from ppo.ppo import PPOTrainer
from ppo.utils import respond_to_batch
from qa_generation import QAGeneration

tqdm.pandas()

config = {
    "batch_size": 4,
    "steps": 100000,
    "forward_batch_size": 2,
    "max_token_len": 128,
    "max_sum_token_len": 60,
    "summary_model_name": "gpt2",
}
checkpoint_idx = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--------------Detected device {device}--------------\n")

if config['summary_model_name'] == "google_pegasus_xsum":
    summary_model = PegasusHeadWithValueModel.from_pretrained("google/pegasus-xsum").to(device)
    summary_model_ref = PegasusHeadWithValueModel.from_pretrained("google/pegasus-xsum").to(device)
    summary_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
elif config['summary_model_name'] == "gpt2":
    summary_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if checkpoint_idx == 0:
        summary_model = GPT2HeadWithValueModel.from_pretrained(
            pretrained_model_name_or_path="./finetuning/output/checkpoint-last").to(device)
    else:
        summary_model = GPT2HeadWithValueModel.from_pretrained(
            pretrained_model_name_or_path=f"./checkpoint/checkpoint-{checkpoint_idx}").to(device)

    summary_model_ref = GPT2HeadWithValueModel.from_pretrained(
        pretrained_model_name_or_path="./finetuning/output/checkpoint-last").to(
        device)
    # summary_model = GPT2HeadWithValueModel.from_pretrained("gpt2")
    # summary_model_ref = GPT2HeadWithValueModel.from_pretrained("gpt2")
else:
    raise NotImplementedError
summary_tokenizer.pad_token = " "

qa_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")
ans_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
ans_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl").to(device)
gen_answer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
gen_answer_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad").to(device)


def prepare_data():
    ds = load_dataset("xsum", split="train")
    ds.set_format('pandas')
    df = ds[:]
    return df


def gen_answer(questions, context):
    batch_question_context = []
    print(f"******************gen-answer******************\nquestion: {questions} \n context: {context}\n")

    for question in questions:
        if question.strip() == '':
            continue
        batch_question_context.append((question, context))

    str_questions = "\n".join(questions)
    if str_questions.strip() == '':
        return []
    encoding = gen_answer_tokenizer.batch_encode_plus(batch_question_context, padding=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
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

    print(f"answer: {answers}\n-------------------------\n")
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
            a_g_a.append(qa['answer'].replace("<pad>", "").strip())
        l_q_g_a.append(q_g_a)
        l_a_g_a.append(a_g_a)

        q_g_t = []
        a_g_t = []
        for qa in t_qa:
            q_g_t.append(qa['question'])
            a_g_t.append(qa['answer'].replace("<pad>", "").strip())
        l_q_g_t.append(q_g_t)
        l_a_g_t.append(a_g_t)

    return l_q_g_a, l_a_g_a, l_q_g_t, l_a_g_t


def norm_levenshtein(seq1, seq2):
    print(f"******************compute reward******************\n{seq1} \nvs\n{seq2}\n")
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
    reward = 1 - (matrix[size_x - 1, size_y - 1]) / max(len(seq1), len(seq2))
    print(f"===>>>> reward = {reward}\n")
    return reward


def reward_calculation(generated_summaries, ground_truth_summaries):
    qa_gen = QAGeneration(model=qa_model, tokenizer=qa_tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer)

    qa_g_a = []
    for gen_sum in generated_summaries:
        if len(gen_sum) >= 5:
            qa_g_a.append(qa_gen(gen_sum))
        else:
            qa_g_a.append([])

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

        a_g_t_ = gen_answer(t_questions, generated_summaries[idx])
        r_t = 0
        for t_idx, a_t in enumerate(a_g_t_):
            r_t += norm_levenshtein(a_t, l_a_g_t[idx][t_idx])
        if (len(a_g_a_) + len(a_g_t_)) == 0:
            reward.append(0)
        else:
            reward.append((r_a + r_t) / (len(a_g_a_) + len(a_g_t_)))

    return torch.tensor(reward).to(device)


def tokenize_document(row):
    # if len(row['document'].split(" ")) < config['max_token_len']:
    #     return None
    encoding = summary_tokenizer.encode_plus(row['document'], return_tensors="pt", truncation=True,
                                             padding="max_length", max_length=config['max_token_len']).to(device)
    row['tokens'] = encoding["input_ids"][0, :]
    row['attention_mask'] = encoding["attention_mask"][0, :]
    return row


def main():
    if not os.path.exists("./datasets"):
        os.makedirs("./datasets")
    # if not os.path.exists("./checkpoint"):
    #     os.makedirs("./checkpoint")

    x_sum_path = f"./datasets/x_sum_{config['summary_model_name']}.pkl"
    if os.path.exists(x_sum_path):
        print("---------->>>>>>>>>>>> load prepared dataset\n")
        df = pd.read_pickle(x_sum_path)
    else:
        print("prepare dataset ---------->>>>>>>>>>>>\n")
        df = prepare_data()
        # TODO increase length of tokens
        df = df.progress_apply(tokenize_document, axis=1)
        # df = df.dropna()
        df['query'] = df['tokens'].progress_apply(lambda x: summary_tokenizer.decode(x))
        df.to_pickle(x_sum_path)

    ppo_trainer = PPOTrainer(summary_model, summary_model_ref, **config)
    for step_idx in range(checkpoint_idx + 1, int(config['steps'] / config['batch_size'])):
        print(f"\n\n---------------training step {step_idx}---------------\n")
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()
        if step_idx*config['batch_size'] > df.shape[0]:
            df_batch = df.sample(config['batch_size'])
        else:
            df_batch = df.iloc[(step_idx-1)*config['batch_size']: min(step_idx*config['batch_size'], df.shape[0])]
        print(df_batch.index)
        game_data['query'] = df_batch['query'].tolist()
        # TODO: padding -> end of text
        query_tensors = torch.stack(df_batch['tokens'].tolist()).to(device)
        ground_truth_sum = df_batch['summary'].tolist()
        t = time.time()

        response_tensors = []
        for i in range(int(config['batch_size'] / config['forward_batch_size'])):
            if config['summary_model_name'] == "gpt2":
                response = respond_to_batch(summary_model, query_tensors[
                                                           i * config['forward_batch_size']:(i + 1) * config[
                                                               'forward_batch_size']],
                                            txt_len=config["max_sum_token_len"])
            elif config['summary_model_name'] == "google_pegasus_xsum":
                response = respond_to_batch(summary_model, query_tensors[
                                                           i * config['forward_batch_size']:(i + 1) * config[
                                                               'forward_batch_size']],
                                            txt_len=config["max_sum_token_len"],
                                            eos_token=summary_tokenizer.eos_token_id, device=device)
            else:
                raise NotImplementedError

            response_tensors.append(response)

        response_tensors = torch.cat(response_tensors).to(device)
        game_data['response'] = [summary_tokenizer.decode(response_tensors[i, :]).strip() for i in
                                 range(config['batch_size'])]
        print_game_data = '\n->'.join(game_data['response'])
        print(f"******************game_data['response']******************\n{print_game_data}\n")
        timing['time/get_response'] = time.time() - t

        t = time.time()
        rewards = reward_calculation(game_data['response'], ground_truth_sum)
        timing['time/get_rewards'] = time.time() - t

        t = time.time()
        if config['summary_model_name'] == "gpt2":
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        elif config['summary_model_name'] == "google_pegasus_xsum":
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards, decoder_input_ids=response_tensors)

        timing['time/optimization'] = time.time() - t

        timing['time/epoch'] = time.time() - t0

        logs['env/reward_mean'] = torch.mean(rewards)
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
        logs.update(timing)
        logs.update(stats)
        print(str(logs))

        if step_idx != 0 and step_idx % 200 == 0:
            summary_model.save_pretrained(
                f"./checkpoint/checkpoint-{step_idx}")


if __name__ == "__main__":
    main()
