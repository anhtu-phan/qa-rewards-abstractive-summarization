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

batch_size = 8
steps = 80
forward_batch_size = 4
x_sum_path = "./datasets/x_sum.pkl"
max_token_len = 512
max_sum_token_len = 128

summary_model_name = "google/pegasus-xsum"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

summary_model = PegasusForConditionalGeneration.from_pretrained(summary_model_name).to(device)
summary_model_ref = PegasusForConditionalGeneration.from_pretrained(summary_model_name).to(device)
summary_tokenizer = PegasusTokenizer.from_pretrained(summary_model_name)

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

    print(answers)
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
            q_g_a.append(qa.question)
            a_g_a.append(qa.answer)
        l_q_g_a.append(q_g_a)
        l_a_g_a.append(a_g_a)

        q_g_t = []
        a_g_t = []
        for qa in t_qa:
            q_g_t.append(qa.question)
            a_g_t.append(qa.answer)
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
    # generated_summaries_tokens = [qa_tokenizer.encode(sum, return_tensors='pt').to(device)[0, :] for sum in generated_summaries]
    # generated_summaries_tokens = t5_tokenizer(generated_summaries).input_ids
    qa_gen = QAGeneration(model=qa_model, tokenizer=qa_tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer)
    qa_g_a = [qa_gen(gen_sum) for gen_sum in generated_summaries]
    print(qa_g_a)
    qa_g_t = [qa_gen(truth_sum) for truth_sum in ground_truth_summaries]
    print(qa_g_t)

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
    return reward


def main():
    if os.path.exists(x_sum_path):
        df = pd.read_pickle(x_sum_path)
    else:
        df = prepare_data()
        # TO-DO increase length of tokens
        df['tokens'] = df['document'].progress_apply(
            lambda x: summary_tokenizer.encode(x, return_tensors="pt", truncation=True, padding="max_length").to(device)[0, :])
        print(df.head(1))
        print("-------------------")
        # print(df[0, :])
        # print(df['tokens'][0].length)
        df['query'] = df['tokens'].progress_apply(lambda x: summary_tokenizer.decode(x))
        df.to_pickle(x_sum_path)
        print(df.head(1))

    ppo_trainer = PPOTrainer(summary_model, summary_model_ref)
    for epoch in tqdm(range(int(steps / batch_size))):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()

        df_batch = df.sample(batch_size)
        game_data['query'] = df_batch['query'].tolist()
        query_tensors = torch.stack(df_batch['tokens'].tolist())
        ground_truth_sum = df_batch['summary'].tolist()
        print(f'query_tensors shape ', query_tensors.shape)

        t = time.time()
        response_tensors = []
        response_tensors_padding = []
        for i in range(int(batch_size / forward_batch_size)):
            response = summary_model.generate(query_tensors[i * forward_batch_size: (i + 1) * forward_batch_size])
            res_pad = []
            for res in response:
                if len(res) < max_sum_token_len:
                    padding = torch.zeros((max_sum_token_len - len(res)), dtype=torch.int8)
                    res_pad.append(torch.cat([res, padding]))
                else:
                    res_pad.append(res[: max_sum_token_len])
            response_tensors.append(response)
            response_tensors_padding.append(res_pad)

        # response_tensors = torch.cat(response_tensors)
        response_tensors_padding = torch.cat(response_tensors_padding)
        game_data['response'] = [summary_tokenizer.decode(response_tensors[i, :]) for i in range(batch_size)]
        timing['time/get_response'] = time.time() - t
        print(game_data['response'])

        t = time.time()
        rewards = []


if __name__ == "__main__":
    # reward_calculation(["summarize: studies have shown that owning a dog is good for you", "42 is the answer to life, universe and everything."], [])
    text4 = "Forrest Gump is a 1994 American comedy-drama film directed by Robert Zemeckis and written by Eric Roth. \
        It is based on the 1986 novel of the same name by Winston Groom and stars Tom Hanks, Robin Wright, Gary Sinise, \
        Mykelti Williamson and Sally Field. The story depicts several decades in the life of Forrest Gump (Hanks), \
        a slow-witted but kind-hearted man from Alabama who witnesses and unwittingly influences several defining \
        historical events in the 20th century United States. The film differs substantially from the novel."
    # gen_answer('What year was Forrest Gump released?', text4)
    context = "The US has passed the peak on new coronavirus cases, " \
              "President Donald Trump said and predicted that some states would reopen this month." \
              "The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, " \
              "the highest for any country in the world."

    questions = ["What was President Donald Trump's prediction?",
                 "How many deaths have been reported from the virus?",
                 "How many cases have been reported in the United States?"]

    # gen_answer(questions, context)
    # print(norm_levenshtein("test", "team"))
    main()
