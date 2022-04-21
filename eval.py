import torch
from model import init_summary_model
from ppo.utils import respond_to_batch
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import argparse


def main():
    result = {}
    result_ref = {}
    for me in measure:
        result[me] = {'precision': 0, 'recall': 0, 'f_measure': 0}
        result_ref[me] = {'precision': 0, 'recall': 0, 'f_measure': 0}

    for i in tqdm(range(0, df.shape[0], batch_size)):
        print(f"--------------- batch {i} ---------------\n")
        df_batch = df.iloc[i:min(i + batch_size, df.shape[0])]
        df_batch.reset_index(inplace=True)
        print(f"df_batch: {df_batch}\n")
        encoding = summary_tokenizer.batch_encode_plus(df_batch['document'], return_tensors="pt", truncation=True,
                                                       padding="max_length", max_length=max_token_len).to(device)
        response_ids = respond_to_batch(summary_model, encoding['input_ids'].to(device), txt_len=80)
        response_ids_ref = respond_to_batch(summary_model_ref, encoding['input_ids'].to(device), txt_len=80)

        response = summary_tokenizer.batch_decode(response_ids)
        response_ref = summary_tokenizer.batch_decode(response_ids_ref)
        for i_r, res in enumerate(response):
            print(f"summary: {str(df_batch['summary'][i_r])}\n\nresponse: {res}\n\nresponse_ref: {response_ref[i_r]}\n")
            res_score = scorer.score(res, df_batch['summary'][i_r])
            ref_score = scorer.score(response_ref[i_r], df_batch['summary'][i_r])
            for me in measure:
                result[me]['precision'] += res_score[me].precision
                result[me]['recall'] += res_score[me].recall
                result[me]['f_measure'] += res_score[me].fmeasure

                result_ref[me]['precision'] += ref_score[me].precision
                result_ref[me]['recall'] += ref_score[me].recall
                result_ref[me]['f_measure'] += ref_score[me].fmeasure
        print(f"result: {result}\n\nresult_ref: {result_ref}\n")

    for me in measure:
        result[me]['precision'] /= df.shape[0]
        result[me]['recall'] /= df.shape[0]
        result[me]['f_measure'] /= df.shape[0]

        result_ref[me]['precision'] /= df.shape[0]
        result_ref[me]['recall'] /= df.shape[0]
        result_ref[me]['f_measure'] /= df.shape[0]

    print(result)
    print(result_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Abstractive Summarization using Question Answering Rewards Evaluation")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_ref_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--max_token_len", type=str, default=512)
    parser.add_argument("--batch_size", type=str, default=2)

    args = parser.parse_args()
    model_path = args.model_path
    model_ref_path = args.model_ref_path
    model_type = args.model_type
    batch_size = args.batch_size
    max_token_len = args.max_token_len

    ds = load_dataset("xsum", split="test")
    ds.set_format('pandas')
    df = ds[:]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary_model, summary_model_ref, summary_tokenizer = init_summary_model(model_type, model_path, model_ref_path, device)

    measure = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(measure)

    main()
