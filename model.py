from constants import PEGASUS, GPT2
from ppo.gpt2 import GPT2HeadWithValueModel
from ppo.pegasus import PegasusHeadWithValueModel
from transformers import (
    PegasusTokenizer,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)


def init_summary_model(model_type, model_path, pretrained_model_path, device, use_cuda_ref_model=False):
    if model_type == PEGASUS:
        model = PegasusHeadWithValueModel.from_pretrained(model_path).to(device)
        model_ref = PegasusHeadWithValueModel.from_pretrained(pretrained_model_path)
        if use_cuda_ref_model:
            model_ref.to(device)
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    elif model_type == GPT2:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2HeadWithValueModel.from_pretrained(pretrained_model_name_or_path=model_path).to(device)
        model_ref = GPT2HeadWithValueModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_path).to(device)
    else:
        raise NotImplementedError
    tokenizer.pad_token = " "

    return model, model_ref, tokenizer


def init_qa_model(device, use_cuda=False):
    qa_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
    qa_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")
    ans_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
    ans_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl")
    gen_answer_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    gen_answer_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    if use_cuda:
        qa_model.to(device)
        ans_model.to(device)
        gen_answer_model.to(device)

    return qa_tokenizer, qa_model, ans_tokenizer, ans_model, gen_answer_tokenizer, gen_answer_model


