# https://github.com/lvwerra/trl/blob/5410be61b4eaf832d8bf3e705da49e23b78b50f7/trl/gpt2.py#L113

__all__ = ['respond_to_batch']

from transformers import top_k_top_p_filtering
import torch.nn.functional as F
import torch


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0, eos_token=None, device="cpu"):
    """Sample text from language model."""
    input_ids = queries
    # input_ids = [batch_size, src_len]
    batch_size = input_ids.shape[0]
    decoder_input_ids = None
    if eos_token is not None:
        decoder_input_ids = torch.empty(batch_size, 1).fill_(eos_token).type(torch.LongTensor).to(device)
    for i in range(txt_len):
        # Get Logits
        if decoder_input_ids is not None:
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        else:
            outputs = model(input_ids)

        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # next_token_logits = [batch_size, vocab_size]

        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        if decoder_input_ids is not None:
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)
        else:
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

    # return response size [batch_size, txt_len]
    if decoder_input_ids is not None:
        return decoder_input_ids[:, -txt_len:]
        # return decoder_input_ids

    return input_ids[:, -txt_len:]
