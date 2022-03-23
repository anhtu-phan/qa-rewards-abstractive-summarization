from transformers import PegasusForConditionalGeneration
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.detach_head = False


class PegasusHeadWithValueModel(PegasusForConditionalGeneration):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = PegasusForConditionalGeneration(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.v_head = ValueHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + transformer_outputs[1:] + (value,)

        return outputs
