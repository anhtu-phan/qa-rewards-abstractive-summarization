from transformers import PegasusForConditionalGeneration
from torch import nn
from torch.nn import Identity
import torch


class ValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.detach_head = False

        self.summary = Identity()
        self.summary = nn.Linear(config.hidden_size, config.num_labels)

        self.activation = Identity()
        self.activation = nn.Tanh()

        self.first_dropout = Identity()
        self.first_dropout = nn.Dropout(config.dropout)

        self.last_dropout = Identity()
        self.last_dropout = nn.Dropout(config.activation_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class PegasusHeadWithValueModel(PegasusForConditionalGeneration):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        config.output_hidden_states = True
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
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = torch.cat(
            [transformer_outputs.encoder_last_hidden_state, transformer_outputs.decoder_hidden_states[-1]], dim=1)

        lm_logits = self.lm_head(transformer_outputs.encoder_last_hidden_state)
        lm_logits = torch.cat([lm_logits, transformer_outputs.logits], dim=1)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + (transformer_outputs.past_key_values, ) + (value,)
        # outputs = (transformer_outputs.logits,) + (transformer_outputs.past_key_values,) + (value,)

        return outputs
