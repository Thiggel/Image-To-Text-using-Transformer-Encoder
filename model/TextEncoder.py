from transformers import BertModel
from torch import Tensor, zeros, ones, long

from model.FrozenModule import FrozenModule


class TextEncoder(FrozenModule):
    def __init__(self) -> None:
        super().__init__(BertModel.from_pretrained("bert-base-uncased"))

    def forward(self, x: Tensor) -> Tensor:
        assert x is not None, "No input specified"

        input_shape = x.size()

        batch_size, seq_length = input_shape
        device = x.device

        # past_key_values_length
        past_key_values_length = 0

        attention_mask = ones((batch_size, seq_length + past_key_values_length), device=device)

        if hasattr(self.model.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.model.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = zeros(input_shape, dtype=long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.model.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.model.get_head_mask(None, self.model.config.num_hidden_layers)

        embedding_output = self.model.embeddings(
            input_ids=x,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        return self.model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.use_return_dict,
        ).last_hidden_state
