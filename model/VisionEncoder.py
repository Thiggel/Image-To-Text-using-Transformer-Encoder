from transformers import ViTModel
from torch import Tensor

from model.FrozenModule import FrozenModule


class VisionEncoder(FrozenModule):
    def __init__(self) -> None:
        super().__init__(ViTModel.from_pretrained("google/vit-base-patch16-224-in21k"))

    def forward(self, x: Tensor) -> Tensor:
        assert x is not None, "No input tensor specified"

        head_mask = self.model.get_head_mask(None, self.model.config.num_hidden_layers)

        embedding_output = self.model.embeddings(x, bool_masked_pos=None, interpolate_pos_encoding=None)

        return self.model.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.use_return_dict,
        ).last_hidden_state
