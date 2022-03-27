from transformers import ViTModel
from torch import Tensor
from torch.nn import Module


class VisionEncoder(Module):
    def __init__(self, convolutional: bool = False) -> None:
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") \
            if not convolutional \
            else ViTModel.from_pretrained("facebook/detr-resnet-50")

        super().__init__(model)

    def forward(self, x: Tensor) -> Tensor:
        # the forward method is redefined so that the last part of the model
        # that uses the class token to classify the input is skipped.
        # Therefore, only the processed sequence returned by the
        # encoder is used as the output of this layer

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
