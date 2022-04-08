from transformers import ViTModel
from torch import Tensor

from model.FrozenModule import FrozenModule


class ImageBackbone(FrozenModule):
    def __init__(self, convolutional: bool = False) -> None:
        """
        Image Processing Backbone module, processes images either through
        attention or convolution. Last hidden state is returned such that
        it can be used e.g. in a multi-modal transformer together with the
        last hidden state of a BERT model. Pre-trained transformer are used that
        can be fine-tuned for the task
        :param convolutional: Whether the backbone should process images
        using convolution or attention
        """

        # Either use standard Vision Transformer (ViT) or Detr (probably will be changed)
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") \
            if not convolutional \
            else ViTModel.from_pretrained("facebook/detr-resnet-50")

        super().__init__(model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Output processed image features which can be fed further down
        a model
        :param x: A batch of images
        :return: The last hidden state of the backbone architecture
        """
        return self.model(x).last_hidden_state
