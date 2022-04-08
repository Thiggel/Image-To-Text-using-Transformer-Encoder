from transformers import BertModel
from torch import Tensor

from model.FrozenModule import FrozenModule


class TextBackbone(FrozenModule):
    def __init__(self) -> None:
        """
        Bert module without last layer - outputs last hidden state sequence.
        Can be complemented with an additional layer and then fine-tuned for
        specific task.
        """
        model = BertModel.from_pretrained("bert-base-uncased")

        super().__init__(model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Output text sequence projected into some latent space to receive
        text features that can be used later on
        :param x: Batch of tokenized text sequences
        :return: last hidden state of BERT
        """
        return self.model(x).last_hidden_state
