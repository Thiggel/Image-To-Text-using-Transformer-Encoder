from transformers import BertModel
from torch import Tensor
from torch.nn import Module
from typing import Dict


class TextEncoder(Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, x: Dict) -> Tensor:
        return self.model(**x).last_hidden_state
