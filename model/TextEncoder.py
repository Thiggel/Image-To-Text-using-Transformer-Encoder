from transformers import BertModel, BertTokenizer
from torch import Tensor
from torch.nn import Module


class TextEncoder(Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def forward(self, x: str) -> Tensor:
        x = self.tokenizer(x, return_tensors="pt", padding=True)

        return self.model(**x).last_hidden_state
