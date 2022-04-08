from torch.utils.data import Dataset
from typing import List, Union
from transformers import BertTokenizer
from torch import Tensor


class ImageTextDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_text(self, data: List, text_key: Union[str, int]) -> Tensor:
        return self.tokenizer([item[text_key] for item in data], return_tensors="pt", padding=True).input_ids

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self) -> int:
        pass
