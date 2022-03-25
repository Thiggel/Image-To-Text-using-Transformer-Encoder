from torch.utils.data import Dataset
from typing import List, Union
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch import tensor, Tensor, long


class ImageTextDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_text(self, data: List, text_key: Union[str, int]) -> Tensor:
        # pad each token list so that it has the same length
        return pad_sequence([
            # create numeric token lists out of raw text
            self.create_vocab_indices(item[text_key])
            for item in data
        ], padding_value=0).transpose(0, 1)

    def create_vocab_indices(self, caption: str) -> Tensor:
        # convert a string to a list of token indices in the vocab
        return tensor(self.tokenizer(caption).input_ids, dtype=long)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self) -> int:
        pass
