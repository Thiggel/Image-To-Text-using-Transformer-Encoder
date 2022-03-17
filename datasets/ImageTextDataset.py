from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import List, Union
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch import tensor, Tensor, long


class ImageTextDataset(Dataset):

    def __init__(self, sentence_list: List[str]):
        super().__init__()

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.build_vocab(sentence_list)

    def build_vocab(self, sentence_list: List[str]) -> Vocab:
        # use <PAD> token to later pad sequences to same length
        return build_vocab_from_iterator(map(self.tokenizer, sentence_list), specials=['<PAD>'])

    def preprocess_text(self, data: List, text_key: Union[str, int]) -> Tensor:
        # pad each token list so that it has the same length
        return pad_sequence([
            # create numeric token lists out of raw text
            self.create_vocab_indices(item[text_key])
            for item in data
        ], padding_value=self.vocab['<PAD>']).transpose(0, 1)

    def create_vocab_indices(self, caption: str) -> Tensor:
        # convert a string to a list of token indices in the vocab
        return tensor(self.vocab(self.tokenizer(caption)), dtype=long)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        pass
