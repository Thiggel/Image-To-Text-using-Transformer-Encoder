from torch.utils.data import Dataset
from typing import List, Union
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch import tensor, Tensor, long


class ImageTextDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self) -> int:
        pass
