from torch.utils.data import Dataset
from torch import tensor, Tensor, cat, long
from typing import Optional, Callable, Tuple, Any, List, Dict
from PIL import Image
from json import load
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from os.path import join
from torch.nn.utils.rnn import pad_sequence


class CocoTrueAndFalseCaptions(Dataset):

    def __init__(
            self,
            image_dir: str,
            annotations_file: str,
            transform: Optional[Callable] = None
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        self.annotations = self.load_annotations(annotations_file)
        self.annotations_size = len(self.annotations)

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.build_vocab()

        self.captions = self.preprocess_captions()

    def build_vocab(self) -> Vocab:
        # use <PAD> token to later pad sequences to same length
        vocab = build_vocab_from_iterator(map(self.tokenizer, self.all_captions()), specials=['<PAD>'])

        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def load_annotations(filename: str) -> List[Any]:
        with open(filename, 'r') as file:
            return load(file)['annotations']

    def preprocess_captions(self):
        # pad each token list so that it has the same length
        return pad_sequence([
            # create numeric token lists out of raw text
            self.create_vocab_indices(item['caption'])
            for item in self.annotations
        ], padding_value=self.vocab['<PAD>']).transpose(0, 1)

    def create_vocab_indices(self, caption: str) -> Tensor:
        # convert a string to a list of token indices in the vocab
        return tensor(self.vocab(self.tokenizer(caption)), dtype=long)

    def load_image(self, index: int) -> Image.Image:
        filename = f"{self.annotations[index]['image_id']:012d}.jpg"

        image = Image.open(join(self.image_dir, filename)).convert("RGB")

        return image

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, str], Any]:
        # target is 1 if index is within size of annotations
        # otherwise it is 0, since a wrong caption is chosen
        target = int(index < self.annotations_size)

        # if the index is bigger than the size of the annotations array,
        # we start from the beginning with the images and choose a false
        # caption. As a result, the dataset gets twice as big, but the
        # second half only consists of false captions
        image = self.load_image(index % self.annotations_size)

        if self.transform:
            image = self.transform(image)

        # make sure we don't choose a correct caption as the false one.
        # Therefore, we take a caption which is 100 indexes further down
        # the array
        caption = self.captions[
            index if target == 1
            else (index + 100) % self.annotations_size
        ]

        return (image, caption), target

    @property
    def sequence_length(self):
        return self.captions.shape[1]

    def __len__(self) -> int:
        # there is one true and one false caption
        # for each item in the annotations file
        return 2 * len(self.annotations)

    def all_captions(self) -> List[str]:
        # get an array of all captions in the dataset
        # used to build the vocab
        return list(map(lambda item: item['caption'], self.annotations))
