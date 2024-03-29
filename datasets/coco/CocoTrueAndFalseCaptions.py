from torch import Tensor, tensor, float
from typing import Optional, Callable, Tuple, Any, List
from PIL import Image
from json import load
from os.path import join

from datasets.ImageTextDataset import ImageTextDataset


class CocoTrueAndFalseCaptions(ImageTextDataset):

    def __init__(
            self,
            image_dir: str,
            annotations_file: str,
            transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.annotations = self.load_annotations(annotations_file)
        self.annotations_size = len(self.annotations)

        self.image_dir = image_dir
        self.transform = transform

        self.captions = self.preprocess_text(self.annotations, 'caption')

        # the output should just be a 1 or a 0,
        # denoting the truth value of the caption
        # in regard to the image
        self.num_classes = 1

    @staticmethod
    def load_annotations(filename: str) -> List[Any]:
        with open(filename, 'r') as file:
            return load(file)['annotations']

    def load_image(self, index: int) -> Image.Image:
        filename = f"{self.annotations[index]['image_id']:012d}.jpg"

        image = Image.open(join(self.image_dir, filename)).convert("RGB")

        return image

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Tensor], Tensor]:
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

        if isinstance(image, Tensor):
            image = image.float()

        # make sure we don't choose a correct caption as the false one.
        # Therefore, we take a caption which is 100 indexes further down
        # the array

        caption = self.captions[
            index if target == 1
            else (index + 100) % self.annotations_size
        ]

        return (image, caption), tensor(target, dtype=float)

    @property
    def sequence_length(self) -> int:
        return self.captions.shape[1]

    def __len__(self) -> int:
        # there is one true and one false caption
        # for each item in the annotations file
        return 2 * len(self.annotations)
