import torchvision.datasets as dset
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, Subset, DataLoader
from typing import List, Optional
from random import choice
from transformers import BertTokenizer
from torch import tensor
import random

from torch.utils.data.dataset import T_co
from torchvision.datasets import VisionDataset


class CocoCaptionGeneration(VisionDataset):
    def __init__(self, coco: VisionDataset, tokenizer) -> None:
        super().__init__("")

        self.coco = coco
        self.tokenizer = tokenizer
        self.sequence_length = 128

    def __getitem__(self, index) -> T_co:
        image, caption = self.coco[index]
        target = 1

        # randomly put wrong captions in
        if random.randint(0, 1) == 0:
            _, caption = self.coco[(index + 100) % len(self.coco)]
            target = 0

        tokenized = [self.tokenizer('[CLS]').input_ids[1]] + self.tokenizer(choice(caption)).input_ids[0:-1]

        if len(tokenized) > self.sequence_length:
            tokenized = tokenized[:self.sequence_length - 1]

        padded = tokenized + [self.tokenizer('[PAD]').input_ids[1]] * (self.sequence_length - len(tokenized))

        return [image, tensor(padded)], target

    def __len__(self):
        return len(self.coco)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


class CocoDataModule(LightningDataModule):
    def __init__(
            self,
            train_images_dir: str,
            train_annotations_file: str,
            val_images_dir: str,
            val_annotations_file: str,
            batch_size: int = 32
    ) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.train_images_dir = train_images_dir
        self.train_annotations_file = train_annotations_file
        self.val_images_dir = val_images_dir
        self.val_annotations_file = val_annotations_file

        self.train_set, self.test_set, self.val_set = None, None, None

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    def split_dataset(dataset: VisionDataset) -> List[Subset[VisionDataset]]:
        train_size = int(len(dataset) // 1.43)
        test_size = len(dataset) - train_size

        return random_split(dataset, [train_size, test_size])

    def setup(self, stage: Optional[str] = None) -> None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor()
        ])

        train_test_set = CocoCaptionGeneration(dset.CocoCaptions(
            root=self.train_images_dir,
            annFile=self.train_annotations_file,
            transform=transform
        ), self.tokenizer)

        self.train_set, self.test_set = self.split_dataset(train_test_set)

        self.val_set = CocoCaptionGeneration(dset.CocoCaptions(
            root=self.val_images_dir,
            annFile=self.val_annotations_file,
            transform=transform
        ), self.tokenizer)

    @property
    def vocab_size(self):
        return self.val_set.vocab_size

    @property
    def sequence_length(self):
        return self.val_set.sequence_length

    @property
    def pad_token(self):
        return self.tokenizer('[PAD]').input_ids[1]

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=12)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=12)
