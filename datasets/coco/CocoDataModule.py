import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.coco.CocoTrueAndFalseCaptions import CocoTrueAndFalseCaptions
from datasets.ImageTextDataModule import ImageTextDataModule


class CocoDataModule(ImageTextDataModule):

    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.full_dir = 'train2017'
        self.full_url = 'http://images.cocodataset.org/zips/train2017.zip'
        self.full_annotations_file = 'annotations/captions_train2017.json'

        self.annotations_dir = 'annotations'
        self.annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

        # download datasets if not yet downloaded
        self.load_dataset()

        coco_full = CocoTrueAndFalseCaptions(
            image_dir=self.full_dir,
            annotations_file=self.full_annotations_file,
            transform=transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.PILToTensor()
            ])
        )
        self.sequence_length = coco_full.sequence_length
        self.num_classes = coco_full.num_classes

        # split into train/test/val with 70/20/10 ratio
        self.coco_train, self.coco_test, self.coco_val = self.split_dataset(coco_full)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.coco_train, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.coco_val, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.coco_test, batch_size=self.batch_size, num_workers=12)

    def load_dataset(self) -> None:
        self.download_if_not_exists(self.full_dir, self.full_url)
        self.download_if_not_exists(self.annotations_dir, self.annotations_url)
