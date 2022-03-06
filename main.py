import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from CocoTrueAndFalseCaptions import CocoTrueAndFalseCaptions
from UnifiedTransformer import UnifiedTransformer


dataset = CocoTrueAndFalseCaptions(
    image_dir='val2017',
    annotations_file='annotations/captions_val2017.json',
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.PILToTensor()
    ])
)

train_dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

val_dataloader = train_dataloader

model = UnifiedTransformer(
    image_size=128,
    num_tokens=dataset.vocab_size,
    sequence_length=dataset.sequence_length,
    num_encoder_layers=6,
    num_classes=2
)

trainer = Trainer()
trainer.fit(model, train_dataloader, val_dataloader)
