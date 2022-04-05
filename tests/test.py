from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import cuda
from os.path import exists
from os import mkdir
from argparse import ArgumentParser
from json import dumps

from datasets.coco.CocoDataModule import CocoDataModule
from datasets.visualgenome.VisualGenomeDataModule import VisualGenomeDataModule
from model.UnifiedTransformer import UnifiedTransformer

from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from torch import long, tensor

from datasets.ImageTextDataset import ImageTextDataset


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

if __name__ == '__main__':

    model = UnifiedTransformer(
        num_classes=2,
    )

    from PIL import Image
    import torchvision.transforms as transforms
    from torch.nn.functional import cross_entropy
    from torch import tensor
    import torch
    from torch.optim import Adam, Optimizer, SGD
    from torch.optim.lr_scheduler import ReduceLROnPlateau


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.PILToTensor()
    ])

    t = ImageTextDataset()

    image = transform(Image.open('cat.jpg')).float()
    image2 = transform(Image.open('dog.jpg')).float()
    images = torch.cat((image.unsqueeze(0), image2.unsqueeze(0)))
    text = t.preprocess_text([["A cat lying on a table."], ["A dog lying on grass."]], 0)

    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    loss_fn = torch.nn.CrossEntropyLoss()

    while True:
        output = model(images, text)

        target = tensor([1, 0])

        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        print(output, loss)
        print(scheduler._last_lr)