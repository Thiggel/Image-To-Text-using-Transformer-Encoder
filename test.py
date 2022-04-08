
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
from torch import tensor
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.ImageTextDataset import ImageTextDataset
from model.UnifiedTransformer import UnifiedTransformer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

if __name__ == '__main__':
    model = UnifiedTransformer(
        num_classes=1
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.PILToTensor()
    ])

    t = ImageTextDataset()

    # We try to fit the model just on two data points.
    # If we fit it on one, it works, but even two doesn't work. Why?
    image = transform(Image.open('cat.jpg')).float()
    image2 = transform(Image.open('dog.jpg')).float()
    images = torch.cat((image.unsqueeze(0), image2.unsqueeze(0)))
    text = t.preprocess_text([["A cat lying on a table."], ["A dog lying on grass."]], 0)

    optimizer = Adam(model.parameters(), lr=0.0001)

    loss_fn = torch.nn.BCELoss()

    while True:
        optimizer.zero_grad()

        output = model(images, text)

        #from torchviz import make_dot
        #import os

        #make_dot(output.mean(), params=dict(model.named_parameters())).render(directory=os.getcwd(), view=True)

        #exit()

        target = tensor([[1], [0]], dtype=torch.float)

        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()