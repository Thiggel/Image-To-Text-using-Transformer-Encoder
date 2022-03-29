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

if __name__ == '__main__':

    model = UnifiedTransformer(
        num_classes=2,
    )

    from PIL import Image
    import torchvision.transforms as transforms
    from torch.nn.functional import cross_entropy
    from torch import tensor
    import torch
    from torch.optim import Adam, Optimizer

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.PILToTensor()
    ])

    image = transform(Image.open('cat.jpg')).float()

    optimizer = Adam(model.parameters(), lr=0.0001)

    loss_fn = torch.nn.NLLLoss()

    while True:
        output = model(image.unsqueeze(0), ["A cat lying on a table."])

        target = tensor([1, 0]).unsqueeze(0)

        print(output.shape, target.shape)

        loss = loss_fn(output.unsqueeze(0), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(output, loss)

    exit()
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--image-embedding')
    arguments = parser.parse_args()

    # create directory for saved models if not exists
    saved_dir = 'saved'
    if not exists(saved_dir):
        mkdir(saved_dir)

    # using the command line arguments, two hyper parameters can be set
    # 1. the dataset and 2. whether images are processed using convolution
    # or pure attention
    data_module = CocoDataModule() if arguments.dataset == 'coco' else VisualGenomeDataModule()
    convolutional_embedding = arguments.image_embedding == 'convolutional'

    model = UnifiedTransformer(
        num_classes=data_module.num_classes,
        filename=f'{saved_dir}/{dumps(vars(arguments))}.pt',
        convolutional_embedding=convolutional_embedding
    )

    print('Hyper Parameters: ', arguments)

    # if a with the given hyper parameters model already exists,
    # then we load it
    model.load()

    trainer = Trainer(
        max_epochs=300,
        gpus=(-1 if cuda.is_available() else 0),
        callbacks=[EarlyStopping(monitor="val_loss")]
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
