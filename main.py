from pytorch_lightning import Trainer
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
        num_classes=10
    )

    import PIL
    import torchvision
    from transformers import BertTokenizer
    import torch

    image = torchvision.transforms.Resize((224, 224))(torchvision.transforms.PILToTensor()(PIL.Image.open('cat.jpg'))).float()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print(model(image.unsqueeze(0), torch.tensor(tokenizer("A cat").input_ids).unsqueeze(0)))

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
    exit()
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
        gpus=(-1 if cuda.is_available() else 0)
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
