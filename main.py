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
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--image-embedding')
    arguments = parser.parse_args()

    # create directory for saved transformer if not exists
    saved_dir = 'saved'
    if not exists(saved_dir):
        mkdir(saved_dir)

    # using the command line arguments, two hyper parameters can be set
    # 1. the dataset and 2. whether images are processed using convolution
    # or pure attention
    print("Loading and preprocessing the data set...", end=" ")
    data_module = CocoDataModule() if arguments.dataset == 'coco' else VisualGenomeDataModule()
    convolutional_embedding = arguments.image_embedding == 'convolutional'
    print("Done!")

    print("Loading model...", end=" ")
    model = UnifiedTransformer(
        num_classes=data_module.num_classes,
        filename=f'{saved_dir}/{dumps(vars(arguments))}.pt',
        convolutional_embedding=convolutional_embedding
    )
    
    print('Done! Hyper Parameters: ', arguments)

    # if a with the given hyper parameters model already exists,
    # then we load it
    model.load()

    trainer = Trainer(
        max_epochs=300,
        gpus=(-1 if cuda.is_available() else 0)
    )

    # train model on data set
    trainer.fit(model, data_module)

    # lastly, test the model on the test set
    trainer.test(model, data_module)
