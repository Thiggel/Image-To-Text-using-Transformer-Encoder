from pytorch_lightning import Trainer
from torch import cuda

from CocoDataModule import CocoDataModule
from model import Model


if __name__ == '__main__':
    data_module = CocoDataModule(
        train_images_dir='../train2017',
        train_annotations_file='../annotations/captions_train2017.json',
        val_images_dir='../val2017',
        val_annotations_file='../annotations/captions_val2017.json',
        batch_size=32
    )

    model = Model(
        vocab_size=data_module.vocab_size,
        pad_token=data_module.pad_token
    )

    trainer = Trainer(
        max_epochs=500,
        gpus=(-1 if cuda.is_available() else 0)
    )

    # train model on data set
    trainer.fit(model, data_module)

    # lastly, test the model on the test set
    trainer.test(model, data_module)
