from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import cuda
from os.path import exists
from os import mkdir

from datasets.coco.CocoDataModule import CocoDataModule
from UnifiedTransformer.UnifiedTransformer import UnifiedTransformer

if __name__ == '__main__':
    # create directory for saved models if not exists
    saved_dir = 'saved'
    if not exists(saved_dir):
        mkdir(saved_dir)

    data_module = CocoDataModule()

    model = UnifiedTransformer(
        num_classes=data_module.num_classes,
        filename=f'{saved_dir}/model.pt',
    )

    # if a with the given hyper parameters model already exists,
    # then we load it
    model.load()

    trainer = Trainer(
        max_epochs=300,
        gpus=(-1 if cuda.is_available() else 0),
       # callbacks=[EarlyStopping(monitor="val_loss")]
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
