from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import cuda
from os.path import exists
from os import mkdir

from datasets.coco.CocoDataModule import CocoDataModule
from model.UnifiedTransformer import UnifiedTransformer

if __name__ == '__main__':
    # create directory for saved models if not exists
    saved_dir = 'saved'
    if not exists(saved_dir):
        mkdir(saved_dir)

    #data_module = CocoDataModule()

    model = UnifiedTransformer(
        num_classes=2,#data_module.num_classes,
        filename=f'{saved_dir}/model.pt',
    )

    # if a with the given hyper parameters model already exists,
    # then we load it
    model.load()

    from PIL import Image
    from torchvision.transforms import PILToTensor, Resize
    from transformers import BertTokenizer
    from torch import tensor
    from torchmetrics import Accuracy

    img = Resize((224, 224))(PILToTensor()(Image.open('val2017/000000000139.jpg'))).float()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    out = model(img.unsqueeze(0), tensor(tokenizer("A man sitting on a bench").input_ids).unsqueeze(0))
    metric = Accuracy()
    acc = metric(out, tensor([1, 1]).unsqueeze(0))

    print(acc)

    #trainer = Trainer(
    #    max_epochs=300,
    #    gpus=(-1 if cuda.is_available() else 0),
    #   # callbacks=[EarlyStopping(monitor="val_loss")]
    #)

    #trainer.fit(model, data_module)
    #trainer.test(model, data_module)
