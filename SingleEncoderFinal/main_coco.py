from numpy.random import seed
from torch import manual_seed, device, cuda
from json import dumps

from UnifiedTransformerCoco import UnifiedTransformerCoco
from Trainer import Trainer


seed(0)
manual_seed(0)


def main():
    DATASET = 'COCO'
    N_EPOCHS = 60
    LR = 0.01
    CONV_LAYERS = 0

    filename = dumps({
        'dataset': DATASET,
        'lr': LR,
        'conv_layers': CONV_LAYERS
    }) + '.pt'

    model = UnifiedTransformerCoco(
        input_shape=(3, 224, 224),
        patch_size=(16, 16),
        embed_dim=512,
        n_heads=8,
        output_dim=1,
        learning_rate=LR,
        conv_layers=CONV_LAYERS
    )

    model.to(device("cuda:0" if cuda.is_available() else "cpu"))

    trainer = Trainer(
        model=model,
        n_epochs=N_EPOCHS,
        checkpoint_filename=filename
    )

    trainer.fit()

    trainer.test()


if __name__ == '__main__':
    main()
