from numpy.random import seed
from torch import manual_seed, device, cuda

from UnifiedTransformerCoco import UnifiedTransformer
from Trainer import Trainer


seed(0)
manual_seed(0)


def main():
    N_EPOCHS = 60
    LR = 0.01

    model = UnifiedTransformer(
        input_shape=(3, 224, 224),
        patch_size=(16, 16),
        embed_dim=512,
        n_heads=8,
        output_dim=1,
        learning_rate=LR
    )

    model.to(device("cuda:0" if cuda.is_available() else "cpu"))

    trainer = Trainer(
        model=model,
        n_epochs=N_EPOCHS
    )

    trainer.fit()

    trainer.test()


if __name__ == '__main__':
    main()
