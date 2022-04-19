from numpy.random import seed
from torch import manual_seed, device, cuda

from UnifiedTransformer import UnifiedTransformer
from Trainer import Trainer


seed(0)
manual_seed(0)


def main():
    N_EPOCHS = 5
    LR = 0.01

    model = UnifiedTransformer(
        input_shape=(1, 28, 28),
        patch_size=(4, 4),
        embed_dim=20,
        n_heads=2,
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
