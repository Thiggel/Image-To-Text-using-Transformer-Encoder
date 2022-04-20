from numpy.random import seed
from torch import manual_seed, device, cuda
from json import dumps
from argparse import ArgumentParser
from optuna import Trial, create_study

from UnifiedTransformerMnist import UnifiedTransformerMnist
from UnifiedTransformerCoco import UnifiedTransformerCoco
from Trainer import Trainer


seed(0)
manual_seed(0)


def objective(trial: Trial) -> float:
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--image-embedding')
    arguments = parser.parse_args()

    MAX_EPOCHS = 50
    LR = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    CONV_LAYERS = 0 if arguments.image_embedding != 'convolutional' else trial.suggest_float('conv_layers', 1, 5)

    filename = dumps({
        'dataset': arguments.dataset,
        'lr': LR,
        'conv_layers': CONV_LAYERS
    }) + '.pt'

    models = {
        'MNIST': UnifiedTransformerMnist(
            input_shape=(1, 28, 28),
            patch_size=(14, 14),
            embed_dim=20,
            n_heads=2,
            output_dim=1,
            learning_rate=LR,
            conv_layers=CONV_LAYERS
        ),

        'COCO': UnifiedTransformerCoco(
            input_shape=(3, 224, 224),
            patch_size=(16, 16),
            embed_dim=512,
            n_heads=8,
            output_dim=1,
            learning_rate=LR,
            conv_layers=CONV_LAYERS
        )
    }

    model = models[arguments.dataset]

    model.to(device("cuda:0" if cuda.is_available() else "cpu"))

    trainer = Trainer(
        model=model,
        n_epochs=MAX_EPOCHS,
        checkpoint_filename=filename
    )

    trainer.fit()

    test_loss = trainer.test()

    return test_loss


if __name__ == '__main__':
    study = create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
