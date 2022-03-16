import optuna
from pytorch_lightning import Trainer
from optuna.trial import Trial
from json import dumps
from os.path import exists
from os import mkdir

from datasets.coco.CocoDataModule import CocoDataModule
from datasets.visualgenome.VisualGenomeDataModule import VisualGenomeDataModule
from model.UnifiedTransformer import UnifiedTransformer


def objective(trial: Trial) -> float:
    # Define the hyper parameters: the numbers represent ranges out of which
    # optuna is going to sample values in a random grid search
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 6, 18)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # We are testing our model in four different settings:
    # 1. We compare two settings, (a) with a patch embedding and (b) with a
    # convolutional embedding
    has_convolutional_embedding = trial.suggest_categorical('has_convolutional_embedding', [True, False])
    # 2. We're testing our model on two different datasets: Coco (true/false captions for images)
    # and Visual Genome (Visual Question Answering)
    dataset = trial.suggest_categorical('data_module', ['Coco', 'Visual Genome'])

    data_module = CocoDataModule() if dataset == 'Coco' else VisualGenomeDataModule()

    # create directory for saved models if not exists
    saved_dir = 'saved'
    if not exists(saved_dir):
        mkdir(saved_dir)
    
    # log hyper parameters
    hyper_parameters = dict(
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        conv_embed=has_convolutional_embedding,
        dataset=dataset
    )

    print("Hyper Parameters: ", hyper_parameters)

    model = UnifiedTransformer(
        image_size=data_module.image_size,
        num_tokens=data_module.vocab_size,
        sequence_length=data_module.sequence_length,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        num_classes=data_module.num_classes,
        filename=f'{saved_dir}/{dumps(hyper_parameters)}.pt',

        # In the experiment, a patch embedding
        # will be contrasted to a convolutional
        # embedding
        convolutional_embedding=has_convolutional_embedding
    )

    model.save()

    trainer = Trainer(max_epochs=15, gpus=-1)

    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
