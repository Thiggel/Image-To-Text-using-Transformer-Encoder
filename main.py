import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.trial import Trial
from json import dumps
from os.path import exists
from os import mkdir

from datasets.visualgenome.VisualGenomeDataModule import VisualGenomeDataModule
from model.UnifiedTransformer import UnifiedTransformer


def objective(trial: Trial) -> float:
    # fix number of encoder layers at 6
    num_encoder_layers = 6

    # Define the hyper parameters: the numbers represent ranges out of which
    # optuna is going to sample values in a random grid search
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.5])

    # We are testing our model in four different settings:
    # 1. We compare two settings, (a) with a patch embedding and (b) with a
    # convolutional embedding
    has_convolutional_embedding = trial.suggest_categorical('has_convolutional_embedding', [True, False])

    data_module = VisualGenomeDataModule()

    # # create directory for saved models if not exists
    # saved_dir = 'saved'
    # if not exists(saved_dir):
    #     mkdir(saved_dir)
    #
    # # log hyper parameters
    # hyper_parameters = dict(
    #     dropout=dropout,
    #     conv_embed=has_convolutional_embedding,
    # )
    #
    # print("Hyper Parameters: ", hyper_parameters)
    #
    # model = UnifiedTransformer(
    #     image_size=data_module.image_size,
    #     num_tokens=data_module.vocab_size,
    #     sequence_length=data_module.sequence_length,
    #     num_encoder_layers=num_encoder_layers,
    #     dropout=dropout,
    #     num_classes=data_module.num_classes,
    #     filename=f'{saved_dir}/{dumps(hyper_parameters)}.pt',
    #
    #     # In the experiment, a patch embedding
    #     # will be contrasted to a convolutional
    #     # embedding
    #     convolutional_embedding=has_convolutional_embedding
    # )
    #
    # # if a with the given hyper parameters model already exists,
    # # then we load it
    # model.load()
    #
    # trainer = Trainer(max_epochs=300, gpus=-1, callbacks=[EarlyStopping(monitor="val_loss")])
    #
    # trainer.fit(model, data_module)
    #
    # return trainer.callback_metrics["val_acc"].item()


if __name__ == '__main__':
   data_module = VisualGenomeDataModule()   

   # study = optuna.create_study(
   #     study_name='multimodal_transformer_study',
   #     storage='sqlite:///study.db',
   #     load_if_exists=True
   # )

   # study.optimize(objective, n_trials=30)

