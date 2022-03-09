import optuna
from pytorch_lightning import Trainer
from optuna.trial import Trial

from CocoDataModule import CocoDataModule
from VisualGenomeDataModule import VisualGenomeDataModule
from UnifiedTransformer import UnifiedTransformer

def objective(trial: Trial) -> float:
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 6, 24)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    DATASET = 'Coco' # 'Visual Genome'
    data_module = CocoDataModule() if DATASET == 'Coco' else VisualGenomeDataModule()

    model = UnifiedTransformer(
        image_size=data_module.image_size,
        num_tokens=data_module.vocab_size,
        sequence_length=data_module.sequence_length,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        num_classes=data_module.num_classes
    )

    trainer = Trainer()

    hyper_parameters = dict(num_encoder_layers=num_encoder_layers, dropout=dropout)
    print("Hyper Parameters: ", hyper_parameters)

    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
