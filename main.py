import optuna
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from optuna.trial import Trial

from CocoTrueAndFalseCaptions import CocoTrueAndFalseCaptions
from UnifiedTransformer import UnifiedTransformer

def objective(trial: Trial) -> float:

    dataset = CocoTrueAndFalseCaptions(
        image_dir='val2017',
        annotations_file='annotations/captions_val2017.json',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.PILToTensor()
        ])
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    val_dataloader = train_dataloader

    num_encoder_layers = trial.suggest_int('num_encoder_layers', 6, 24)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    model = UnifiedTransformer(
        image_size=128,
        num_tokens=dataset.vocab_size,
        sequence_length=dataset.sequence_length,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        num_classes=2
    )

    trainer = Trainer()

    hyper_parameters = dict(num_encoder_layers=num_encoder_layers, dropout=dropout)
    trainer.logger.log_hyperparams(hyper_parameters)

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
