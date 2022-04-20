from pytorch_lightning import LightningModule
from torch import load, no_grad, cuda
from torch.utils.data import DataLoader
from typing import Tuple
from EarlyStopping import EarlyStopping


class Trainer:
    def __init__(self, model: LightningModule, n_epochs: int = 5) -> None:
        self.model = model
        self.n_epochs = n_epochs

        self.early_stopping = EarlyStopping(patience=10, verbose=True)

    def fit(self) -> None:
        for epoch in range(self.n_epochs):
            train_loss = 0.0
            for batch_idx, batch in enumerate(self.model.train_dataloader()):
                loss = self.model.training_step(batch, batch_idx)
                train_loss += loss.item()

                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.n_epochs} loss: {train_loss:.2f}")

            if cuda.is_available():
                cuda.memory_summary(device=None, abbreviated=False)

            self.validate()

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def test_validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        test_accuracy = 0.0
        test_loss = 0.0

        with no_grad():
            for batch_idx, batch in enumerate(dataloader):
                loss, acc = self.model.test_step(batch, batch_idx)
                test_loss += loss
                test_accuracy += acc

        test_accuracy /= len(dataloader)

        return test_loss, test_accuracy

    def validate(self) -> None:
        test_loss, test_accuracy = self.test_validate(self.model.val_dataloader())

        self.early_stopping(test_loss, self.model)

        print(f"Validation loss: {test_loss:.2f}")
        print(f"Validation accuracy: {test_accuracy * 100:.2f}%")

    def test(self) -> None:
        self.model.load_state_dict(load('checkpoint.pt'))

        test_loss, test_accuracy = self.test_validate(self.model.val_dataloader())

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
