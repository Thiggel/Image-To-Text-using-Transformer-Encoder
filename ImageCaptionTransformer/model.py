from pytorch_lightning import LightningModule
from torch import Tensor
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from typing import List, Tuple
from torchmetrics import Accuracy


class Model(LightningModule):
    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()

        self.model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

        self.model.mlp = torch.nn.Identity()

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

        self.learning_rate = learning_rate

        self.accuracy = Accuracy()

    def forward(self, images, captions) -> Tensor:
        hidden_states = self.model(images.float(), captions, target_mask=None)

        final_class_tokens = hidden_states[:, 0]

        return self.MLP(final_class_tokens).flatten()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerType]]:
        """
        Decide what optimizer and scheduler Pytorch Lightning should use
        :return: a Tuple of a list of optimizers and a list of schedulers,
        in our case we just use one of each
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=3),
            'monitor': 'train_loss'
        }

        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        A training step for Pytorch Lightning
        :param batch: Batch of Images, Text and Targets
        :param batch_idx: index of that batch in the dataset
        :return: The loss for that batch
        """

        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions)

        loss = F.binary_cross_entropy(predicted, targets.float())

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch: Tensor, _) -> Tensor:
        """
        A validation step for Pytorch Lightning
        :param batch: Batch of Images, Text and Targets
        :param batch_idx: index of that batch in the dataset
        :return: The loss for that batch
        """
        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions)
        loss = F.binary_cross_entropy(predicted, targets.float())
        accuracy = self.accuracy(torch.round(predicted), targets)

        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

        print("Validation Loss: ", loss)
        print("Validation Accuracy: ", accuracy)

        return loss

    def test_step(self, batch: Tensor, _) -> Tensor:
        """
        A test step for Pytorch Lightning
        :param batch: Batch of Images, Text and Targets
        :param batch_idx: index of that batch in the dataset
        :return: The loss for that batch
        """
        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions)
        loss = F.binary_cross_entropy(predicted, targets.float())
        accuracy = self.accuracy(torch.round(predicted), targets)

        self.log('test_loss', loss)
        self.log('test_acc', accuracy)

        return loss