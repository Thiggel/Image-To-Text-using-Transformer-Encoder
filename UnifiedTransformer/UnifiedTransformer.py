from torch import Tensor, cat, zeros, no_grad, save, load
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer, Linear, Dropout, Softmax
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from typing import List, Tuple
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from os.path import isfile

from UnifiedTransformer.VisionEncoder import VisionEncoder
from UnifiedTransformer.TextEncoder import TextEncoder


class UnifiedTransformer(LightningModule):
    def __init__(
            self,
            num_classes: int,
            num_encoder_layers: int = 6,
            nhead: int = 12,
            dropout: float = 0.1,
            learning_rate: float = 0.05,
            filename: str = 'model.pt'
    ) -> None:
        super().__init__()

        self.filename = filename

        self.image_embedding = VisionEncoder()
        self.text_embedding = TextEncoder()

        assert (
            self.image_embedding.model.config.hidden_size == self.text_embedding.model.config.hidden_size,
            "The embedding dimensions for the pretrained image and text encoder must be the same"
        )

        self.d_model = self.image_embedding.model.config.hidden_size

        self.class_token = Parameter(zeros(1, 1, self.d_model))

        # we use a transformer encoder as the main part of the network.
        # There are num_encoder_layers in this encoder.
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dropout=dropout
            ),
            num_encoder_layers
        )

        self.MLP_head = Linear(self.d_model, num_classes)

        self.dropout = Dropout(dropout)

        self.softmax = Softmax(dim=0)

        self.learning_rate = learning_rate

        self.accuracy = Accuracy()

    def forward(self, images: Tensor, text: List[str]) -> Tensor:
        with no_grad():
            images_embedded = self.image_embedding(images)
            text_embedded = self.text_embedding(text)

        x = cat((images_embedded, text_embedded), dim=1)
        x = self.transformer_encoder(x)

        # get the class embeddings out of all embeddings
        final_class_token = x[:, 0]

        # lastly, feed it into the MLP
        x = self.MLP_head(final_class_token)

        # add dropout to prevent overfitting
        x = self.dropout(x)

        # compute probabilities between 0 and 1
        # using the softmax function
        return self.softmax(x)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerType]]:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=3),
            'monitor': 'train_loss'
        }

        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions)
        loss = cross_entropy(predicted, targets)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions)
        loss = cross_entropy(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions)
        loss = cross_entropy(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        self.log('test_loss', loss)
        self.log('test_acc', accuracy)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy)

        # save the model after every epoch
        self.save()

    def save(self) -> None:
        print("Saving model at: " + self.filename)
        save(self.state_dict(), self.filename)

    def load(self) -> None:
        if isfile(self.filename):
            print("Loading model from: " + self.filename)
            self.approximation_function.load_state_dict(load(self.filename))
