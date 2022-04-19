from pytorch_lightning import LightningModule
from torch.nn import Embedding, Linear, Sigmoid, Softmax, Parameter, CrossEntropyLoss, BCELoss
from torch import Tensor, zeros, cat, round, save, load
from typing import Tuple, List
from transformer_encoder import TransformerEncoder
from transformer_encoder.utils import PositionalEncoding
from torch.optim import Adam, Optimizer
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.utilities.types import LRSchedulerType
from os.path import isfile
from torchmetrics import Accuracy

from PatchEmbedding import PatchEmbedding


class Model(LightningModule):
    def __init__(
            self,
            vocab_size: int,
            image_shape: Tuple = (3, 224, 224),
            patch_size: int = 16,
            embed_dim: int = 512,
            hidden_size: int = 2048,
            num_heads: int = 8,
            depth: int = 6,
            output_dim: int = 1,
            dropout: float = 0.1,
            learning_rate: float = 1e-3,
            pad_token: int = 0,
            filename: str = 'model.pt'
    ) -> None:
        super().__init__()

        self.filename = filename

        self.image_embedding = PatchEmbedding(
            image_shape, patch_size, embed_dim
        )

        self.text_embedding = Embedding(vocab_size, embed_dim)

        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        self.class_token = Parameter(zeros(1, 1, embed_dim))

        self.encoder = TransformerEncoder(
            embed_dim, hidden_size, num_heads, depth, dropout
        )

        self.linear = Linear(embed_dim, output_dim)

        self.output_dim = output_dim

        self.output_activation = Softmax(dim=1) if output_dim > 1 else Sigmoid()

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=10, max_epochs=200)

        self.loss_fn = CrossEntropyLoss() if output_dim > 1 else BCELoss()

        self.pad_token = pad_token

        self.accuracy = Accuracy()

    def forward(self, images: Tensor, text: Tensor, mask: Tensor) -> Tensor:
        # convert images to patches and linearly project
        images_embedded = self.image_embedding(images)

        text_embedded = self.text_embedding(text)

        # replicate class token N times
        class_tokens = self.class_token.expand(text.shape[0], -1, -1)

        # concatenate the three tensors into one sequence to be further processed
        concatenated = cat((class_tokens, text_embedded, images_embedded), dim=1)

        # add positional encoding to account for order invariance of transformer
        positionally_encoded = self.positional_encoding(concatenated)

        # feed through transformer encoder
        encoded = self.encoder(positionally_encoded, mask)

        # only take the class tokens
        final_class_tokens = encoded[:, 0]

        # feed class tokens through MLP
        class_probs = self.output_activation(
            self.linear(final_class_tokens)
        )

        if self.output_dim == 1:
            class_probs = class_probs.flatten()

        # add dropout to prevent overfitting
        return class_probs

    def create_pad_mask(self, text: Tensor) -> Tensor:
        num_patches = self.image_embedding.num_patches

        # + 1 for the class tokens
        images_mask = Tensor([True]).repeat(text.shape[0], num_patches + 1)

        captions_mask = text != self.pad_token

        full_mask = cat((images_mask.cuda(), captions_mask.cuda()), dim=1)

        return full_mask

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerType]]:
        """
        Decide what optimizer and scheduler Pytorch Lightning should use
        :return: a Tuple of a list of optimizers and a list of schedulers,
        in our case we just use one of each
        """

        scheduler = {
            'scheduler': self.scheduler,
            'monitor': 'train_loss'
        }

        return [self.optimizer], [scheduler]

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        A training step for Pytorch Lightning
        :param batch: Batch of Images, Text and Targets
        :param batch_idx: index of that batch in the dataset
        :return: The loss for that batch
        """

        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions, self.create_pad_mask(captions))
        loss = self.loss_fn(predicted, targets.float())

        self.log('train_loss', loss)
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        A validation step for Pytorch Lightning
        :param batch: Batch of Images, Text and Targets
        :param batch_idx: index of that batch in the dataset
        :return: The loss for that batch
        """
        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions, self.create_pad_mask(captions))
        loss = self.loss_fn(predicted, targets.float())
        accuracy = self.accuracy(round(predicted), targets)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        A test step for Pytorch Lightning
        :param batch: Batch of Images, Text and Targets
        :param batch_idx: index of that batch in the dataset
        :return: The loss for that batch
        """
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, _) -> None:
        """
        Hook that fires after each epoch.
        """

        # save the model after every epoch
        self.save()

    def save(self) -> None:
        """
        Save model to given filename
        """
        save(self.state_dict(), self.filename)

    def load(self) -> None:
        """
        Load model from given filename
        """
        if isfile(self.filename):
            self.load_state_dict(load(self.filename))
