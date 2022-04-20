from torch import rand, stack, vstack, ones, zeros, cat
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import Tensor
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Sequential, \
    Linear, \
    Softmax, \
    Sigmoid, \
    Embedding, \
    Parameter, \
    BCELoss, \
    CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import random
from typing import Tuple
from torchmetrics import Accuracy

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from PatchEmbedding import PatchEmbedding
from Encoder import Encoder


class UnifiedTransformerMnist(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            patch_size: Tuple[int, int] = (4, 4),
            conv_layers: int = 0,
            text_length: int = 1,
            embed_dim: int = 8,
            n_heads: int = 2,
            vocab_size: int = 10,
            output_dim: int = 1,
            learning_rate: float = 1e-3
    ):
        super(UnifiedTransformerMnist, self).__init__()

        self.output_dim = output_dim

        self.patch_embedding = PatchEmbedding(input_shape, patch_size, embed_dim, conv_layers)

        self.sequence_length = self.patch_embedding.n_patches + text_length + 1

        self.embedding = Embedding(vocab_size, embed_dim)

        self.class_token = Parameter(rand(1, embed_dim))

        self.encoder = Encoder(self.sequence_length, input_shape, n_heads, embed_dim)

        self.MLP = Sequential(
            Linear(embed_dim, output_dim),
            Sigmoid() if output_dim == 1 else Softmax()
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, 60, verbose=True)
        self.loss_fn = BCELoss() if output_dim == 1 else CrossEntropyLoss()

        self.accuracy = Accuracy()

        transform = ToTensor()
        self.train_set, self.val_set = random_split(
            MNIST(root='./../datasets', train=True, download=True, transform=transform), [55_000, 5_000]
        )
        self.test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    def forward(self, images, text):
        images_embedded = self.patch_embedding(images)

        text_embedded = self.embedding(text).unsqueeze(1)

        tokens = cat((text_embedded, images_embedded), dim=1)

        tokens = stack([
            vstack((self.class_token, tokens[i])) for i in range(len(tokens))
        ])

        encoded = self.encoder(tokens)

        final_class_tokens = encoded[:, 0]

        output = self.MLP(final_class_tokens)

        if self.output_dim == 1:
            output = output.flatten()

        return output

    @staticmethod
    def create_random_true_false_batch(batch: Tensor) -> Tuple[Tensor, Tensor]:
        targets = ones(batch.shape[0])
        batch_new = zeros(batch.shape[0]).long()

        for index, label in enumerate(batch):
            if random.randint(0, 1) == 0:
                batch_new[index] = (label + random.randint(1, 9)) % 10
                targets[index] = 0
            else:
                batch_new[index] = batch[index]

        return batch_new, targets

    def training_step(self, batch: Tensor, _: int) -> Tensor:
        x, y = batch
        y_new, targets = self.create_random_true_false_batch(y)

        y_hat = self(x, y_new)
        loss = self.loss_fn(y_hat, targets)

        return loss

    def validation_step(self, batch: Tensor, _: int) -> Tuple[Tensor, Tensor]:
        x, y = batch
        y_new, targets = self.create_random_true_false_batch(y)

        y_hat = self(x, y_new)
        loss = self.loss_fn(y_hat, targets)
        acc = self.accuracy(y_hat, targets.long())

        return loss, acc

    def test_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor]:
        return self.validation_step(batch, batch_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, shuffle=True, batch_size=32)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, shuffle=False, batch_size=32)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, shuffle=False, batch_size=32)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, shuffle=False, batch_size=32)
