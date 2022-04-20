from torch import rand, stack, vstack, cat
from torch import Tensor, device, cuda
from pytorch_lightning import LightningModule
from torch.nn import Sequential, \
    Linear, \
    Softmax, \
    Sigmoid, \
    Embedding, \
    Parameter, \
    BCELoss, \
    CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Tuple
from torchmetrics import Accuracy

from CocoDataModule import CocoDataModule

from PatchEmbedding import PatchEmbedding
from Encoder import Encoder


class UnifiedTransformerCoco(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            patch_size: Tuple[int, int] = (4, 4),
            conv_layers: int = 0,
            embed_dim: int = 8,
            n_heads: int = 2,
            output_dim: int = 1,
            learning_rate: float = 1e-3
    ):
        super(UnifiedTransformerCoco, self).__init__()

        self.dev = device("cuda:0" if cuda.is_available() else "cpu")

        self.data_module = CocoDataModule(
            train_images_dir='../train2017',
            train_annotations_file='../annotations/captions_train2017.json',
            val_images_dir='../val2017',
            val_annotations_file='../annotations/captions_val2017.json',
            batch_size=32
        )

        self.vocab_size = self.data_module.vocab_size
        self.text_length = self.data_module.sequence_length

        self.output_dim = output_dim

        self.patch_embedding = PatchEmbedding(input_shape, patch_size, embed_dim, conv_layers)

        self.sequence_length = self.patch_embedding.n_patches + self.text_length + 1

        self.embedding = Embedding(self.vocab_size, embed_dim)

        self.class_token = Parameter(rand(1, embed_dim))

        self.encoder = Encoder(self.sequence_length, input_shape, n_heads, embed_dim).to(self.dev)

        self.MLP = Sequential(
            Linear(embed_dim, output_dim),
            Sigmoid() if output_dim == 1 else Softmax()
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=60, verbose=True)
        self.loss_fn = BCELoss() if output_dim == 1 else CrossEntropyLoss()

        self.accuracy = Accuracy()

    def forward(self, images, text):
        images_embedded = self.patch_embedding(images)

        text_embedded = self.embedding(text)

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

    def training_step(self, batch: Tensor, _: int) -> Tensor:
        [x, y], targets = batch
        x, y, targets = x.to(self.dev), y.to(self.dev), targets.to(self.dev)

        y_hat = self(x, y)
        loss = self.loss_fn(y_hat, targets.float())

        return loss

    def validation_step(self, batch: Tensor, _: int) -> Tuple[Tensor, Tensor]:
        [x, y], targets = batch 
        x, y, targets = x.to(self.dev), y.to(self.dev), targets.to(self.dev)

        y_hat = self(x, y)
        loss = self.loss_fn(y_hat, targets.float())
        acc = self.accuracy(y_hat, targets.long())

        return loss, acc

    def test_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor]:
        return self.validation_step(batch, batch_idx)

    def train_dataloader(self) -> DataLoader:
        return self.data_module.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.data_module.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.data_module.test_dataloader()
