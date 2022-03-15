from torch import zeros, cat, Tensor, save, load
from pytorch_lightning import LightningModule
from os.path import isfile
from torch.nn import \
    Parameter, \
    TransformerEncoder, \
    TransformerEncoderLayer, \
    Linear, \
    Embedding, \
    Dropout, \
    Softmax
from torch.nn.functional import cross_entropy
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from torchmetrics import Accuracy
from typing import Tuple, List

from model.PatchEmbedding import PatchEmbedding
from model.ConvolutionalEmbedding import ConvolutionalEmbedding


class UnifiedTransformer(LightningModule):
    def __init__(
            self,
            image_size: int,
            num_tokens: int,
            sequence_length: int,
            num_encoder_layers: int,
            num_classes: int,
            convolutional_embedding: bool = False,
            patch_size: int = 16,
            num_heads: int = 12,
            embed_dim: int = 768,
            dropout: float = 0.1,
            learning_rate: float = 0.05,
            filename: str = 'model.pt'
    ) -> None:

        super().__init__()

        self.filename = filename

        # the embedding dimension is used throughout the entire model
        # it is a hyperparameter of the transformer encoder layer
        # and hence the image patches have to be projected to this dimension.
        self.embed_dim = embed_dim

        # The patch embedding module takes a sequence of images and for each image,
        # it splits it into 16x16 patches, flattens them and projects them to
        # the embedding dimension `embed_dim`
        # hence, the resulting vector will have the shape (num_images, num_patches, embed_dim)
        self.patch_embed = PatchEmbedding(image_size, patch_size=patch_size, embed_dim=embed_dim) \
            if not convolutional_embedding \
            else ConvolutionalEmbedding(embed_dim)

        # The tokens in the text sequences are embedded using a word embedding.
        # This refers to a hyperspace that maps words to positions
        # where words more similar in meaning are closer together.
        # Thus, a lot more information is encoded than with one-hot-encodings.
        self.word_embed = Embedding(num_tokens, embed_dim)

        # We prepend a class token `[class]` to the image and text sequence
        # this class token is a learnable parameter and is at the end fit into
        # the MLP head for the eventual classification.
        # It thus has the same dimensions as a single image patch (embed_dim)
        self.class_token = Parameter(zeros(1, 1, embed_dim))

        # We concatenate each image patch and text token with its positional embedding
        # so that this information is not lost in the transformer
        # (as the order of tokens fed into a transformer normally does
        # not make a difference) it also is a parameter the model learns
        # Its second dimension is larger than the number of patches and tokens by exactly 1,
        # because we prepend the class token to the patch embedding before
        # adding the positional embedding.
        # Therefore, the shape is (1, num_patches + num_tokens + 1, embed_dim) as it is added to
        # one patch embedding (hence the 1 in front), and each token in the embedding
        # has embed_dim dimensions.
        self.positional_encoding = Parameter(
            zeros(
                1,
                self.patch_embed.sequence_length + sequence_length + 1,
                embed_dim
            )
        )

        # we use a transformer encoder as the main part of the network.
        # There are num_encoder_layers in this encoder.
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout
            ),
            num_encoder_layers
        )

        self.MLP_head = Linear(embed_dim, num_classes)

        self.dropout = Dropout(dropout)

        self.softmax = Softmax()

        self.learning_rate = learning_rate

        self.accuracy = Accuracy()

    def forward(self, images: Tensor, text: Tensor) -> Tensor:
        # create patch embeddings
        image_patches = self.patch_embed(images)

        # embed the text sequence
        text_embedded = self.word_embed(text)

        # concatenate image and text embeddings
        x = cat((image_patches, text_embedded), dim=1)

        # replicate class token as many times as there
        # are tokens in the tensor
        n_class_tokens = self.class_token.expand(x.shape[0], -1, -1)

        # prepend class tokens to embeddings
        x = cat((n_class_tokens, x), dim=1)

        # add positional encoding
        x += self.positional_encoding

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
            'monitor': LearningRateMonitor('epoch')
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
