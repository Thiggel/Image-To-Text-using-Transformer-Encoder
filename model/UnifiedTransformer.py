from torch import Tensor, cat, zeros, save, load
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer, Linear, Dropout, Softmax
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from typing import List, Tuple
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from os.path import isfile

from model.VisionEncoder import VisionEncoder
from model.TextEncoder import TextEncoder


class UnifiedTransformer(LightningModule):
    def __init__(
            self,
            num_classes: int,
            num_encoder_layers: int = 6,
            nhead: int = 12,
            dropout: float = 0.1,
            learning_rate: float = 0.05,
            filename: str = 'model.pt',
            convolutional_embedding: bool = False
    ) -> None:
        super().__init__()

        self.filename = filename

        # to embed the image, we use a vision transformer, pretrained on
        # object detection. The forward function of this transformer
        # has been altered so that the processing is concluded before the
        # class token is fed into the MLP head. Henceforth, we obtain
        # a processed sequence of embedded patches that we treat as
        # our 'image embedding'
        self.image_embedding = VisionEncoder(convolutional_embedding)

        # In the same way, we use a modified pretrained BERT model,
        # pretrained on various tasks including question answering
        # and natural language inference
        self.text_embedding = TextEncoder()

        # assert that the output sizes of the two embeddings are the same
        # so that they can be concatenated
        assert  \
            self.image_embedding.model.config.hidden_size == self.text_embedding.model.config.hidden_size, \
            "The embedding dimensions for the pretrained image and text encoder must be the same"

        # save our embedding dimension (taken from the embedding layers)
        self.d_model = self.image_embedding.model.config.hidden_size

        # we append a class token when processing input, which is later used
        # to classify the output
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

    def forward(self, images: Tensor, text: Tensor) -> Tensor:
        # we first embed the image and text using the pretrained
        # ViT and BERT models. The parameters of those will not be trained.
        # so that training goes faster and we simply use the processed
        # information from the two models
        images_embedded = self.image_embedding(images)
        text_embedded = self.text_embedding(text)

        # replicate class token as many times as there
        # are tokens in the tensor
        n_class_tokens = self.class_token.expand(images_embedded.shape[0], -1, -1)

        # concatenate the three tensors
        x = cat((n_class_tokens, images_embedded, text_embedded), dim=1)

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
        print('Validation Loss: ', loss)
        print('Validation Accuracy: ', accuracy)
        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # get columns of batch
        [images, captions], targets = batch

        predicted = self.forward(images, captions)
        loss = cross_entropy(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        print('Test Loss: ', loss)
        print('Test Accuracy: ', accuracy)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy)
        loss = compute_epoch_loss_from_outputs(outputs)
        print('Training loss for Epoch: ', loss)
        # save the model after every epoch
        self.save()

    def save(self) -> None:
        print("Saving model at: " + self.filename)
        save(self.state_dict(), self.filename)

    def load(self) -> None:
        if isfile(self.filename):
            print("Loading model from: " + self.filename)
            self.load_state_dict(load(self.filename))
