from torch import Tensor, cat, zeros, save, load
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer, Linear, Dropout, Softmax, CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from typing import List, Tuple
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from os.path import isfile

from model.ImageBackbone import ImageBackbone
from model.TextBackbone import TextBackbone


class UnifiedTransformer(LightningModule):
    def __init__(
            self,
            num_classes: int,
            num_encoder_layers: int = 6,
            nhead: int = 12,
            dropout: float = 0.1,
            learning_rate: float = 0.001,
            filename: str = 'model.pt',
            convolutional_embedding: bool = False
    ) -> None:
        """
        Process both text and images simultaneously for Visual Question Answering (VQA)
        :param num_classes: Number of classes for the eventual classifier (e.g. vocab size)
        :param num_encoder_layers: Depth of multi-modal transformer encoder block
        :param nhead: Number of heads
        :param dropout: Dropout rate at end of model and in multi-modal transformer encoder block
        :param learning_rate: used in training (using ADAM optimizer)
        :param filename: where model will be saved after each epoch
        :param convolutional_embedding: Whether the ImageBackbone should be based on convolution or attention
        """
        super().__init__()

        self.filename = filename

        # to embed the image, we use a vision transformer, pretrained on
        # object detection. The forward function of this transformer
        # has been altered so that the processing is concluded before the
        # class token is fed into the MLP head. Henceforth, we obtain
        # a processed sequence of embedded patches that we treat as
        # our 'image embedding'
        self.image_backbone = ImageBackbone(convolutional_embedding)

        # In the same way, we use a modified pretrained BERT model,
        # pretrained on various tasks including question answering
        # and natural language inference
        self.text_backbone = TextBackbone()

        # assert that the output sizes of the two embeddings are the same
        # so that they can be concatenated
        assert  \
            self.image_backbone.model.config.hidden_size == self.text_backbone.model.config.hidden_size, \
            "The embedding dimensions for the pretrained image and text encoder must be the same"

        # save our embedding dimension (taken from the embedding layers)
        self.d_model = self.image_backbone.model.config.hidden_size

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

        self.softmax = Softmax(dim=1)

        self.learning_rate = learning_rate

        # Training/Testing:

        self.accuracy = Accuracy()

        self.loss_function = CrossEntropyLoss()

    def forward(self, images: Tensor, text: List) -> Tensor:
        """
        Perform attention within and between images and text and output
        class probabilities
        :param images: Batch of image tensors
        :param text: Batch of text sequences
        :return: A 1d-tensor of class probabilities
        """

        # we first embed the image and text using the pretrained
        # ViT and BERT models. The parameters of those will not be trained.
        # so that training goes faster, and we simply use the processed
        # information from the two models
        images_embedded = self.image_backbone(images)
        text_embedded = self.text_backbone(text)

        # concatenate the three tensors
        x = cat((text_embedded, images_embedded), dim=1)

        x = self.transformer_encoder(x)

        # get the class tokens from the sequence (BERT appends a class token)
        final_class_token = x[:, 0]

        # lastly, feed it into the MLP
        x = self.MLP_head(final_class_token)

        # add dropout to prevent overfitting
        x = self.dropout(x)

        # compute probabilities between 0 and 1
        # using the softmax function
        return self.softmax(x)

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

        print(predicted.shape, targets.shape)
        loss = self.loss_function(predicted, targets)
        print(loss)

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
        loss = self.loss_function(predicted, targets)
        accuracy = self.accuracy(predicted, targets.long())

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
        loss = self.loss_function(predicted, targets)
        accuracy = self.accuracy(predicted, targets.long())

        self.log('test_loss', loss)
        self.log('test_acc', accuracy)

        return loss

    def training_epoch_end(self, _) -> None:
        """
        Hook that fires after each epoch. Saves the model
        to a file.
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
