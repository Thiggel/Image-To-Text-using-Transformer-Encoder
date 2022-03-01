from PatchEmbedding import PatchEmbedding
from torch import zeros, cat, Tensor
from torch.nn import \
    Module, \
    Parameter, \
    TransformerEncoder, \
    TransformerEncoderLayer, \
    Linear, \
    Embedding


class UnifiedTransformer(Module):
    def __init__(
            self,
            image_size: int,
            num_tokens: int,
            num_encoder_layers: int,
            num_classes: int,
            patch_size: int = 16,
            num_heads: int = 12,
            embed_dim: int = 768,
            dropout: float = 0.1
    ) -> None:

        super().__init__()

        # the embedding dimension is used throughout the entire model
        # it is a hyperparameter of the transformer encoder layer
        # and hence the image patches have to be projected to this dimension.
        self.embed_dim = embed_dim

        # The patch embedding module takes a sequence of images and for each image,
        # it splits it into 16x16 patches, flattens them and projects them to
        # the embedding dimension `embed_dim`
        # hence, the resulting vector will have the shape (num_images, num_patches, embed_dim)
        self.patch_embed = PatchEmbedding(image_size, patch_size=patch_size, embed_dim=embed_dim)

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
        self.positional_embedding = Parameter(
            zeros(
                1,
                self.patch_embed.num_patches + num_tokens + 1,
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

    def forward(self, image: Tensor, text: Tensor) -> Tensor:

        # create patch embeddings
        image_patches = self.patch_embed(image)

        # embed the text sequence
        text_embedded = self.word_embed(text)

        # concatenate image and text embeddings
        x = cat((image_patches, text_embedded), dim=1)

        # replicate class token as many times as there
        # are tokens in the tensor
        n_class_tokens = self.class_token.expand(x.shape[0], -1, -1)

        # prepend class tokens to embeddings
        x = cat((n_class_tokens, x), dim=1)

        # add positional embedding
        x += self.positional_embedding

        x = self.transformer_encoder(x)

        # get the class embeddings out of all embeddings
        final_class_token = x[:, 0]

        # lastly, feed it into the MLP
        return self.MLP_head(final_class_token)
