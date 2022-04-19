from numpy import sin, cos
from torch import ones, device, cuda
from torch import Tensor, device as Device
from torch.nn import Module, \
    Sequential, \
    Linear, \
    LayerNorm, \
    ReLU
from typing import Tuple

from MultiHeadAttention import MultiHeadAttention


class Encoder(Module):
    def __init__(
            self,
            sequence_length: int,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            n_heads: int = 2,
            embed_dim: int = 8
    ) -> None:
        super(Encoder, self).__init__()

        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        self.input_shape = input_shape

        self.norm1 = LayerNorm([sequence_length, embed_dim])

        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm2 = LayerNorm([sequence_length, embed_dim])

        self.MLP = Sequential(
            Linear(embed_dim, embed_dim),
            ReLU()
        )

        self.dev = device("cuda:0" if cuda.is_available() else "cpu")

    def get_positional_embeddings(self) -> Tensor:
        result = ones(self.sequence_length, self.embed_dim)

        for i in range(self.sequence_length):
            for j in range(self.embed_dim):
                result[i][j] = sin(i / (10000 ** (j / self.embed_dim))) \
                    if j % 2 == 0 \
                    else cos(i / (10000 ** ((j - 1) / self.embed_dim)))

        return result

    def forward(self, tokens: Tensor) -> Tensor:
        tokens += self.get_positional_embeddings().repeat(tokens.shape[0], 1, 1).to(self.dev)

        out = tokens + self.attention(self.norm1(tokens))

        out = out + self.MLP(self.norm2(out))

        return out
