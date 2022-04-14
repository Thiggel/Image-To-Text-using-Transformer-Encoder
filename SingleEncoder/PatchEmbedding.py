from torch import Tensor
from torch.nn import Module, Linear
from typing import Tuple


class PatchEmbedding(Module):
    def __init__(
            self,
            input_shape: Tuple = (1, 224, 224),
            patch_size: int = 16,
            embed_dim: int = 512
    ) -> None:
        super().__init__()

        self.patch_size = patch_size

        self.input_shape = input_shape

        self.linear_projection = Linear(
            patch_size ** 2,
            embed_dim
        )

    @property
    def num_patches(self) -> int:
        channels, width, height = self.input_shape

        return channels * (width // self.patch_size) * (height // self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        patches = x.unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .flatten(1, 3) \
            .flatten(2, 3)

        return self.linear_projection(patches)
