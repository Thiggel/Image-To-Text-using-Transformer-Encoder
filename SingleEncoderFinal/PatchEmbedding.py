from torch import Tensor
from torch.nn import Module, \
    Linear
from typing import Tuple


class PatchEmbedding(Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            patch_size: Tuple[int, int] = (4, 4),
            embed_dim: int = 8
    ):
        super(PatchEmbedding, self).__init__()

        self.input_shape = input_shape
        self.patch_size = patch_size

        self.linear_projection = Linear(self.input_dim, embed_dim)

    @property
    def input_dim(self) -> int:
        return self.input_shape[0] * self.patch_size[0] * self.patch_size[1]

    @property
    def n_patches(self) -> int:
        _, width, height = self.input_shape
        patch_width, patch_height = self.patch_size

        return (width // patch_width) * (height // patch_height)

    def forward(self, images: Tensor) -> Tensor:
        patches = images.reshape(images.shape[0], self.n_patches, self.input_dim)

        return self.linear_projection(patches)
