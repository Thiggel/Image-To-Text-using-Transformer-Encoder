from torch import Tensor
from torch.nn import Module, \
    Linear, \
    Sequential, \
    Conv3d
from typing import Tuple


class PatchEmbedding(Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            patch_size: Tuple[int, int] = (4, 4),
            embed_dim: int = 8,
            conv_layers: int = 0,
    ):
        super(PatchEmbedding, self).__init__()

        self.input_shape = input_shape
        self.patch_size = patch_size

        channels = input_shape[0]
        kernel_size = 3
        hidden_channels = 6

        self.conv = Sequential(*[
            Conv3d(
                channels if index == 0 else hidden_channels,
                hidden_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, 1, 1)
            )
            for index in range(conv_layers)
        ])

        # (h, w) -> (h - kernel_size + 1, w - kernel_size + 1)

        self.conv_layers = conv_layers

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
        patches = images.unfold(2, self.patch_size[0], self.patch_size[1]) \
            .unfold(3, self.patch_size[0], self.patch_size[1]) \
            .flatten(2, 3)

        #print(patches.shape)

        #if self.conv_layers != 0:
        #    patches = self.conv(patches)

        patches = patches.transpose(1, 2).flatten(2, 4)

        #print(patches.shape)
        #exit()

        return self.linear_projection(patches)
