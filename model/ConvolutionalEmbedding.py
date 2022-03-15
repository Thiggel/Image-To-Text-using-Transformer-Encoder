from torch.nn import Module, Conv2d, Sequential, GroupNorm, ReLU, MaxPool2d
from torch import Tensor
from functools import reduce


class ConvNetLayer(Module):

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: int,
            stride: int = 1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        self.model = Sequential(
            Conv2d(input_channels, output_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride)),
            # a group size of 32 was found to lead to the lowest error
            GroupNorm(num_groups=32, num_channels=output_channels),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )

    def output_size(self, input_size):
        return ((input_size - self.kernel_size) / self.stride + 1) // 2

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ConvolutionalEmbedding(Module):

    def __init__(
            self,
            embed_dim: int,
            image_size: int = 128
    ) -> None:
        super().__init__()

        self.image_size = image_size

        self.model = Sequential(
            # suppose image size is 128x128
            # size: (128 - 6 + 1) // 2 = 61
            ConvNetLayer(input_channels=3, output_channels=32, kernel_size=6),
            # size: (61 - 5 + 1) // 2 = 28
            ConvNetLayer(input_channels=32, output_channels=64, kernel_size=5),
            # size: (28 - 4 + 1) // 2 = 12
            ConvNetLayer(input_channels=64, output_channels=128, kernel_size=4),
            # size: (12 - 3 + 1) // 2 = 5
            ConvNetLayer(input_channels=128, output_channels=embed_dim, kernel_size=3)
        )

    @staticmethod
    def calculate_intermediate_output_size(intermediate_output_size: int, layer: Module) -> int:
        return layer.output_size(intermediate_output_size)

    @property
    def sequence_length(self) -> int:
        return int(reduce(
            self.calculate_intermediate_output_size,
            list(iter(self.model)),
            self.image_size
        )) ** 2

    def forward(self, x: Tensor) -> Tensor:
        # shape: (num_images, channels, image_size, image_size) ->
        # (num_images, sequence_length, embed_dim)
        return self.model(x).flatten(2).transpose(1, 2)
