from torch import Tensor
from torch.nn import Module, Conv2d


class PatchEmbedding(Module):

    def __init__(self, image_size: int, channels: int = 3, patch_size: int = 16, embed_dim: int = 768) -> None:
        """
        Split a tensor of at least two dimensions
        into patches of a specified size
        within the first two dimensions

        (To input a PIL image, use torchvision.transforms.PILToTensor())
        ----------------------------------------------------------------
        :param image_size: image size (square)
        :param patch_size: patch size (square)
        """
        super().__init__()

        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size

        self.linear_projection = Conv2d(channels, embed_dim, (patch_size, patch_size), (patch_size, patch_size))

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def forward(self, x: Tensor) -> Tensor:
        """
        Call the class to create the patches
        :param x: the input tensor
        :return: an array of patches
        """

        # shape: (num_images, channels, image_size, image_size) -> (num_images, num_patches, embed_dim)
        return self.linear_projection(x) \
            .flatten(2) \
            .transpose(1, 2)
