from transformers import ViTModel
from torch import Tensor
from torch.nn import Module


class VisionEncoder(Module):
    def __init__(self, convolutional: bool = False) -> None:
        super().__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") \
            if not convolutional \
            else ViTModel.from_pretrained("facebook/detr-resnet-50")
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).last_hidden_state
