from torch.nn import Module


class FrozenModule(Module):
    def __init__(self, model: Module) -> None:
        super().__init__()

        self.model = model

        # go through all parameters and set them to
        # non-trainable
        for _, param in self.model.named_parameters():
            param.requires_grad = False
