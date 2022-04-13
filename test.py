import torch
from torch.optim import Adam
from torchmetrics import Accuracy

from datasets.coco.CocoDataModule import CocoDataModule
from model.UnifiedTransformer import UnifiedTransformer

if __name__ == '__main__':

    # using the command line arguments, two hyper parameters can be set
    # 1. the dataset and 2. whether images are processed using convolution
    # or pure attention
    print("Loading and preprocessing the data set...", end=" ")
    data_module = CocoDataModule()
    train_dataloader = data_module.train_dataloader()
    batch = next(iter(train_dataloader))
    print("Done!")
    print("Batch", batch)

    print("Loading model...", end=" ")
    model = UnifiedTransformer(
        num_classes=data_module.num_classes
    )
    print("Done!")

    optimizer = Adam(model.parameters(), lr=0.0001)

    loss_fn = torch.nn.BCELoss()
    acc = Accuracy()

    while True:
        [images, captions], targets = batch
        targets = targets.float()

        predicted = model(images, captions)

        optimizer.zero_grad()

        print("Predicted: ", predicted)
        print("Targets: ", targets)

        loss = loss_fn(predicted, targets)
        accuracy = acc(predicted, targets.long())

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)
        print("\n\n\n\n")

        loss.backward()

        optimizer.step()