from CocoDataModule import CocoDataModule
from model import Model
from torch.optim import Adam
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

if __name__ == '__main__':

    data_module = CocoDataModule(
        train_images_dir='../train2017',
        train_annotations_file='../annotations/captions_train2017.json',
        val_images_dir='../val2017',
        val_annotations_file='../annotations/captions_val2017.json',
        batch_size=8
    )

    batch1 = next(iter(data_module.train_dataloader()))
    batch2 = next(iter(data_module.train_dataloader()))

    model = Model(
        vocab_size=data_module.vocab_size,
        pad_token=data_module.pad_token
    )

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=200)

    print(model)

    def training_step(batch):
        [images, captions], targets = batch

        predicted = model(images, captions, model.create_pad_mask(captions))

        print("Predicted: ", predicted)

        print("Targets: ", targets)

        loss = model.loss_fn(predicted, targets.float())

        print("Loss: ", loss)

        print("Learning Rate: ", scheduler.get_last_lr()[0])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    step = 1
    while True:
        print("Step: ", step)

        training_step(batch1)
        training_step(batch2)

        scheduler.step()

        step += 1