import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from CocoTrueAndFalseCaptions import CocoTrueAndFalseCaptions
from UnifiedTransformer import UnifiedTransformer


dataset = CocoTrueAndFalseCaptions(
    image_dir='val2017',
    annotations_file='annotations/captions_val2017.json',
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.PILToTensor()
    ])
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


[images, captions], targets = next(iter(dataloader))

model = UnifiedTransformer(
    image_size=128,
    num_tokens=dataset.vocab_size,
    sequence_length=dataset.sequence_length,
    num_encoder_layers=6,
    num_classes=2
)

print(model.forward(images.float(), captions))


# print(
#     model(
#         torch.stack(images).float(),
#         list(captions)
#     )
# )

# print(len(dataset))
#
# (img1, caption1), target1 = dataset.__getitem__(10)
# img1 = np.transpose(img1, (1,2,0))
# (img2, caption2), target2 = dataset.__getitem__(len(dataset) + 10)
# img2 = np.transpose(img2, (1,2,0))
#
# #plt.imshow(img1)
# print(caption1)
# plt.imshow(img2)
# print(caption2)
#
# plt.show()

# dataset = dset.CocoDetection(
#     root ='val2017',
#     annFile = 'annotations/captions_val2017.json',
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
# )
#
# loader = DataLoader(dataset, batch_size=64, shuffle=True)
#
# iterator = iter(loader)
#
# inputs, targets = iterator.next()
#
# print(inputs[0])
# print()
# print("-"*100)
# print()
# print(targets[0])

#all_captions = list(itertools.chain.from_iterable(captions))

#print(all_captions)

#tokenizer = get_tokenizer('basic_english')
#vocab = build_vocab_from_iterator(map(tokenizer, cap), specials=['<unk>'])
#vocab.set_default_index(vocab['<unk>'])

#print('Number of samples: ', len(cap))
#img, target = cap[3] # load 4th sample

#print("Image Size: ", img.size())
#print(target)

#img = np.transpose(img, (1,2,0))
#plt.imshow(img)
#plt.show()

#model = UnifiedTransformer(image_size=128, num_tokens=100, num_encoder_layers=6, num_classes=10)

#model(torch.cat([list(img.detach())]), torch.tensor([list(torch.randint(1, 10, (100,)).detach())]))

