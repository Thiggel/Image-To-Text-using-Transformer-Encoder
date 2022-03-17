from json import load
from typing import Tuple, List, Optional, Callable
from itertools import chain
from torchtext.data.utils import get_tokenizer
from PIL import Image
from os.path import join, exists
from torch import tensor, Tensor
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomAffine, ToPILImage

from datasets.ImageTextDataset import ImageTextDataset


class VisualGenomeQuestionsAnswers(ImageTextDataset):

    def __init__(
            self,
            images_part1_dir: str,
            images_part2_dir: str,
            questions_answers_file: str,
            patch_size: int = 16,
            image_size: int = 128,
            transform: Optional[Callable] = None
    ) -> None:
        self.image_size = image_size

        self.tokenizer = get_tokenizer('basic_english')
        self.data = self.load_questions(questions_answers_file)
        self.num_images = len(self.data)

        super().__init__(sentence_list=self.word_list())

        self.questions = self.preprocess_text(self.data, text_key=1)

        # images are provided in two downloadable
        # packages, which is why we have to check
        # where an image is when loading it
        self.images_part1_dir = images_part1_dir
        self.images_part2_dir = images_part2_dir

        self.transform = transform

        # if there is a patch embedding in the network that is trained on
        # this set, a patch size can be defined so that the image datapoints
        # are augmented, being shifted by up to patch_size-1 pixels in each
        # direction
        self.patch_size = patch_size

        # for data augmentation, a certain number of transforms
        # are performed on each image. A corresponding
        # array of transforms is initialized hereafter
        self.augmentations = self.init_augmentations()

        # there is as many classes as words in the vocabulary
        # as an answer to a question can be any one word
        self.num_classes = self.vocab_size
	
        from os import mkdir
        from os.path import exists
        if not exists('testimgs'): mkdir('testimgs')
	
        for i in range(len(self.augmentations)):
            index = i * len(self.data)
            ToPILImage()(self.__getitem__(index)[0][0]).save('testimgs/' + str(index) + '.jpg')
            ToPILImage()(self.__getitem__(index + 1)[0][0]).show('testimgs/' + str(index + 1) + '.jpg')
            ToPILImage()(self.__getitem__(index + 2)[0][0]).show('testimgs/' + str(index + 2) + '.jpg')

        for i in range(len(self.augmentations)):
            index = i * len(self.data)
            ToPILImage()(self.__getitem__(index)[0][0]).save('testimgs/' + str(index) + '.jpg')
            ToPILImage()(self.__getitem__(index + 1)[0][0]).show('testimgs/' + str(index + 1) + '.jpg')
            ToPILImage()(self.__getitem__(index + 2)[0][0]).show('testimgs/' + str(index + 2) + '.jpg')

    def preprocess_answer(self, answer: str) -> List:
        return self.tokenizer(answer.replace('.', ''))

    def preprocess_datapoint(self, datapoint) -> Tuple[int, str, str]:
        answer = self.preprocess_answer(datapoint['answer'])

        # we filter out all the answers that contain
        # more than one word, as we don't train our model
        # on generating sentences but just classifying
        # single words
        if len(answer) == 1:
            return datapoint['image_id'], datapoint['question'], answer[0]

    def load_questions(self, question_answers_file: str) -> List[Tuple[int, str, str]]:
        with open(question_answers_file, 'r') as file:
            json = load(file)

            return list(filter(None, [
                self.preprocess_datapoint(datapoint)
                for image in json
                for datapoint in image['qas']
            ]))

    def get_shifting_augmentations(self) -> List[Callable]:
        onePixelAsPercent = 1 / self.image_size

        return [
            RandomAffine(degrees=(0, 0), translate=(x * onePixelAsPercent, y * onePixelAsPercent))
            for y in range(self.patch_size)
            for x in range(self.patch_size)
        ]

    def init_augmentations(self) -> List[Callable]:
        return [
            *self.get_shifting_augmentations(),
            RandomHorizontalFlip(p=1),
            RandomVerticalFlip(p=1),
            ColorJitter(brightness=(.2, .2)),
            ColorJitter(brightness=(.3, .3)),
            ColorJitter(brightness=(.4, .4)),
            ColorJitter(brightness=(.5, .5)),
            ColorJitter(brightness=(.6, .6)),
            ColorJitter(contrast=(.2, .2)),
            ColorJitter(contrast=(.3, .3)),
            ColorJitter(contrast=(.4, .4)),
            ColorJitter(contrast=(.5, .5)),
            ColorJitter(contrast=(.6, .6)),
        ]

    def load_image(self, index: int) -> Image.Image:
        filename = f"{self.data[index][0]}.jpg"

        # as there are two folders with images provided
        # by the Visual Genome team, we want to save
        # the work of copying them into the same folder
        # and just check where our image exists
        dir1 = join(self.images_part1_dir, filename)
        dir2 = join(self.images_part2_dir, filename)

        directory = dir1 if exists(dir1) else dir2

        image = Image.open(directory).convert("RGB")

        return image

    def load_target(self, index: int) -> Tensor:
        return tensor(self.vocab[self.data[index][2]])

    def __getitem__(self, index: int):
        augmentations_index = index // self.num_images
        datapoint_index = index % self.num_images

        # transform target word to numeric tensor using vocab
        answer = self.load_target(datapoint_index)

        question = self.questions[datapoint_index]

        # get image tensor
        image = self.load_image(datapoint_index)

        if self.transform:
            image = self.transform(image)

        # apply augmentations
        image = self.augmentations[augmentations_index](image)

        if isinstance(image, Tensor):
            image = image.float()

        return (image, question), answer

    def word_list(self) -> List[str]:
        return list(chain.from_iterable([datapoint[1:] for datapoint in self.data]))

    @property
    def sequence_length(self) -> int:
        return self.questions.shape[1]

    def __len__(self) -> int:
        return self.num_images * len(self.augmentations)
