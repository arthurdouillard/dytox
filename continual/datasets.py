# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import json
import os
import warnings

from continuum import ClassIncremental
from continuum.datasets import CIFAR100, ImageNet100, ImageFolderDataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import functional as Fv

try:
    interpolation = Fv.InterpolationMode.BICUBIC
except:
    interpolation = 3


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set.lower() == 'cifar':
        dataset = CIFAR100(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == 'imagenet100':
        dataset = ImageNet100(
            args.data_path, train=is_train,
            data_subset=os.path.join('./imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
    elif args.data_set.lower() == 'imagenet1000':
        dataset = ImageNet1000(args.data_path, train=is_train)
    else:
        raise ValueError(f'Unknown dataset {args.data_set}.')

    scenario = ClassIncremental(
        dataset,
        initial_increment=args.initial_increment,
        increment=args.increment,
        transformations=transform.transforms,
        class_order=args.class_order
    )
    nb_classes = scenario.nb_classes

    return scenario, nb_classes


def build_transform(is_train, args):
    if args.aa == 'none':
        args.aa = None

    with warnings.catch_warnings():
        resize_im = args.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)

            if args.input_size == 32 and args.data_set == 'CIFAR':
                transform.transforms[-1] = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=interpolation),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        if args.input_size == 32 and args.data_set == 'CIFAR':
            # Normalization values for CIFAR100
            t.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        else:
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        return transforms.Compose(t)
