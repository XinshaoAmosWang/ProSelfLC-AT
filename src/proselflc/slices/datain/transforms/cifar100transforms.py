import torchvision.transforms as transforms

from proselflc.exceptions import ParamException
from proselflc.trainer.utils import intlabel2onehot

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CLASS_NUM = 100

cifar100_transform_train_data = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),  # standard
        # transforms.Pad(padding=[4, 4, 4, 4]),
        # transforms.RandomResizedCrop(
        #    32, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        # ),
        transforms.RandomHorizontalFlip(),  # flip
        transforms.RandomRotation(15),  # rotation
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ]
)

cifar100_transform_test_data = transforms.Compose(
    [
        # no pad
        # no crop
        # no flip
        # no rotation
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ]
)


def cifar100_transform_intlabel2onehot(intlabel: int):
    if intlabel not in list(range(CLASS_NUM)):
        error_msg = "intlabe: {}".format(
            intlabel
        ) + " not in the range of [0, {}]".format(CLASS_NUM - 1)
        raise ParamException(error_msg)

    return intlabel2onehot(class_num=CLASS_NUM, intlabel=intlabel)
