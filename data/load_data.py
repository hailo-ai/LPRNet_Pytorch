# from torch.utils.data import *
from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
from PIL import Image
import os
import imgaug.augmenters as iaa

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}


def image_aug(img):
    seq = iaa.Sequential([
        iaa.SomeOf((0, 5),[
            iaa.Crop(percent=(0, 0.05)),
            iaa.OneOf([  # add blur
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11)),
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen each image
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            iaa.LinearContrast((0.75, 1.5)),  # improve/destroy image contrast
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.2),  # add noise
            iaa.Multiply((0.5, 0.7), per_channel=0.2),  # change brightness
            iaa.Grayscale(alpha=(0.0, 1.0)),  # remove/strenghts colors
            iaa.OneOf([
                iaa.Dropout((0.005, 0.01), per_channel=0.015),
                iaa.CoarseDropout(
                    (0.005, 0.01), size_percent=(0.005, 0.01),
                    per_channel=0.015
                ),
            ]),
            iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.9),  # move pixels
            iaa.Dropout((0.01, 0.05), per_channel=0.9),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # distort local areas
            iaa.Invert(0.05, per_channel=True),
            iaa.Affine(
                scale={"x": (0.99, 1.01), "y": (0.99, 1.01)},
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-10, 10),
                shear=(-10, 10),
                order=[0, 1],
                cval=(0, 255)
            )
        ], random_order=True)
    ], random_order=True)
    return np.squeeze(seq(images=np.expand_dims(np.array(img, np.uint8), axis=0)))


def pixelize(img, width, height, factors=(4, 5), p=0.5):
    factor = random.choice(factors)
    w = int(width/factor)
    h = int(height/factor)
    interp = Image.NEAREST if random.random() >= p else Image.BILINEAR
    img = np.expand_dims(np.array(Image.fromarray(np.squeeze(img)).resize((w, h))), axis=0)
    img = np.array(Image.fromarray(np.squeeze(img)).resize((width, height), resample=interp))
    return img


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None, train=True):
        self.train = train
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)  # BGR
        if len(Image.shape) == 2:
            Image = np.repeat(np.expand_dims(Image, axis=-1), 3, axis=-1)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            label.append(CHARS_DICT[c])

        return Image, label, len(label)

    def transform(self, img):
        if self.train:
            img = image_aug(img)
            if random.random() >= 0.5:
                img = pixelize(img, width=self.img_size[0], height=self.img_size[1])
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125 # * (1/128)
        img = np.transpose(img, (2, 0, 1)) # H, W, C --> C, H, W

        return img
