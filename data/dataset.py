import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def readIndex(index_path, shuffle=False):
    img_list = []
    with open(index_path, 'r') as f:
        for line in f:
            if not line:
                break
            item = line.strip().split()
            item[0] = item[0].replace('\\', '/')
            item[1] = item[1].replace('\\', '/')
            img_list.append(item)
    if shuffle:
        random.shuffle(img_list)
    return img_list

class dataReadPip(object):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, item):

        img = cv2.imread(item[0], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {item[0]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        lab = cv2.imread(item[1], cv2.IMREAD_GRAYSCALE)
        if lab is None:
            raise FileNotFoundError(f"Mask not found: {item[1]}")

 
        if self.transforms is not None:
            img, lab = self.transforms(img, lab)

        img = _preprocess_img(img)   
        lab = _preprocess_lab(lab)   
        return img, lab

def _preprocess_img(np_image):
    """
    np_image: HWC, float32 already normalized by A.Normalize(mean,std)
    return:   CHW torch.float32
    """
    if np_image.dtype != np.float32:
      
        np_image = np_image.astype(np.float32)
    np_image = np.transpose(np_image, (2, 0, 1))
    return torch.from_numpy(np_image)

def _preprocess_lab(cvImage):
    """
    mask as 0 (bg) or 255 (crack) -> 0 or 1
    """
    cvImage = (cvImage.astype(np.float32) / 255.0)
    return torch.from_numpy(cvImage)

class loadedDataset(Dataset):
    def __init__(self, dataset, preprocess=None):
        super().__init__()
        self.dataset = dataset
        self.preprocess = preprocess if preprocess is not None else (lambda x: x)

    def __getitem__(self, index):
        return self.preprocess(self.dataset[index])

    def __len__(self):
        return len(self.dataset)