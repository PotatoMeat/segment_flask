import os
import cv2

from torch.utils.data import Subset, Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")


class SegmentDataset(Dataset):
    def __init__(self, dirPath, imageDir, masksDir):
        self.imgDirPath = os.path.join(dirPath, imageDir)
        self.maskDirPath = os.path.join(dirPath, masksDir)
        self.nameImgFile = sorted(os.listdir(self.imgDirPath))
        self.nameMaskFile = sorted(os.listdir(self.maskDirPath))

    def __len__(self):
        return len(self.nameImgFile)

    def __getitem__(self, index):
        imgPath = os.path.join(self.imgDirPath, self.nameImgFile[index])
        maskPath = os.path.join(self.maskDirPath, self.nameMaskFile[index])
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)

        train_transform = A.Compose(
            [
                A.Resize(1024, 1024),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dt = train_transform(image=img, mask=mask)
        return dt['image'], dt['mask']
