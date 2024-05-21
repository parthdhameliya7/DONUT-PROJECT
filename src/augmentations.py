import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from src.config import * 
from typing import Optional, Tuple

class Transforms:

    def __init__(self, image_size : Optional[Tuple] = (256, 256), aug_p : float = 0.2):
        self.image_size = image_size
        self.aug_p = aug_p

    def get_transforms(self):
        train_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            #A.Normalize(mean = cfg['MEAN'], std = cfg['STD']),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, brightness_by_max=False, always_apply=False, p=0.35),
            A.CLAHE(p = 0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.4),
            A.HueSaturationValue(p=0.5),
            #A.Superpixels(n_segments=10000, max_size= 615, p = 0.4),
            A.Defocus(radius=(1, 2), alias_blur=(0, 0.05), p = 0.3),
            A.ImageCompression(quality_lower=20, quality_upper=21, p = 0.3),
            A.Cutout(num_holes = 55, max_h_size = 10, max_w_size = 10, fill_value = 0, p = 0.65),
            ToTensorV2(p = 1.0)
        ], additional_targets = {'image' : 'image'})


        valid_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            #A.Normalize(mean = cfg['MEAN'], std = cfg['STD']),
            ToTensorV2(p = 1.0)
        ], additional_targets = {'image' : 'image'})

        return train_transform, valid_transform