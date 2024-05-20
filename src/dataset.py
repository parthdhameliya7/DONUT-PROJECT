import torch 
import cv2
import pandas as pd

from torch.utils.data import Dataset 
from typing import Optional, Any


class DonutDataset(Dataset):
    def __init__(self,
                 df : pd.DataFrame = None,
                 data_dir : Optional[str] = None,
                 dataset_dir : Optional[str] = None,
                 augmentations : Optional[Any] = None,
                 ) -> None:

        self.df = df
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations 

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index : int) -> dict:

        row = self.df.iloc[index]
        filepath = row.filepaths #retrived filepath
        filepath = f'{self.data_dir}{self.dataset_dir}{filepath}' #processed filepath
        images = cv2.imread(filepath)
        ground_truth = row.ground_truth

        return {
            'images' : images,
            'targets' : ground_truth 
        }

