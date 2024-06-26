import torch 
import json
import cv2
import pandas as pd
from src.config import * 
from src.utils import json2token
from ast import literal_eval
import pickle

from transformers import DonutProcessor

from torch.utils.data import Dataset 
from typing import Optional, Any, TypedDict

class DataDict(TypedDict):
    images: torch.Tensor
    target_sequences : str
    targets: str

class DonutDataset(Dataset):
    def __init__(self,
                 df : pd.DataFrame = None,
                 data_dir : Optional[str] = '',
                 dataset_dir : Optional[str] = '',
                 augmentations : Optional[Any] = None,
                 processor : Optional[Any] = None
                 ) -> None:

        self.df = df
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations 
        self.processor = processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index : int) -> DataDict:

        row = self.df.iloc[index]
        img_filepath = row.image_filepath 
        json_filepath = row.json_filepaths
        image = cv2.imread(img_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        f = open(json_filepath)
        target = json.load(f)

        target_sequence = json2token(target) + self.processor.tokenizer.eos_token
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens = False,
            max_length = params['max_length'],
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt',
        )['input_ids'].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = params['ignore_id']

        if self.augmentations is not None:
            image = self.augmentations(image = image)['image'] / 255.0

        return {
            'images' : image,
            'target_sequences' : target_sequence,
            'targets' : labels 
        }