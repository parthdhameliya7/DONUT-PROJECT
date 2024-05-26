from src import (
    params,
    DonutDataset,
    DonutModel,
    Transforms
)

import numpy as np 
import pandas as pd
from transformers import DonutProcessor
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

train_df = pd.read_csv(params['data_csv'] + params['train_csv'])
valid_df = pd.read_csv(params['data_csv'] + params['valid_csv'])

train_augs, valid_augs = Transforms(image_size=params['image_size']).get_transforms()
processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base')
processor.tokenizer.eos_token = params['end_token']
processor.tokenizer.add_tokens(params['special_tokens'] + [params['start_token']] + [params['end_token']])
params['vocab_size'] = len(processor.tokenizer)

train_dataset = DonutDataset(
    df = train_df,
    data_dir=params['data_dir'],
    dataset_dir=params['train_data_dir'],
    augmentations=train_augs,
    processor=processor
)

valid_dataset = DonutDataset(
    df=valid_df,
    data_dir=params['data_dir'],
    dataset_dir=params['valid_data_dir'],
    augmentations=valid_augs,
    processor=processor
)

data = train_dataset[0]

config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = list(params['image_size'])
config.decoder.max_length = params['max_length']

donut = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)
donut.decoder.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=16)
donut.config.pad_token_id = processor.tokenizer.pad_token_id
donut.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([params['start_token']])[0]

model = DonutModel(donut)
model.fit(
    train_dataset = train_dataset,
    valid_dataset = valid_dataset,
    device = 'cuda',
    train_bs = params['train_bs'],
    valid_bs = params['valid_bs'],
    epochs = params['epochs'],
)