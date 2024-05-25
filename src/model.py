from torchret import Model 
import torch.nn as nn 
import torch
import re 

from nltk import edit_distance
from typing import Optional, Dict
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from src.config import *


class DonutModel(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        #Vision encoder decoder configurations
        self.processor = None
        self.config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
        self.config.encoder.image_size = list(params['image_size'])
        self.config.decoder.max_length = params['max_length']

        #Initializing model with assigned configurations
        self.donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=self.config)
        self.donut_model.decoder.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=16)

        self.donut_model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.donut_model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(['<s-donut>'])[0]

    def monitor_metrics(self, outputs, target_sequences)-> Dict[None, None]:
        return {}
    
    def valid_metrics(self, outputs, target_sequences)-> Dict[str, float]:
        return {}
    
    def valid_model_fn(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
            batch_size = data[k].shape[0]
        answers = data['target_sequences']
        decoder_input_ids = torch.full((batch_size, 1), self.donut_model.config.decoder_start_token_id, device = self.device)
        output = self.donut_model.generate(
                data['images'],
                decoder_input_ids = decoder_input_ids,
                max_length = params['max_length'],
                early_stopping = True, 
                pad_token_id = self.processor.tokenizer.pad_token_id,
                eos_token_id = self.processor.tokenizer.eos_token_id,
                use_cache = True, 
                num_beams = 1,
                bad_words_ids = [[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate = True
            )
        output_logits, loss = self(**data)
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(output.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            sim = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(1 - sim)
        metrics = {'edit_distance' : sum(scores)}
        
        return output_logits, loss, metrics
    
    def valid_one_step(self, data):
        _, loss, metrics = self.valid_model_fn(data)
        return loss, metrics
    
    def fetch_optimizer(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=params['lr'],
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, 
            T_0 = params['T_0'],
            eta_min = params['eta_min']
        )
        return opt, sch

    def forward(self,
                images : Optional[torch.Tensor] = None,
                targets : Optional[str] = None,
                target_sequences : Optional[str] = None) -> torch.Tensor:
        
        outputs = self.donut_model(images)
        logits = outputs.logits

        if target_sequences is not None:
            loss = outputs.loss
            if self.training is True:
                metrics = self.monitor_metrics(outputs, target_sequences)
                return logits, loss, metrics
            else:
                return logits, loss
        return logits, 0, {}

            


