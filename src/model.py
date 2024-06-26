from torchret import Model 
import torch.nn as nn 
import torch
import re 
from ast import literal_eval
from src.dataset import DonutDataset
from src.custom_optimizers import Ranger

from nltk import edit_distance
from typing import Optional, Dict
from src.config import *
import numpy as np 
import neptune
from neptune.types import File

class DonutModel(Model):
    def __init__(self, donut, processor) -> None:
        super().__init__()
        self.donut = donut
        self.processor = processor
        self.ignore_for_device = 'target_sequences'

        self.model_path = params['model_path']
        self.save_best_model = params['save_best_model']
        self.save_on_metric = params['save_on_metric']
        self.save_model_at_every_epoch = params['save_model_at_every_epoch']

        self.num_workers = 12
        self.pin_memory = True
        self.scheduler = True
        self.step_scheduler_after = 'epoch'
    
    def setup_logger(self):
        neptune_api = input('Enter-neptune-key : ')
        self.run = neptune.init_run(
            project=input('Enter-project-name : '),
            api_token=neptune_api,
            capture_stdout=True,   
            capture_stderr=True,      
            capture_traceback=True,    
            capture_hardware_metrics=True,  
            source_files='src/*.py' 
        )
        self.run['parameters'] = params

    def train_one_step_logs(self, batch_id, data, logits, loss, metrics):
        self.run['train/step-loss'].append(loss)


    def valid_one_step_logs(self, batch_id, data, logits, loss, metrics, temp_pred, temp_answer):
        self.run['valid/step-loss'].append(loss)
        if batch_id % 3 == 0:
            images = data['images']
            images = images.permute(0, 2, 3, 1).squeeze().cpu()
            for i in range(len(images)):
                description = f'True Label:{temp_answer[i]}\nPrediction:{temp_pred[i]}'
                self.run["valid/prediction_example"].append(File.as_image(images[i]), description = description)

    def train_one_epoch_logs(self, loss, monitor):
        self.run['train/loss'].append(loss)
        self.run['train/monitors'].append(monitor)
    
    def valid_one_epoch_logs(self, loss, monitor):
        self.run['valid/loss'].append(loss)
        self.run['valid/monitors'].append(monitor)
    
    def valid_model_fn(self, batch_id, data):
        for k, v in data.items():
            if k == self.ignore_for_device:
                pass
            else:
                data[k] = v.to(self.device)
                batch_size = data[k].shape[0]
        answers = data['target_sequences']
        decoder_input_ids = torch.full((batch_size, 1), self.donut.config.decoder_start_token_id, device = self.device)
        output = self.donut.generate(
                data['images'],
                decoder_input_ids = decoder_input_ids,
                max_length = params['max_length'],
                early_stopping = False, 
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
        temp_pred = []
        temp_answer = []
        
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            sim = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(1 - sim)
            j_pred = self.processor.token2json(pred)
            temp_pred.append(j_pred)
            j_answer = self.processor.token2json(answer)
            temp_answer.append(j_answer)

        final_score = torch.tensor(sum(scores) / len(scores))
        metrics = {'edit_distance' : final_score}

        return temp_pred, temp_answer, output_logits, loss, metrics

    def monitor_metrics(self, images, target_sequences):

        answers = target_sequences
        batch_size = images.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), self.donut.config.decoder_start_token_id, device = self.device)
        output = self.donut.generate(
                images,
                decoder_input_ids = decoder_input_ids,
                max_length = params['max_length'],
                early_stopping = False, 
                pad_token_id = self.processor.tokenizer.pad_token_id,
                eos_token_id = self.processor.tokenizer.eos_token_id,
                use_cache = True, 
                num_beams = 1,
                bad_words_ids = [[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate = True
            )
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

        final_score = torch.tensor(sum(scores) / len(scores))
        metrics = {'edit_distance' : final_score}

        return metrics
    
    def valid_one_step(self, batch_id, data):
        temp_pred, temp_answer, _, loss, metrics = self.valid_model_fn(batch_id, data)
        if self.logger is True:
            self.valid_one_step_logs(batch_id, data, _, loss, metrics, temp_pred, temp_answer)
        return loss, metrics
    
    def fetch_optimizer(self):
        opt = Ranger(
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
                targets : Optional[torch.Tensor] = None,
                target_sequences : Optional[str] = None) -> torch.Tensor:
        
        outputs = self.donut(pixel_values = images, labels = targets)
        logits = outputs.logits
        loss = outputs.loss
        if self.training is True:
            metrics = self.monitor_metrics(images, target_sequences)
            return logits, loss, metrics
        return logits, loss