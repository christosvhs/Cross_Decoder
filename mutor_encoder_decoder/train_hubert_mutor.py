from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoFeatureExtractor
import pandas as pd
import numpy as np
import torch
import os
import json
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import json
import copy
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import sys
from modeling_mutor import MutorEncoderDecoder 
import wandb
from time import time
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key="2d69de9e3c64591a043301cb32cef81bcba9f3e5")
wandb.init(project="EncoderXDecoder_vol2", name="MuToR_Encoder_Decoder")
wb_config = wandb.config

#################################################### Parameters

steps = 200
batch_size = 8 
accumulations_steps = 8
lr = 2e-4
betas = (0.9, 0.98)
eps = 1e-8
weight_decay = 0.01
warmup_steps = 4000
max_epoch = 5000
early_stop = 15
data_cache_folder = '/scratch/cvlachos/LibriSpeech/'
output_dir = '/scratch/cvlachos/SpeechBrain_LCM/encoderXdecoder/hubert_mutor'
min_eval_loss = np.inf
count = 0
flag = False
total_steps = -1
start_epoch = 0
last_step = 0
max_length = 512
training_report = 20


if not os.path.exists(output_dir) : os.makedirs(output_dir)

def warmup_schedule(step):
    if step < warmup_steps:
        return 9*(1-(float(step)/float(max(1, warmup_steps)))) + 1
    else:
        return 1.0

#################################################### model loading
model = MutorEncoderDecoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
#scheduler = ReduceLROnPlateau(optimizer, 'min')
warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)


########################################## Data loading/preprocessing


train_librispeech = load_dataset("openslr/librispeech_asr", "clean", split='train.100', cache_dir=data_cache_folder)
dev_librispeech = load_dataset("openslr/librispeech_asr", "clean", split='validation', cache_dir=data_cache_folder)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)
    
def collate_fn(data):
   return data


train_loader = DataLoader(Dataset(train_librispeech), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(Dataset(dev_librispeech), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


################################### Evaluation/testing
def evaluate(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start = time()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data = model.encoder_preprocess([b["audio"]["array"] for b in batch])
            input_text = model.decoder_tokenize([b["text"].lower() for b in batch], add_mask=False)


            outputs = model(
                input=input_data.input_values,
                encoder_attention_mask=input_data.attention_mask,
                input_text=input_text
            )

            loss = outputs.loss  

            labels = input_text.input_ids.to(device).clone()
            labels[labels == model.tokenizer.pad_token_id] = -100

            n_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    print("Evaluation time:", time() - start)
    print(f"Eval loss: {avg_loss:.4f} || Eval Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity



####################################### Training
history = {"epoch": [], "step":[], "train_loss":[], "val_loss":[], "val_perplexity":[], "learning rate" :[]}


if "best.pt" in os.listdir(output_dir):
  checkpoint = torch.load(os.path.join(output_dir, 'best.pt'), map_location=torch.device(device))
  model.load_state_dict(checkpoint['state_dict'])
  start_epoch = checkpoint['epoch']
  optimizer.load_state_dict(checkpoint['optimizer'])
  warmup_scheduler.load_state_dict(checkpoint['warm_up'])
  last_step = checkpoint.get('step', 0)
  history = checkpoint.get("history", history)
  min_eval_loss = history["val_loss"][-1]

  for i in range(len(history["epoch"])):
     wandb.log({"Epoch": history['epoch'][i] if len(history['epoch']) > i else 0,
                 "Step": history['step'][i] if len(history['step']) > i else 0, 
                 "Running Average Training Loss": history['train_loss'][i] if len(history['train_loss']) > i else 0,
                 "Validation Loss" : history['val_loss'][i] if len(history['val_loss']) > i else 0, 
                 "Validation Perplexity" : history['val_perplexity'][i] if len(history['val_perplexity']) > i else 0, 
                "Learning Rate": history['learning rate'][i] if len(history['learning rate']) > i else 0,
                })
     

for epoch in range(start_epoch, max_epoch):
    model.train()
    total_loss = 0
    #total_loss = []
    optimizer.zero_grad()

    for n, batch in enumerate(train_loader):
        
        # with open("toy_batch.pckl", 'ab') as f:
        #   pickle.dump(batch, f)

        d = int(np.random.normal(loc=4))
        if d <=1 : d=2

        input_data = model.encoder_preprocess([b["audio"]["array"] for b in batch])
        input_text = model.decoder_tokenize([b["text"].lower() for b in batch], d=d)

        st = n + last_step
        total_steps += 1      

        outputs = model(input=input_data.input_values, encoder_attention_mask=input_data.attention_mask, 
                        input_text=input_text)
        loss = outputs.loss/accumulations_steps
        loss.backward()

        if (n+1) % accumulations_steps == 0:
          optimizer.step()
          optimizer.zero_grad()

        total_loss += outputs.loss.detach().cpu().item()
        #total_loss.append(loss.detach().cpu().item())

        warmup_scheduler.step()
        if st >= warmup_steps:  
          pass
          #cosine_scheduler.step()

        input_ids, attention_mask, labels = None, None, None
        torch.cuda.empty_cache()

        if total_steps%steps == 0:
          print('~~~~~~~~~~~~~~~~~~~~~~~~~ Epoch ', epoch, ' | step', st, '~~~~~~~~~~~~~~~~~~~~~~~~~')
          print(f'Training loss for step: {loss.detach().cpu().item()}')
          print(f'Average training loss: {total_loss/(n+1)}')
          #print(f'Average training loss: {np.mean(total_loss[-training_report:])}')

          eval_loss, eval_pl = evaluate(model, dev_loader)
          #scheduler.step(eval_loss)
          print(optimizer.param_groups[0]['lr'])

          wandb.log({"Epoch": epoch, "Step": st, "Running Average Training Loss": total_loss/(n+1), "Validation Loss" : eval_loss, "Validation Perplexity" : eval_pl, "Learning Rate": optimizer.param_groups[0]['lr']} )
          #wandb.log({"Epoch": epoch, "Step": st, "Running Average Training Loss": np.mean(total_loss[-training_report:]), "Validation Loss" : eval_loss, "Validation Perplexity" : eval_pl, "Learning Rate": optimizer.param_groups[0]['lr']} )

          history["epoch"].append(epoch)
          history["step"].append(st)
          history["train_loss"].append(total_loss/(n+1))
          #history["train_loss"].append(np.mean(total_loss[-training_report:]))
          history["val_loss"].append(eval_loss)
          history["val_perplexity"].append(eval_pl)
          history["learning rate"].append(optimizer.param_groups[0]['lr'])

          checkpoint = {
            "epoch" : epoch,
            "step" : st,
            "optimizer" : optimizer.state_dict(),
            "lr" : optimizer.param_groups[0]['lr'],
            "training_loss" : total_loss/(n+1),
            "validation_loss" : eval_loss,
            "state_dict" : model.state_dict(),
            "history" : history,
            "warm_up" : warmup_scheduler.state_dict()
          }

          torch.save(checkpoint, os.path.join(output_dir, 'last.pt'))

          if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            count = 0
            torch.save(checkpoint, os.path.join(output_dir, 'best.pt'))
          else :
            count+=1
            print("Early Stopping count...",count)

          # if count == early_stop :
          #   flag = True
          #   break

        sys.stdout.flush()
    last_step = 0
    if flag :
      break

wandb.finish()