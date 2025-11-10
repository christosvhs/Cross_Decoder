from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
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
from cross_decoder import GPT2CrossDecoder 
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


wandb.login(key="38963e7021e91aea9d72a73f6c44738517a9de60")
wandb.init(project="Cross_Decoder", name="GPT2_Medium")
wb_config = wandb.config

#################################################### Parameters
corpus_data = "bookcorpus"
steps = 400
batch_size = 8 #687.500*2 steps
lr = 2e-4
betas = (0.9, 0.98)
eps = 1e-8
weight_decay = 0.01
warmup_steps = 4000
max_epoch = 5000
model_name = "gpt2-medium"
early_stop = 15
model_cache_folder = '/efs/cvlachos/model_cache/'
data_cache_folder = '/efs/cvlachos/data_cache/Bookcorpus'
output_dir = '/efs/cvlachos/SpeechBrain_LCM/gpt2_medium'
min_eval_loss = np.inf
count = 0
flag = False
total_steps = -1
start_epoch = 0
last_step = 0
max_length = 512
dev_percentage = 0.15
test_percentage = 0.15
training_report = 20

if not os.path.exists(output_dir) : os.makedirs(output_dir)

def warmup_schedule(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))  # Gradually increase
    else:
        return 1.0

#################################################### model loading
config = GPT2Config().from_pretrained(model_name)


model = GPT2LMHeadModel(config).to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
#scheduler = ReduceLROnPlateau(optimizer, 'min')
warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)

print("Total params :", model.num_parameters())
########################################## Data loading/preprocessing

data = load_dataset("bookcorpus",trust_remote_code=True, cache_dir=data_cache_folder)
train_bookcorpus = data['train'][:11000000]["text"]
dev_bookcorpus = data['train'][11000000:11000000+100000]["text"]

# train_bookcorpus = data['train'][:(len(data['train']) - int(len(data['train'])*dev_percentage) - int(len(data['train'])*test_percentage))]["text"]
# dev_bookcorpus = data['train'][(len(data['train']) - int(len(data['train'])*dev_percentage) - int(len(data['train'])*test_percentage)) : (len(data['train']) - int(len(data['train'])*test_percentage))]["text"]


# test_data = data['train'][(len(data['train']) - int(len(data['train'])*test_percentage)) : ]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)


train_loader = DataLoader(Dataset(train_bookcorpus), batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(Dataset(dev_bookcorpus), batch_size=batch_size, shuffle=False)



################################### Evaluation/testing
def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    total = 0
    all_token_count = 0 
    CE = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            
            batch = tokenizer(batch, padding="longest", truncation=True, return_tensors='pt', max_length=1024)
            input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

            labels = input_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            all_token_count += input_ids.shape[0]*input_ids.shape[1]
            CE += loss.item()*input_ids.shape[0]*input_ids.shape[1]


    print(f'Eval loss: {total_loss/len(test_loader)} || Eval PL: {np.exp(CE/all_token_count)} ')
    return total_loss/len(test_loader), np.exp(CE/all_token_count) 


####################################### Training
history = {"epoch": [], "step":[], "train_loss":[], "val_loss":[], "val_perplexity":[], "learning rate" :[]}


if "best.pt" in os.listdir(output_dir):
  checkpoint = torch.load(os.path.join(output_dir, 'best.pt'))
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
    

    for n, batch in enumerate(train_loader):
        
        batch = tokenizer(batch, padding="longest", truncation=True, return_tensors='pt', max_length=1024)

        st = n + last_step
        total_steps += 1
        
        
        optimizer.zero_grad()
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        labels = batch.input_ids.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()
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

          torch.save(checkpoint, os.path.join(output_dir, 'epoch_{}_batch_{}.pt'.format(epoch, st)))

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