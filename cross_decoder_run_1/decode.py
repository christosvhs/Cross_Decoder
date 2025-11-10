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
from cross_decoder_test import GPT2CrossDecoder 
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = ["Senior US officials are"]


#output_dir = '/efs/cvlachos/SpeechBrain_LCM/cross_decoder'
# model = GPT2CrossDecoder()

# checkpoint = torch.load(os.path.join(output_dir, 'best.pt'), map_location=torch.device(device))
# model.load_state_dict(checkpoint['state_dict'])

# outputs = model.generate(input_txt=inputs, max_new_tokens=10)
# print(outputs)


output_dir = '/efs/cvlachos/SpeechBrain_LCM/gpt2'
config = GPT2Config()
model = GPT2LMHeadModel(config)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
inputs = tokenizer(inputs, return_tensors="pt")

input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

checkpoint = torch.load(os.path.join(output_dir, 'best.pt'), map_location=torch.device(device))
model.load_state_dict(checkpoint['state_dict'])

print(tokenizer.batch_decode(model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)))



