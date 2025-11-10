# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""


import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from datasets import Audio
from transformers import AutoFeatureExtractor, AutoConfig, AutoModel, AutoTokenizer, BertLMHeadModel


import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss



class BERTEncoderXDecoder(nn.Module):
    
    def __init__(self, encoder_name = "facebook/hubert-base-ls960", decoder_name = "bert-base-uncased"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_model_weights = BertLMHeadModel.from_pretrained(decoder_name)


        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            encoder_name, do_normalize=True, return_attention_mask=True)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        special = {"additional_special_tokens" : ["[startoftext]", "[endoftext]"]}   
        self.tokenizer.add_special_tokens(special)   
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.eos_token = "[endoftext]"


        mask_config = AutoConfig.from_pretrained(decoder_name)
        mask_config.add_cross_attention=True
        mask_config.is_decoder=True
        self.decoder = BertLMHeadModel(mask_config).to(self.device)
        self.decoder.load_state_dict(encoder_model_weights.state_dict(), strict=False)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        encoder_model_weights = None

    
    def encoder_preprocess(self, data):
        
        inputs = self.feature_extractor(
            data,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return inputs

    def decoder_tokenize(self, data):
        data = [ "[startoftext] " + i for i in data]
        return self.tokenizer(data, padding="longest", truncation=True, return_tensors='pt', max_length=512, add_special_tokens=False)
    
    def decode(self, data):        
        return self.tokenizer.batch_decode(data)
    
    def num_parameters(self):
        return self.encoder.num_parameters() + self.token_decoder.num_parameters() + self.decoder.num_parameters()
        

    def forward(self, input = None, encoder_attention_mask = None, labels = None, decoder_attention_mask=None):

        out_encoder = self.encoder(input_values=input, attention_mask=encoder_attention_mask)

        out = self.decoder(input_ids=labels.clone(), attention_mask=decoder_attention_mask, encoder_hidden_states=out_encoder.last_hidden_state, labels = labels)

        return out
    
    def get_token_encoder(self):
        return self.token_decoder
    
    def get_mask_decoder(self):
        return self.decoder    

    
    def generate(self, input_txt=None, max_new_tokens = 10):

        past_key_values = None
        position_ids = torch.tensor([[self.decoder_tokenize([i]).input_ids.shape[1]] for i in input_txt])

        input = self.decoder_tokenize(input_txt)

        input_ids = input.input_ids
        attention_mask = input.attention_mask

        self.token_decoder.eval()
        self.decoder.eval()

        self.decoder.generation_config.pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                
                out_encoder = self.token_decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                last_hidden_state = out_encoder.last_hidden_state

                input_decoder = torch.zeros((input_ids.shape[0],1), dtype = torch.long) + self.tokenizer.convert_tokens_to_ids("<|mask|>")

                out = self.decoder.generate(max_new_tokens=1, num_return_sequences=1, input_ids=input_decoder, attention_mask=attention_mask, encoder_hidden_states=last_hidden_state, position_ids=position_ids)
                
                pad_right = torch.ones((input_ids.shape[0], 1), dtype = torch.int)*self.tokenizer.eos_token_id
                att_right = torch.zeros((input_ids.shape[0], 1), dtype = torch.int)

                input_ids = torch.cat((input_ids, pad_right), dim=1)
                attention_mask = torch.cat((attention_mask, att_right), dim=1)

                for idx in range(position_ids.shape[0]):
                    input_ids[idx][position_ids[idx][-1]] = out[idx][-1]
                    attention_mask[idx][position_ids[idx][-1]] = 1
                    
                position_ids+=1
            
        return self.decode(input_ids)





if __name__ == "__main__":

    import pickle

    with open("encoder_decoder/toy_batch.pckl", 'rb') as f:
        batch = pickle.load(f)
   

    model = BERTEncoderXDecoder()
    
    input_data = model.encoder_preprocess([b["audio"]["array"] for b in batch])
    label_data = model.decoder_tokenize([b["text"].lower() for b in batch])

    outputs = model(input=input_data.input_values.to(dtype=torch.float32), encoder_attention_mask=input_data.attention_mask, 
                        labels=label_data.input_ids, decoder_attention_mask=label_data.attention_mask)
    
    print(outputs)
