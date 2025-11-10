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
from transformers import AutoFeatureExtractor, AutoConfig, AutoModel, AutoTokenizer, GPT2LMHeadModel


import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss



class GPT2EncoderXDecoder(nn.Module):
    
    def __init__(self, encoder_name = "facebook/hubert-base-ls960", decoder_name = "gpt2"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.decoder = GPT2LMHeadModel(mask_config).to(self.device)
        self.decoder.resize_token_embeddings(len(self.tokenizer))


    
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
        data = [ "[startoftext] " + i+" [endoftext]" for i in data]
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

    
    def generate(self, encoder_input = None, decoder_input_ids= None, max_new_tokens = 0):

        with torch.no_grad():
            out_encoder = self.encoder(input_values=encoder_input.input_values.to(self.device), 
                                       attention_mask=encoder_input.attention_mask.to(self.device))

            out_ids = self.decoder.generate(
                encoder_hidden_states=out_encoder.last_hidden_state,
                #encoder_attention_mask=encoder_input.attention_mask,
                input_ids=decoder_input_ids,
                max_length=50)
            
        return self.decode(out_ids)





# if __name__ == "__main__":
#     import warnings
#     warnings.filterwarnings("ignore")       
#     from datasets import load_dataset

#     data_cache_folder = '/scratch/cvlachos/LibriSpeech/'
#     output_dir = '/scratch/cvlachos/SpeechBrain_LCM/encoderXdecoder/hubert_gpt2_baseline'

#     train_librispeech = load_dataset("openslr/librispeech_asr", "clean", split='train.100', cache_dir=data_cache_folder)
#     dev_librispeech = load_dataset("openslr/librispeech_asr", "clean", split='validation', cache_dir=data_cache_folder)   


#     model = GPT2EncoderXDecoder()
#     checkpoint = torch.load(os.path.join(output_dir, 'best.pt'), map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['state_dict'])

#     idx = 15

#     input_data = model.encoder_preprocess([dev_librispeech[idx]['audio']['array']])
#     label_data = model.decoder_tokenize([""])

    
    

#     # outputs = model(input=input_data.input_values, encoder_attention_mask=input_data.attention_mask, 
#     #                     labels=label_data.input_ids, decoder_attention_mask=label_data.attention_mask) 
    
#     # print(outputs)
#     print()
#     print('Prediction   :', model.generate(input_data, label_data.input_ids[:,:1])[0].replace("[startoftext] ",""))
#     print('Ground Truth :', dev_librispeech[idx]['text'].lower())

if __name__ == "__main__":
     
    from datasets import load_dataset
    from tqdm import tqdm
    from jiwer import wer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cache_folder = '/scratch/cvlachos/LibriSpeech/'
    output_dir = '/scratch/cvlachos/SpeechBrain_LCM/encoderXdecoder/hubert_gpt2_baseline'

    #train_librispeech = load_dataset("openslr/librispeech_asr", "clean", split='train.100', cache_dir=data_cache_folder)
    test_clean = load_dataset("openslr/librispeech_asr", split='test.clean', cache_dir=data_cache_folder)  
    test_other = load_dataset("openslr/librispeech_asr", split='test.other', cache_dir=data_cache_folder)  

    test_clean = load_dataset("openslr/librispeech_asr", "clean", split='train.100', cache_dir=data_cache_folder)


    model = GPT2EncoderXDecoder().to(device)
    checkpoint = torch.load(os.path.join(output_dir, 'best.pt'), map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])

    clean_error = 0

    for idx in tqdm(range(len(test_clean))): 
        input_data = model.encoder_preprocess([test_clean[idx]['audio']['array']])
        label_data = model.decoder_tokenize([""])  

        hypothesis =  model.generate(input_data.to(device), label_data.input_ids[:,:1].to(device))[0].replace("[startoftext] ","").replace("<|endoftext|>","")
        reference = test_clean[idx]['text'].lower()

        # print()
        # print('Prediction   :', hypothesis)
        # print('Ground Truth :', reference)

        clean_error += wer(reference, hypothesis)

    print("Clean WER", clean_error/len(test_clean))


    # other_error = 0

    # for idx in tqdm(range(len(test_other))): 
    #     input_data = model.encoder_preprocess([test_other[idx]['audio']['array']])
    #     label_data = model.decoder_tokenize([""])  

    #     hypothesis =  model.generate(input_data.to(device), label_data.input_ids[:,:1].to(device))[0].replace("[startoftext] ","").replace("<|endoftext|>","")
    #     reference = test_other[idx]['text'].lower()

    #     # print()
    #     # print('Prediction   :', hypothesis)
    #     # print('Ground Truth :', reference)

    #     other_error += wer(reference, hypothesis)

    # print("Clean WER", other_error/len(test_other))