from speechbrain.augment.time_domain import AddNoise
import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import torchaudio
import torch
import mutagen
from mutagen.wave import WAVE
from IPython.display import Audio as AudioPlayer



data_cache_folder = '/scratch/cvlachos/LibriSpeech/'


train_librispeech_100 = load_dataset("openslr/librispeech_asr", "clean", split='train.100', cache_dir=data_cache_folder)

def compute_length(example):
    audio = example["audio"]
    length_seconds = len(audio["array"]) / audio["sampling_rate"]
    example["length_seconds"] = length_seconds
    return example

# Apply transformation
ds_with_lengths = train_librispeech_100.map(compute_length)

#ds_sorted_desc = ds_with_lengths.sort("length_seconds")

ds_with_lengths.save_to_disk("'/scratch/cvlachos/LibriSpeech/train_100_with_length")

print('done')