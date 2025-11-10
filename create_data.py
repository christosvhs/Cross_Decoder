
import pandas as pd
from pyarrow import dataset
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
import torch
from datasets import DatasetDict
import os


# data = load_dataset("CLAPv2/LibriSpeech", trust_remote_code=True, cache_dir="/scratch/cvlachos/LibriSpeech")
# data = load_dataset("openslr/librispeech_asr", cache_dir="/scratch/cvlachos/LibriSpeech")





# CONFIGS
MODEL_NAME = "facebook/wav2vec2-base-960h"
OUTPUT_DIR = "/scratch/cvlachos/LibriSpeech/LibriSpeechBatches/dev"
SPLIT = "validation"  # You can change to train.360 or train.960
NUM_PROC = 4         # Adjust for your CPU cores
BATCH_SIZE = 8

# Load processor (tokenizer + feature extractor)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# Load LibriSpeech (clean subset)
dataset = load_dataset("openslr/librispeech_asr", "clean", split=SPLIT, cache_dir="/scratch/cvlachos/LibriSpeech")

# Ensure audio is in correct format
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# Preprocess function
def preprocess_batch(batch):
    audio = batch["audio"]
    input_values = processor(audio["array"], sampling_rate=16000).input_values[0]
    return {"input_values": input_values, "labels": batch["text"]}

# Apply preprocessing with multiprocessing
print("Preprocessing... This might take a while.")
processed_dataset = dataset.map(
    preprocess_batch,
    remove_columns=dataset.column_names,
    num_proc=NUM_PROC
)

# Save processed dataset to disk
print(f"Saving preprocessed dataset to: {OUTPUT_DIR}")
processed_dataset.save_to_disk(OUTPUT_DIR)

print("Done! You can now reload this dataset using `load_from_disk()`.")


