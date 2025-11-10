#!/bin/bash
#SBATCH --job-name=gpt2_libri
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu-1
#SBATCH --constraint=g5.2xlarge
#SBATCH --exclusive
#SBATCH --output=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/d_encoderxdecoder_gpt2_train_eval.out
#SBATCH --error=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/e_encoderxdecoder_gpt2_train_eval.out

python encoderXdecoder/train_cross_decoder_gpt2.py 
