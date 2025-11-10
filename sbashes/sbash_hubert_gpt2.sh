#!/bin/bash
#SBATCH --job-name=baseline_gpt2
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu-1
#SBATCH --constraint=g5.2xlarge
#SBATCH --exclusive
#SBATCH --output=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/d_hubert_gpt2_adapters.out
#SBATCH --error=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/e_hubert_gpt2_adapters.out

python encoderDecoderX_reg/train_hubert_gpt2.py 
#python encoderDecoderX/modeling_hubert_gpt2_adapters.py 
