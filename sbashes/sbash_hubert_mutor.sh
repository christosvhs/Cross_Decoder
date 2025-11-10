#!/bin/bash
#SBATCH --job-name=hubert_mutor
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu-1
#SBATCH --constraint=g5.2xlarge
#SBATCH --exclusive
#SBATCH --output=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/d_hubert_mutor.out
#SBATCH --error=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/e_hubert_mutor.out

#python mutor_encoder_decoder/train_hubert_mutor.py 
python mutor_encoder_decoder/modeling_mutor.py 
