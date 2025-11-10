#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu-1
#SBATCH --constraint=g5.2xlarge
#SBATCH --exclusive
#SBATCH --output=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/d_hubert_bert_train_eval.out
#SBATCH --error=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/e_hubert_bert_train_eval.out

python encoderΧdecoder/train_hubert_bert.py 
