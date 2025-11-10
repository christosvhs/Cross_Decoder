#!/bin/bash
#SBATCH --job-name=gpt2_encoderdecoerx
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu-1
#SBATCH --constraint=g5.2xlarge
#SBATCH --exclusive
#SBATCH --output=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/d_encoderdecoderx_gpt2_dapters_regLoss_AUG%j.out
#SBATCH --error=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/e_encoderdecoderx_gpt2_adapters_regLoss_AUG%j.out

#python encoderDecoderX_reg_aug/train_cross_decoder_gpt2.py 
python encoderDecoderX_reg_aug/modeling_cross_decoder_gpt2_adapters_regLoss.py 
