#!/bin/bash
#SBATCH --job-name=gpt2_encoderdecoerx_tok
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=gpu-1
#SBATCH --constraint=g5.2xlarge
#SBATCH --exclusive
#SBATCH --output=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/d_encoderdecoderx_gpt2_dapters_regLoss_tokenizer%j.out
#SBATCH --error=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/e_encoderdecoderx_gpt2_adapters_regLoss_tokenizer%j.out

#python encoderDecoderX_reg_tokenizer/train_cross_decoder_gpt2.py 
python encoderDecoderX_reg_tokenizer/modeling_cross_decoder_gpt2_adapters_regLoss_tok.py 
