#!/bin/bash
#SBATCH --job-name=encoderxdecoder
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=cpu
#SBATCH --exclusive
#SBATCH --output=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/calculate_lengths.out
#SBATCH --error=/fsx3/workspace2/cvlachos/projects/Cross_Decoder/out/e_calculate_lengths.out

python calculate_lengths.py 
