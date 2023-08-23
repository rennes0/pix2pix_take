#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=7
#PBS -l place=shared
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N pix2pix			

cd  ~/pix2pix/pix2pix_take										

source ~/.bashrc											
conda activate DCGAN #這裡要記得改你的虛擬環境
module load cuda-11.7			
python3 main.py	--num_epoch 10000  
#--add_attribute d_DI_AR
