#!/bin/bash

#SBATCH --qos normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH -J test


#module load anaconda/2022.5
#source activate /public/groups/ai4ec/libs/conda/deepmd/2.2.9/gpu  
#conda activate dmff_310
module load cuda/12.1
#python run.py > vacuum_slab_corr_new_conp_ffield.out 
#dp train input.json 1>>dp_train.stderr 2>>dp_train.stdout
#dp freeze -o graph.pb 1>>dp_freeze.stderr 2>>dp_freeze.stdout
python test_electrode.py 
