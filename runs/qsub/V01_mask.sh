#!/bin/sh
#qsub -q gpu -l nodes=1:ppn=1:gpus=1,mem=120g,walltime=24:00:00 V01_mask.sh
module load anaconda

source /hpf/largeprojects/smiller/users/Katharine/python_environments/monai/bin/activate
cd /hpf/largeprojects/smiller/users/Katharine/brain_segmentation/segment/
python /hpf/largeprojects/smiller/users/Katharine/brain_segmentation/segment/run.py /hpf/largeprojects/smiller/users/Katharine/brain_segmentation/config_files/V01/mask.txt False
