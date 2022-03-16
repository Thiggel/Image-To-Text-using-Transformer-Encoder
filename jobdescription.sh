#!/bin/bash

#SBATCH --job-name=MultiModalTransformer
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.m.de.sousa.horta.osorio.laitenberger@student.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --time=7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:6

module load CUDA/9.1.85
module load Boost/1.66.0-foss-2018a-Python-3.6.4

source /data/$USER/.envs/python386-bachelors/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python3 main.py