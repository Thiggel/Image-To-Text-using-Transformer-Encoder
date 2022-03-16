#!/bin/bash

#SBATCH --job-name=MultiModalTransformer
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.m.de.sousa.horta.osorio.laitenberger@student.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load Python/3.8.6-GCCcore-10.2.0

source /data/$USER/.envs/python386-bachelors/bin/activate

pip install -r requirements.txt

python3 main.py
