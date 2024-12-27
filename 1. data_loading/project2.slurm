#!/bin/bash
#SBATCH --partition=courses
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=04:00:00
#SBATCH --output=/home/wei.shao/project2.log
#SBATCH --error=/home/wei.shao/project2.err

module load anaconda3/2021.05

source activate base

python /home/wei.shao/Project/project2.py

python /home/wei.shao/Project/project_performance.py

