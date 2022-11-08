#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH --workdir=/home/amji/shortcuts-chest-xray/ # working directory
#SBATCH --gres=gpu:1 # use gpu
#SBATCH --time=12:00:00 # time limit
#SBATCH --mem=64G # memory per node
#SBATCH --partition=red

#SBATCH -o chtest.%N.%J.%u.out #output file
#SBATCH -e chtest.%N.%J.%u.err # error file

echo "Running on $(hostname):" # showing which node it is running on

module load Anaconda3/2021.05 # loading anaconda

# sourcing our .bashrc
source /home/amji/.bashrc

# activate the virtual environment
conda activate torchgpu

# run script
python -u bin/test_model.py --device 0 --num_workers 2 --model_path /home/amji/shortcuts-chest-xray/logdirs/logdir-30k-2 --in_csv_path /home/amji/shortcuts-chest-xray/config/my_test.csv --cfg_path /home/amji/shortcuts-chest-xray/config/
