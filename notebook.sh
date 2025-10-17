#!/bin/bash
#SBATCH --account EUHPC_D27_102
#SBATCH --job-name=vllm_inference
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1            # 1 GPU
#SBATCH --cpus-per-task=8       # 1/4 of 32 CPUs
#SBATCH --mem=120G              # ~1/4 of total RAM (≈480 GB / 4)
#SBATCH --time=04:00:00         # walltime (hh:mm:ss)
#SBATCH --output=%x_%j.out

# get tunneling info
module load jupyter-notebook
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate weight-interp
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH=$(pwd)

port=$(shuf -i 10000-65000 -n 1)  # Get random port to avoid conflicts
node=$(hostname -s)
user=$(whoami)

echo "Jupyter Lab will run on node ${node}, port ${port}"
echo "Connect via: ssh -N -L ${port}:${node}:${port} ${user}@hendrixgate04fl"

echo "Hostname: $(hostname)"
echo "Full hostname: $(hostname -f)"
echo "IP addresses:"
hostname -I

# Get the actual IP address
node_ip=$(hostname -I | awk '{print $1}')

# Run Jupyter Lab
jupyter lab --no-browser --port=${port} --ip=${node}