#!/usr/local_rwth/bin/zsh
#SBATCH --time=2:30:00
#SBATCH --partition=c18g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tag_task
#SBATCH --account=#######


module load cuda/11.2
module load cudnn/8.2.1


source ~/.zshrc
conda activate tensor_new

# set environmental variables
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}

# print some useful info
echo $RANK
module list
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"

# run the script
bash script.sh



