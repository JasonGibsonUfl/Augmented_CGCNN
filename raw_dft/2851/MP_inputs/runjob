#!/bin/bash
#SBATCH --job-name=2851#SBATCH -o out_%j.log
#SBATCH -e err_%j.log
#SBATCH --qos=hennig
#SBATCH --ntasks=16
#SBATCH --ntasks-per-socket=16
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000mb

#SBATCH -t 6:00:00

cd $SLURM_SUBMIT_DIR

module purge
module load intel/2019.1.144
module load openmpi/4.0.1

srun --mpi=pmix_v3 /home/joshuapaul/vasp_10-23-19_5.4.4/bin/vasp_stand > job.log
echo Done