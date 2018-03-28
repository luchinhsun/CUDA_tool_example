#!/bin/sh
### Job Name
#PBS -N LUtest
### submits the job to the n4 queue
#PBS -q n2
### requests 2 nodes, and each node has 2 GPUs.
#PBS -l nodes=2:ppn=1
### Output files
#PBS -o openmpi_gnu.log
#PBS -e openmpi_gnu.err
### Declare job non-rerunable
#PBS -r n

export P4_GLOBMEMSIZE=99187896

cd $PBS_O_WORKDIR

NPROCS=`wc -l < $PBS_NODEFILE`

echo "Starting on `hostname` at `date`"

/opt/openmpi/intel/bin/mpirun --mca btl ^tcp -hostfile $PBS_NODEFILE -np $NPROCS -x LD_LIBRARY_PATH ./a.out

echo "Job Ended at `date`"
