#!/bin/bash
#BSUB -q bulletmpi
#BSUB -W 8:00
#BSUB -J Chinchilla-400-adg-train-DSG_0.684_0.17
#BSUB -oo Chinchilla-400-adg-train-DSG_0.684_0.17.%J.out
#BSUB -n 16
#BSUB -R span[ptile=16]
#BSUB -x 

mpirun --map-by ppr:8:node adg-train DSG_0.684_0.17.yaml

