#!/bin/bash

#OAR -n Training
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-fumultispoc
#OAR -p gpumodel='A100'

cd ~/unrolled_demosaicking

python3 train.py

git add logs/*
git commit -m "Added new training"
git push
