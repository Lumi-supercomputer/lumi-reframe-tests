#!/bin/bash

echo Starting at `date`
sleep 15
echo Step $SLURM_STEP_ID Rank $PMI_RANK on `hostname`
echo Ending at `date`
