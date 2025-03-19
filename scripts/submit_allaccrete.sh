#!/bin/bash

# Define an array of strings
args=("h206_A1_res33000" "h113_A4_res33000" "h29_A2_res33000" "h2_A8_res33000")

# Loop through the arguments and submit each one to sbatch
for arg in "${args[@]}"
do
    sbatch /home/ias627/projects/massive_halos/scripts/jobsubmit_quest_serial_paper.sh "$arg"
done
