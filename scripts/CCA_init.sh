#!/usr/bin/env bash

cd /home/jovyan/home
conda install vim htop numpy matplotlib numba git astropy pyyaml

mkdir tools projects

cd tools
git clone https://github.com/isulta/itk
git clone https://github.com/agurvich/abg_python

cd ../projects
git clone https://github.com/isulta/massive-halos

export PYTHONPATH="${PYTHONPATH}:/home/jovyan/home/tools"

cd /home/jovyan/home/projects/massive-halos
git config --global user.email "isultan030@gmail.com"
git config --global user.name "Imran Sultan"

