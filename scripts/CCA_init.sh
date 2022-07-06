#!/usr/bin/env bash
set -e

cd /home/jovyan/home
conda install -y vim htop numpy matplotlib=3.4.2 numba git astropy pyyaml tqdm hdf5
conda install -y -c conda-forge ffmpeg
conda install -y -c conda-forge silx

mkdir tools projects

cd tools
git clone https://github.com/isulta/itk
git clone https://github.com/agurvich/abg_python
cd abg_python
pip install -e .
cd ..
git clone https://github.com/agurvich/FIRE_studio

cd ../projects
git clone https://github.com/isulta/massive-halos

export PYTHONPATH="${PYTHONPATH}:/home/jovyan/home/tools"

cd /home/jovyan/home/projects/massive-halos
git config --global user.email "isultan030@gmail.com"
git config --global user.name "Imran Sultan"
git config --global alias.st status
git config --global alias.ci commit

cd /home/jovyan/home
mkdir data
cd data
ln -s /home/jovyan/fire2/CR_suite/m12f_mass56000/cr_700 m12f_noAGNfb
ln -s /home/jovyan/fire2/CR_suite/m12i_mass56000/cr_700 m12i_noAGNfb
ln -s /home/jovyan/fire2/CR_suite/m12q_res57000/cr_700 m12q_noAGNfb