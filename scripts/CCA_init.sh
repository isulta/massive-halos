#!/usr/bin/env bash
set -e

cd /home/jovyan/home
conda install -y vim htop numpy matplotlib numba git astropy pyyaml tqdm hdf5
conda install -y -c conda-forge ffmpeg
conda install -y -c conda-forge silx
conda install -y -c anaconda joblib
conda install -y -c conda-forge screen

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

export PYTHONPATH="${PYTHONPATH}:/home/jovyan/home/tools:/home/jovyan/home/tools/itk:/home/jovyan/home/projects/massive-halos"

cd /home/jovyan/home/projects/massive-halos
git config --global user.email "isultan030@gmail.com"
git config --global user.name "Imran Sultan"
git config --global alias.st status
git config --global alias.ci commit

mkdir /home/jovyan/home/data
python -c "from scripts.simulations import controlsims; controlsims(create_symlinks=True)"