Bootstrap: docker
From: ubuntu:16.04

%post

# Pasteur setup, ymmv
mkdir -p /selinux /misc /net
mkdir -p /pasteur
mkdir -p /local-storage /mnt/beegfs /baycells/home /baycells/scratch /c6/shared /c6/eb /local/gensoft2 /c6/shared/rpm /Bis/Scratch2 /mnt/beegfs
mkdir -p /c7/shared /c7/scratch /c7/home
# 

# update and requirements for miniconda
# non interactive debian
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get -y upgrade
apt-get -y install bzip2 curl

# install miniconda
curl -qsSLkO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
&& bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
&& rm Miniconda3-latest-Linux-x86_64.sh
/opt/miniconda3/bin/conda update conda && /opt/miniconda3/bin/conda update --all

# https://pytorch.org/get-started/locally/
PATH=/opt/miniconda3/bin:$PATH
export PATH
/opt/miniconda3/bin/conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# shapegan via pip
python3 -m pip install \
	 mesh-to-sdf \
	 trimesh \
	 pyopengl \
	 pyrender \
	 sklearn \
	 scikit-image \
	 pygame \
	 opencv-python  \

# missing X11?
apt-get -y update
apt-get -y install libx11-6 libxext6 libgl1 libsm6 libxrender1 libx11-dev nano git zip unzip libglu1-mesa libglib2.0
 
%environment
PATH=/opt/miniconda3/bin:$PATH
export PATH


