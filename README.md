Licence
-------
Apache 2.0

Installation guide
==================

Preparing prerequisites
-----------------------

### Install boost
sudo apt-get install libboost-dev

### Install CMake
sudo apt-get install cmake

### Install git
sudo apt-get install git

Building project
================

### Clone mixup project repository
git clone https://github.com/speechpro/mixup.git
cd mixup

### Register submodule Kaldi
git submodule init

### Clone submodule Kaldi
git submodule update

### Build Kaldi dependencies
cd kaldi/tools
make

Or if you want to speedup the building process run: make -j $(nproc)
In case of errors or if you want to check the prerequisites for Kaldi see INSTALL file.

### Build Kaldi
cd ../src
./configure --shared
make depend -j $(nproc)
make -j $(nproc)
In case of errors or for additinal  see INSTALL file.

### Generate mixup project
cd ../..
mkdir build
cd build
cmake ..

### Build mixup modules
make -j $(nproc)

### Install mixup modules
make install
This operation will place mixup modules in to the corresponding Kaldi binary folders.

How to use
==========


