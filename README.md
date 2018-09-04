Licence
-------
[Apache 2.0](https://github.com/speechpro/mixup/blob/master/LICENSE)

Installation guide
==================

Prerequisites
-------------

### Install boost
    $ sudo apt-get install libboost-all-dev

### Install CMake
    $ sudo apt-get install cmake

### Install git
    $ sudo apt-get install git

Building project
----------------

### Clone mixup project repository
    $ git clone https://github.com/speechpro/mixup.git
    
    $ cd mixup

### Clone Kaldi submodule
    $ git submodule init
    
    $ git submodule update

### Build Kaldi dependencies
    $ cd kaldi/tools
    
    $ make

or if you want to speedup the building process run:

    $ make -j $(nproc)

In case of errors or if you want to check the prerequisites for Kaldi see INSTALL file.

### Build Kaldi
    $ cd ../src
    
    $ ./configure --shared
    
    $ make depend -j $(nproc)
    
    $ make -j $(nproc)
    
In case of errors or for additinal building options see INSTALL file.

### Generate mixup project
    $ cd ../..
    
    $ mkdir build
    
    $ cd build
    
    $ cmake ..

### Build mixup modules
    $ make -j $(nproc)

### Install mixup modules
    $ make install
    
This operation will place mixup modules in to the corresponding Kaldi binary folders.

How to use
==========

Utilities nnet3-mixup-egs and nnet3-chain-mixup-egs are intended to be used instead of nnet3-copy-egs and nnet3-chain-copy-egs in training scripts. In order to use mixup utilities you should replace nnet3-copy-egs and/or nnet3-chain-copy-egs here

https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/train/frame_level_objf/common.py

line ~122
```
ark,bg:nnet3-copy-egs {frame_opts} {multitask_egs_opts}
```
with
```
ark,bg:nnet3-mixup-egs {frame_opts} {multitask_egs_opts}
```
and here

https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/train/chain_objf/acoustic_model.py

line ~199
```
ark,bg:nnet3-chain-copy-egs {multitask_egs_opts}
```
with 
```
ark,bg:nnet3-chain-mixup-egs {multitask_egs_opts}
```
respectively.

Supported parameters
====================
Mixup utilities have a number of parameters and modes of operation. In order to simplify their embedding all parameters can be passed in two ways: in command line and as environment variables.

### nnet3-mixup-egs
|Command line|Environment variable|Allowable values|Meaning|
|---|---|---|---|
|--mix-mode|MIXUP_MIX_MODE|local, global, class, shift|Mixup mode|
|--distrib|MIXUP_DISTRIB|uniform:min,max, beta:alpha, beta2:alpha|Mixup scaling factors distribution|
|--transform|MIXUP_TRANSFORM|sigmoid:k|Mixup scaling factor transform function for labels|
|--min-num|MIXUP_MIN_NUM|integer > 0|Minimum number of admixtures|
|--max-num|MIXUP_MAX_NUM|integer >= min-num|Maximum number of admixtures|
|--min-shift|MIXUP_MIN_SHIFT|integer > 0|Minimum sequence shift size (shift mode)|
|--max-shift|MIXUP_MAX_SHIFT|integer > min-shift|Maximum sequence shift size (shift mode)|
|--fixed-egs|MIXUP_FIXED_EGS|float in the range (0, 1)|Portion of examples to leave untouched|
|--fixed-frames|MIXUP_FIXED_FRAMES|float in the range (0, 1)|Portion of frames to leave untouched|
|--left-range|MIXUP_LEFT_RANGE|integer > 0|Left range to pick an admixture frame (local mode)|
|--right-range|MIXUP_RIGHT_RANGE|integer > 0|Right range to pick an admixture frame (local mode)|
|--buff-size|MIXUP_BUFF_SIZE|integer > 0|Buffer size for data shuffling (global mode)|
|--compress|MIXUP_COMPRESS|0, 1|Compress features and i-vectors|

References
==========
[1] https://docs.google.com/viewerng/viewer?url=https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2191.pdf
