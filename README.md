Introduction
============
This repository contains Kaldi-compatible implementation of the mixup technique presented in the Interspeech 2018 paper "An Investigation of Mixup Training Strategies for Acoustic Models in ASR".

If you use this code for your research, please cite our paper:

```
@inproceedings{Medennikov_mixup2018,
  author={Ivan Medennikov and Yuri Khokhlov and Aleksei Romanenko and Dmitry Popov and Natalia Tomashenko and Ivan Sorokin and Alexander Zatvornitskiy},
  title={An Investigation of Mixup Training Strategies for Acoustic Models in ASR},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={2903--2907},
  doi={10.21437/Interspeech.2018-2191},
  url={http://dx.doi.org/10.21437/Interspeech.2018-2191}
}
```

If you have any questions on the paper or this implementation, please ask Ivan Medennikov (medennikov@speechpro.com).

Licence
=======
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

You may need to add line
```
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
```
to your ``path.sh``.

How to use
==========

Utilities nnet3-mixup-egs and nnet3-chain-mixup-egs are intended to be used instead of nnet3-copy-egs and nnet3-chain-copy-egs in Kaldi training scripts. In order to use mixup utilities you should replace nnet3-copy-egs and/or nnet3-chain-copy-egs here

[``common.py, rev. eacf34a85ab7ece6a76bd73b9443bc2fe62ac6f1``](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/train/frame_level_objf/common.py)

method **``train_new_models()``**, line ~122
```
ark,bg:nnet3-copy-egs {frame_opts} {multitask_egs_opts}
```
with
```
ark,bg:nnet3-mixup-egs {frame_opts} {multitask_egs_opts}
```
and here

[``acoustic_model.py, rev. bba22b58407a3243e3fa847986753266e122d015``](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/train/chain_objf/acoustic_model.py)

method **``train_new_models()``**, line ~199
```
ark,bg:nnet3-chain-copy-egs {multitask_egs_opts}
```
with 
```
ark,bg:nnet3-chain-mixup-egs {multitask_egs_opts}
```
respectively.

Program options
===============
Mixup utilities have a number of parameters and modes of operation. In order to simplify their embedding all parameters can be passed in two equivalent ways: as command line program options and as environment variables.

You can find detailed explanation of the parameters and investigation of the mixup effectiveness in various operation modes in [**``[1]``**](https://docs.google.com/viewerng/viewer?url=https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2191.pdf).

### nnet3-mixup-egs
|Command line|Environment variable|Allowable values|Default|Meaning|
|---|---|---|---|---|
|--mix-mode|MIXUP_MIX_MODE|local, global, class, shift|global|Mixup mode|
|--distrib|MIXUP_DISTRIB|uniform:min,max, beta:alpha, beta2:alpha|uniform:0.0,0.5|Mixup scaling factors distribution|
|--transform|MIXUP_TRANSFORM|"", sigmoid:k|""|Mixup scaling factor transform function for labels|
|--min-num|MIXUP_MIN_NUM|integer > 0|1|Minimum number of admixtures|
|--max-num|MIXUP_MAX_NUM|integer >= min-num|1|Maximum number of admixtures|
|--min-shift|MIXUP_MIN_SHIFT|integer > 0|1|Minimum sequence shift size (shift mode)|
|--max-shift|MIXUP_MAX_SHIFT|integer >= min-shift|3|Maximum sequence shift size (shift mode)|
|--fixed-egs|MIXUP_FIXED_EGS|float in the range [0, 1]|0.1|Portion of examples to leave untouched|
|--fixed-frames|MIXUP_FIXED_FRAMES|float in the range [0, 1]|0.1|Portion of frames to leave untouched|
|--left-range|MIXUP_LEFT_RANGE|integer > 0|3|Left range to pick an admixture frame (local mode)|
|--right-range|MIXUP_RIGHT_RANGE|integer > 0|3|Right range to pick an admixture frame (local mode)|
|--buff-size|MIXUP_BUFF_SIZE|integer > 0|500|Buffer size for data shuffling (global mode)|
|--compress|MIXUP_COMPRESS|0, 1|0|Compress features and i-vectors|

### nnet3-chain-mixup-egs
|Command line|Environment variable|Allowable values|Default|Meaning|
|---|---|---|---|---|
|--mix-mode|MIXUP_MIX_MODE|global, shift|global|Mixup mode|
|--distrib|MIXUP_DISTRIB|uniform:min,max, beta:alpha, beta2:alpha|uniform:0.0,0.5|Mixup scaling factors distribution*|
|--scale-fst-algo|MIXUP_SCALE_FST_ALGO|"", default[:scale[,eps]], balanced[:scale[,eps]]|""|Scale supervision FSTs algorithm**|
|--swap-scales|MIXUP_SWAP_SCALES|true, false|false|Swap supervision FST scales|
|--max-super|MIXUP_MAX_SUPER|true, false|false|Get supervision from example with maximum scale|
|--min-shift|MIXUP_MIN_SHIFT|integer > 0|1|Minimum sequence shift size (shift mode)|
|--max-shift|MIXUP_MAX_SHIFT|integer >= min-shift|3|Maximum sequence shift size (shift mode)|
|--fixed|MIXUP_FIXED|float in the range [0, 1]|0.1|The portion of the data to leave untouched|
|--buff-size|MIXUP_BUFF_SIZE|integer > 0|500|Buffer size for data shuffling (global mode)|
|--frame-shift|MIXUP_FRAME_SHIFT|integer >= 0|0|Allows you to shift time values in the supervision data (excluding iVector data) - useful in augmenting data. Note, the outputs will remain at the closest exact multiples of the frame subsampling|
|--compress|MIXUP_COMPRESS|0, 1|0|Compress features and i-vectors|

\* **``Mixup scaling factors distribution.``** In case of --distrib=beta:alpha we use the standard beta probability distribution with symmetric shape (β=α). But when --distrib=beta2:alpha we use modified beta distribution: if sampled value ρ greater 0.5 we use (1-ρ).

```C++
float RandomScaleBeta2::Value() {
    const float value = (*distrib)(rand_gen);
    if (value <= 0.5) {
        return value;
    } else {
        return (1.0 - value);
    }
}
```

** **``Scale supervision FSTs algorithm.``** When merging supervision FSTs we apply epsilon restriction as folows. If scaling factor less **``eps``** we leave example FST unchanged. If 1.0 minus scaling factor less **``eps``** we use admixture FST instead of fusion. Default value of **``eps``** is 0.001.

```C++
void ExampleMixer::FuseGraphs(const fst_t& _admixture, float _admx_scale, fst_t& _example) const {
    if (_admx_scale < scale_eps) {
        return;
    } else if ((1.0 - _admx_scale) < scale_eps) {
        _example = _admixture;
        return;
    }
    ...
    ...
}
```

References
==========
**``[1]``** [Ivan Medennikov, Yuri Khokhlov, Aleksei Romanenko, Dmitry Popov, Natalia Tomashenko, Ivan Sorokin, Alexander Zatvornitskiy, "An investigation of mixup training strategies for acoustic models in ASR", Proceedings of the Annual Conference of International
Speech Communication Association (INTERSPEECH), 2018](https://docs.google.com/viewerng/viewer?url=https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2191.pdf)
