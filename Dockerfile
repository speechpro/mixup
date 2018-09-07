FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER Khokhlov Yuri <khokhlov@speechpro.com>

WORKDIR /speechpro

RUN apt-get -qq -y update \
  && apt-get -qq -y --no-install-recommends install \
    apt-utils locales git curl wget subversion sox zlib1g-dev \
    ca-certificates bzip2 cmake htop lsb-release libatlas3-base \
    build-essential automake autoconf libtool libboost-all-dev mc

ENV CUDA_HOME /usr/local/cuda

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
  && bash /tmp/miniconda.sh -bfp /usr/local \
  && rm -rf /tmp/miniconda.sh \
  && conda install -y python=2 \
  && conda update conda \
  && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN git clone --recursive https://github.com/speechpro/mixup.git \
  && cd mixup/kaldi/tools && make -j $(nproc) \
  && cd ../src && ./configure --shared \
  && make depend -j $(nproc) && make -j $(nproc) \
  && cd ../.. && mkdir build && cd build \
  && cmake -DCMAKE_LIBRARY_PATH=$CUDA_HOME/lib64/stubs .. \
  && make -j $(nproc) && make install

ENV KALDI_ROOT /speechpro/mixup/kaldi
ENV PATH $KALDI_ROOT/tools/openfst/bin:$PATH
ENV LD_LIBRARY_PATH $KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH

CMD ["$KALDI_ROOT/tools/config/common_path.sh && /bin/bash"]
#ENTRYPOINT $KALDI_ROOT/tools/config/common_path.sh && /bin/bash

