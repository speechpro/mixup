FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER Khokhlov Yuri <khokhlov@speechpro.com>

WORKDIR /speechpro

RUN apt-get -qq -y update \
  && apt-get -qq -y --no-install-recommends install \
    apt-utils locales git curl wget subversion sox zlib1g-dev \
    ca-certificates bzip2 cmake htop lsb-release libatlas3-base \
    build-essential automake autoconf libtool libboost-all-dev mc \
    less file vim

ENV CUDA_HOME /usr/local/cuda

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
  && bash /tmp/miniconda.sh -bfp /usr/local \
  && rm -rf /tmp/miniconda.sh \
  && conda install -y python=2 \
  && conda update conda \
  && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN git clone --recursive https://github.com/speechpro/mixup.git \
  && cd mixup/kaldi/tools && make -j $(nproc) && rm -rf openfst/src \
  && cd ../src && ./configure --shared \
  && make depend -j $(nproc) && make -j $(nproc) \
  && find ./ -type f -name '*.a' -delete \
  && find ./ -type f -name '*.o' -delete \
  && cd ../.. && mkdir build && cd build \
  && cmake -DCMAKE_LIBRARY_PATH=$CUDA_HOME/lib64/stubs .. \
  && make -j $(nproc) && make install \
  && cd .. && rm -r build

ENV KALDI_ROOT /speechpro/mixup/kaldi
ENV PATH $KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:\
${KALDI_ROOT}/src/bin:${KALDI_ROOT}/src/chainbin:${KALDI_ROOT}/src/featbin:\
${KALDI_ROOT}/src/fgmmbin:${KALDI_ROOT}/src/fstbin:${KALDI_ROOT}/src/gmmbin:\
${KALDI_ROOT}/src/ivectorbin:${KALDI_ROOT}/src/kwsbin:${KALDI_ROOT}/src/latbin:\
${KALDI_ROOT}/src/lmbin:${KALDI_ROOT}/src/nnet2bin:${KALDI_ROOT}/src/nnet3bin:\
${KALDI_ROOT}/src/nnetbin:${KALDI_ROOT}/src/online2bin:${KALDI_ROOT}/src/onlinebin:\
${KALDI_ROOT}/src/rnnlmbin:${KALDI_ROOT}/src/sgmm2bin:${KALDI_ROOT}/src/sgmmbin:\
${KALDI_ROOT}/src/tfrnnlmbin:$PATH

ENV LD_LIBRARY_PATH $KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH

CMD ["/bin/bash"]

