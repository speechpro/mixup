FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER Khokhlov Yuri <khokhlov@speechpro.com>

WORKDIR /stc

RUN apt-get -qq -y update \
  && apt-get -qq -y --no-install-recommends install \
    git curl wget subversion sox zlib1g-dev python3 \
    cmake libatlas3-base build-essential automake \
    autoconf libtool libboost-all-dev less vim mc

ENV CUDA_HOME /usr/local/cuda

RUN git clone --recursive https://github.com/speechpro/mixup.git \
  && cd mixup && git checkout optim && cd .. \
  && cd mixup/kaldi/tools && make -j $(nproc) && rm -rf openfst/src \
  && cd ../src && ./configure --shared \
  && make depend -j $(nproc) && make -j $(nproc) \
  && find ./ -type f -name '*.a' -delete \
  && find ./ -type f -name '*.o' -delete \
  && cd ../.. && mkdir build && cd build \
  && cmake -DCMAKE_LIBRARY_PATH=$CUDA_HOME/lib64/stubs .. \
  && make -j $(nproc) && make install \
  && cd .. && rm -r build

RUN apt-get -qq -y purge libboost* cmake build-essential automake autoconf libtool

ENV KALDI_ROOT /stc/mixup/kaldi
ENV PATH $KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:\
${KALDI_ROOT}/src/bin:${KALDI_ROOT}/src/chainbin:${KALDI_ROOT}/src/featbin:\
${KALDI_ROOT}/src/fgmmbin:${KALDI_ROOT}/src/fstbin:${KALDI_ROOT}/src/gmmbin:\
${KALDI_ROOT}/src/ivectorbin:${KALDI_ROOT}/src/kwsbin:${KALDI_ROOT}/src/latbin:\
${KALDI_ROOT}/src/lmbin:${KALDI_ROOT}/src/nnet2bin:${KALDI_ROOT}/src/nnet3bin:\
${KALDI_ROOT}/src/nnetbin:${KALDI_ROOT}/src/online2bin:${KALDI_ROOT}/src/onlinebin:\
${KALDI_ROOT}/src/rnnlmbin:${KALDI_ROOT}/src/sgmm2bin:${KALDI_ROOT}/src/sgmmbin:\
${KALDI_ROOT}/src/tfrnnlmbin:$PATH

CMD ["/bin/bash"]

