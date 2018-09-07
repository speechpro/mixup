FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER Khokhlov Yuri <khokhlov@speechpro.com>

WORKDIR /speechpro

RUN apt-get -qq -y update \
  && apt-get -qq -y --no-install-recommends install \
    apt-utils locales git curl wget subversion sox zlib1g-dev \
    ca-certificates bzip2 cmake htop lsb-release \
    build-essential automake autoconf libboost-all-dev mc

ENV CUDA_HOME=/usr/local/cuda

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
  && bash /tmp/miniconda.sh -bfp /usr/local \
  && rm -rf /tmp/miniconda.sh \
  && conda install -y python=2 \
  && conda install -y python=3 \
  && conda update conda \
  && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/speechpro/mixup.git \
  && cd mixup && git submodule init && git submodule update

CMD /bin/bash

