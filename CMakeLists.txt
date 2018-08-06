cmake_minimum_required (VERSION 2.8)

if (DEFINED ENV{KaldiMixup_INCLUDED})
    return()
endif()
set (ENV{KaldiMixup_INCLUDED} 1)

project (KaldiMixup)

#####################
### Find packages ###
#####################

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

#
#  Find Boost
#
if("${BOOST_ROOT}" STREQUAL "" AND NOT "$ENV{BOOST_ROOT}" STREQUAL "")
  set(BOOST_ROOT $ENV{BOOST_ROOT})
endif()
find_package(Boost 1.57 REQUIRED COMPONENTS system filesystem thread date_time regex unit_test_framework chrono program_options)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  message("-- Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
  message("-- Boost_LIBRARIES:    ${Boost_LIBRARIES}")
endif()
add_definitions(-DBOOST_ALL_NO_LIB)
add_definitions(-DBOOST_FILESYSTEM_VERSION=3)

#
#  Find Atlas
#
find_package(Atlas REQUIRED)
if(Atlas_FOUND)
  include_directories(${ATLAS_INCLUDE_DIR})
  message("-- ATLAS_INCLUDE_DIR: ${ATLAS_INCLUDE_DIR}")
  message("-- ATLAS_LIBRARIES: ${ATLAS_LIBRARIES}")
endif()
add_definitions(-DHAVE_ATLAS)

#
#  Find CUDA
#
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message("-- CUDA_VERSION:          ${CUDA_VERSION}")
  message("-- CUDA_INCLUDE_DIRS:     ${CUDA_INCLUDE_DIRS}")
  message("-- CUDA_CUDA_LIBRARY:     ${CUDA_CUDA_LIBRARY}")
  message("-- CUDA_CUDART_LIBRARY:   ${CUDA_CUDART_LIBRARY}")
  message("-- CUDA_cublas_LIBRARY:   ${CUDA_cublas_LIBRARY}")
  message("-- CUDA_cusparse_LIBRARY: ${CUDA_cusparse_LIBRARY}")
  message("-- CUDA_curand_LIBRARY:   ${CUDA_curand_LIBRARY}")
  set(CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY})
  include_directories("${CUDA_INCLUDE_DIRS}")
  add_definitions(-DHAVE_CUDA=1)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

add_definitions(-DKALDI_DOUBLEPRECISION=0)
add_definitions(-DHAVE_POSIX_MEMALIGN)
add_definitions(-DHAVE_EXECINFO_H=1)
add_definitions(-DHAVE_CXXABI_H)

#####################
### Add externals ###
#####################

include_directories(${CMAKE_SOURCE_DIR}/kaldi/tools/openfst/include)
set(FST_LIBRARIES ${CMAKE_SOURCE_DIR}/kaldi/tools/openfst/lib/libfst.so)

include_directories(${CMAKE_SOURCE_DIR}/kaldi/src)
link_directories(${CMAKE_SOURCE_DIR}/kaldi/src/lib)
set(Kaldi_LIBRARIES "kaldi-nnet3;kaldi-chain;kaldi-cudamatrix;kaldi-decoder;kaldi-lat")
set(Kaldi_LIBRARIES "${Kaldi_LIBRARIES};kaldi-fstext;kaldi-hmm;kaldi-transform;kaldi-gmm")
set(Kaldi_LIBRARIES "${Kaldi_LIBRARIES};kaldi-tree;kaldi-util;kaldi-matrix;kaldi-base")

###################
### Compilation ###
###################

#
# Project nnet3-mixup-egs
#

add_executable(nnet3-mixup-egs src/nnet3bin/nnet3-mixup-egs.cc)
target_link_libraries(nnet3-mixup-egs ${Kaldi_LIBRARIES})
target_link_libraries(nnet3-mixup-egs rt dl)
target_link_libraries(nnet3-mixup-egs ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${FST_LIBRARIES} ${ATLAS_LIBRARIES})

#
# Project nnet3-chain-mixup-egs
#

add_executable(nnet3-chain-mixup-egs src/chainbin/nnet3-chain-mixup-egs.cc)
target_link_libraries(nnet3-chain-mixup-egs ${Kaldi_LIBRARIES})
target_link_libraries(nnet3-chain-mixup-egs rt dl)
target_link_libraries(nnet3-chain-mixup-egs ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${FST_LIBRARIES} ${ATLAS_LIBRARIES})

####################
### Installation ###
####################

install(TARGETS nnet3-mixup-egs RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/kaldi/src/nnet3bin)
install(TARGETS nnet3-chain-mixup-egs RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/kaldi/src/chainbin)

