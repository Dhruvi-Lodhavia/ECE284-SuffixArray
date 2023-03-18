#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
cd build
cmake  -DTBB_DIR=${HOME}/oneTBB-2019_U9  -DCMAKE_PREFIX_PATH=${HOME}/oneTBB-2019_U9/cmake ..
make -j4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release

## Basic run with 8 threads and kmerSize of 12
## HINT: needs changes to parameter values for Assignment 1
# ./suffix --reference ../data/reference.fa --numThreads 8

## Run command using nvprof profiler
## HINT: Useful for profiling tasks in Assignment 2 
./suffix -r ../data/reference.fa -T 8
