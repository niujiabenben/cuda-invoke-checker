#! /bin/bash


LD_LIBRARY_PATH=./build/lib
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export LD_LIBRARY_PATH

python ./script/test_checker.py
