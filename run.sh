#! /bin/bash


LD_LIBRARY_PATH=./build/lib
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export LD_LIBRARY_PATH

# python ./script/test_checker.py
# python ./script/test_checker_mt.py 1000 sequential
# python ./script/test_checker_mt.py 1000 parallel
