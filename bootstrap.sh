#!/bin/bash

set -e  # fail and exit on any command erroring

root_dir=$(cd `dirname $0`; pwd)
checkpoint_dir=${root_dir}/checkpoint
inputs_dir=${root_dir}/inputs
outputs_dir=${root_dir}/outputs

if [ ! -d ${inputs_dir} ];then
  mkdir ${inputs_dir}
fi

if [ ! -d ${outputs_dir} ];then
  mkdir ${outputs_dir}
  mkdir ${outputs_dir}/results
  mkdir ${outputs_dir}/test_results
fi

if [ ! -d ${checkpoint_dir} ];then
  mkdir ${checkpoint_dir}
fi
