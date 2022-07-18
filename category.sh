#!/bin/bash

input_dim=2
hidden_dim=2
layer=2
flows=1
loss='normal'
train_ratio=10
de_type='sum'

input_dir='inputs'
output_root_dir='outputs/main_results'
req_dist='zipf'
test_workloads=('cate_longlat-200M')
declare -A seeds
seeds=([longitudes-200M]=1000000013
        [longlat-200M]=1000000007
        [ycsb-200M]=1000000013
        [lognormal-200M]=1000000007
        [books-200M]=1000000007
        [fb-200M]=1000000003
        [wiki-ts-200M]=1000000013
        [cate_longlat-200M]=1000000007)

declare -A shifts
shifts=([longitudes-200M]=100000
        [longlat-200M]=1000000
        [ycsb-200M]=1000000
        [lognormal-200M]=1000000
        [books-200M]=1000000
        [fb-200M]=10000000
        [wiki-ts-200M]=1000000
        [cate_longlat-200M]=1000000)

output_dir=${output_root_dir}
if [ ! -d ${output_dir} ]; then
  mkdir ${output_dir}
fi
flow_para=${input_dim}'D-'${layer}'L-'${hidden_dim}'H-'${loss}
for workload in ${test_workloads[*]}
do
  dataset=${workload}'-80R-'${req_dist}
  (set -x; python3 numerical_flow.py --dataset=${dataset} --input_dir=${input_dir} \
                                      --output_dir=${output_dir} --seed=${seeds[$workload]} \
                                      --flows=${flows} --paras=${flow_para} --train_ratio=${train_ratio} \
                                      --de_type=${de_type} --all_pos=True --shifts=${shifts[$workload]})
  mv ${output_dir}/${dataset}_results.log ${output_dir}/${dataset}_results_${input_dim}D${hidden_dim}H${layer}L${flows}F${train_ratio}T.log
  mv ${output_dir}/${dataset}-weights.txt ${output_dir}/${dataset}_${input_dim}D${hidden_dim}H${layer}L_weights.txt
  cp ${output_dir}/${dataset}_${input_dim}D${hidden_dim}H${layer}L_weights.txt ${output_dir}/${workload}-100R-${req_dist}_${input_dim}D${hidden_dim}H${layer}L_weights.txt
  cp ${output_dir}/${dataset}_${input_dim}D${hidden_dim}H${layer}L_weights.txt ${output_dir}/${workload}-20R-${req_dist}_${input_dim}D${hidden_dim}H${layer}L_weights.txt
  cp ${output_dir}/${dataset}_${input_dim}D${hidden_dim}H${layer}L_weights.txt ${output_dir}/${workload}-0R-${req_dist}_${input_dim}D${hidden_dim}H${layer}L_weights.txt
done