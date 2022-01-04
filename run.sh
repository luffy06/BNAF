#!/bin/bash

test=true
input_dir='inputs'
if [ ${test} = true ];then
  output_dir='outputs/test_results/'
else
  output_dir='outputs/results/'
fi
req_dist='zipf'
read_frac_list=(100 80 20 0)
test_workloads=('longitudes-200M' 'longlat-200M' 'ycsb-200M' 'lognormal-200M'
                'books-200M' 'fb-200M' 'wiki-ts-200M')

declare -A flow_para
flow_para=([lognormal-190M-var(1)]='1S-2D-2L-1H-normal' 
            [lognormal-190M-var(2)]='1S-2D-2L-1H-normal'
            [lognormal-190M-var(4)]='1S-2D-2L-1H-normal'
            [lognormal-190M-var(8)]='1S-2D-2L-1H-normal'
            [uniform-190M]='1S-2D-2L-1H-normal'
            [longitudes-200M]='1S-2D-2L-1H-normal'
            [longlat-200M]='1S-2D-2L-1H-normal'
            [ycsb-200M]='1S-2D-2L-1H-normal'
            [books-200M]='1S-2D-2L-1H-normal'
            [fb-200M]='1S-2D-2L-2H-normal'
            [osm-cellids-200M]='1S-2D-2L-1H-normal'
            [wiki-ts-200M]='1S-2D-2L-1H-normal'
            [lognormal-200M]='1S-2D-2L-1H-normal')

declare -A seeds
seeds=([longitudes-200M]=1000000013
        [longlat-200M]=1000000007
        [ycsb-200M]=1000000013
        [lognormal-200M]=1000000007
        [books-200M]=1000000007
        [fb-200M]=1000000003
        [wiki-ts-200M]=1000000013
        [lognormal-190M-var(1)]=1000000013
        [lognormal-190M-var(2)]=1000000013
        [lognormal-190M-var(4)]=1000000013
        [lognormal-190M-var(8)]=1000000013
        [uniform-190M]=1000000013
        [osm-cellids-200M]=1000000013)

declare -A normalize
normalize=([longitudes-200M]=100000
        [longlat-200M]=1000000
        [ycsb-200M]=1000000
        [lognormal-200M]=1000000
        [books-200M]=1000000
        [fb-200M]=10000000
        [wiki-ts-200M]=1000000
        [lognormal-190M-var(1)]=1000000
        [lognormal-190M-var(2)]=1000000
        [lognormal-190M-var(4)]=1000000
        [lognormal-190M-var(8)]=1000000
        [uniform-190M]=1000000
        [osm-cellids-200M]=1000000)

for workload in ${test_workloads[*]}
do
  for read_frac in ${read_frac_list[*]}
  do
    dataset=${workload}'-'${read_frac}'R-'${req_dist}
    echo 'Process '${dataset}
    python3 numerical_flow.py --dataset=${dataset} --input_dir=${input_dir} --output_dir=${output_dir} --seed=${seeds[$workload]} --normalize=${normalize[$workload]} --paras=${flow_para[$workload]}
    if [ ${test} = true ];then
      break
    fi
  done
done