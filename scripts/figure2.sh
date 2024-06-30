#!/bin/bash

base=$(pwd)
res_parent_dir="${base}/results/figure2"

# 2-450 clusters
k_range='[2'
for i in {3..450}
do
k_range="${k_range},${i}"
done 
k_range="${k_range}]"

# PCA on the full dataset
red_method="pca"
res_dir="${res_parent_dir}/all_${red_method}"

ARGUMENTS="++red_method=${red_method} ++reduction.n_dimensions=60 ++base_dir=${base} ++hydra.run.dir=${res_dir} ++clustering.agglomerative.n_clusters=${k_range} 
++hypotheses=[] ++plotting.family=[] ++plotting.superfamily=[] ++plotting.subfamily=[] ++testing.plot=false ++reload_families=true"

python pipeline/run_pipeline.py $ARGUMENTS


# Autoencoder on the full dataset
red_method="autoencoder"
res_dir="${res_parent_dir}/all_${red_method}"

ARGUMENTS="++red_method=${red_method} ++base_dir=${base} ++hydra.run.dir=${res_dir} ++clustering.agglomerative.n_clusters=${k_range} 
++hypotheses=[] ++plotting.family=[] ++plotting.superfamily=[] ++plotting.subfamily=[] ++testing.plot=false"

python pipeline/run_pipeline.py $ARGUMENTS


# PCA on the top 10 dataset
res_dir="${res_parent_dir}/top10_pca"

ARGUMENTS="++base_dir=${base} ++hydra.run.dir=${res_dir} ++red_method=pca ++subset.rank=10 ++clustering.agglomerative.n_clusters=${k_range} 
++hypotheses=[] ++plotting.family=[] ++plotting.superfamily=[] ++plotting.subfamily=[] ++testing.plot=false"

python pipeline/run_pipeline.py $ARGUMENTS