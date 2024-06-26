#!/bin/bash

base=$(pwd)
res_dir="${base}/results/figure4"

ARGUMENTS="++red_method=autoencoder ++base_dir=${base} ++hydra.run.dir=${res_dir} ++clustering.agglomerative.n_clusters=[7] 
 ++testing.plot=false"

python pipeline/run_pipeline.py $ARGUMENTS "++hypotheses=['alpha-carbonic anhydrase','CMGC Ser/Thr protein kinase','G-protein coupled receptor 1']"