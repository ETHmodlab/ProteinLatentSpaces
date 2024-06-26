# Protein binding site representation in latent space

This repository contains a pipeline for end-to-end analysis of protein latent representations learned by deep learning models.

## Setup
### Environment
This code was written in ```Python 3.8.13```. We recommend using a conda environment and installing the required packages as follows:
```bash
conda create -n protein_env python=3.8.13
conda activate protein_env
pip install -r requirements.txt
```

### Downloading the data
Download the data by running
```bash
bash scripts/download_data.sh
```

## Running the Pipeline
A tutorial on how to prepare your data and run the pipeline can be found in ```tutorial.ipynb```.


## Reproducing the results
The code for reproducing the figures from the paper can be found in ```figures.ipynb```