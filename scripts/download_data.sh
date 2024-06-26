#!/bin/bash

#TODO: Download our data

echo ">> Loading ligand files from PDBBind"
wget http://pdbbind.org.cn/download/PDBbind_v2020_sdf.tar.gz
tar -xf PDBbind_v2020_sdf.tar.gz -C data/
rm PDBbind_v2020_sdf.tar.gz

echo ">> Loading similarity file from UniProt"
cd data/annotations/
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/similar -O similar.txt
sed '1,16d' similar.txt > tmp
sed '202404,202408d' tmp > similar.txt
rm tmp
cd ../..

echo ">> Loaded all data"