#!/bin/bash

# Create run script for cluster
python ./cluster_script_generator.py $1

# Push script to cluster
sshpass -p 7szbUPrP scp $PWD/run_expe.sh dbouche@tsi-cluster.enst.fr:/tsi/clusterhome/dbouche/ExpeScripts

# Remove script 
rm run_expe.sh

# Launch script on cluster
sshpass -p 7szbUPrP ssh dbouche@tsi-cluster.enst.fr 'cd pyfunreg; git pull origin master; cd ..; cd ExpeScripts; qsub run_expe.sh'
