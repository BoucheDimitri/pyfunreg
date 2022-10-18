#!/bin/bash

sshpass -p 7szbUPrP scp dbouche@tsi-cluster.enst.fr:/tsi/clusterhome/dbouche/pyfunreg/outputs.zip /home/dimitri/Desktop/Telecom/Latex/These/pyfunreg/outputs_cluster/"outputs_$(date +'%d-%m-%Y_%H-%M').zip"

