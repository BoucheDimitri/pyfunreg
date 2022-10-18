#!/bin/bash

sshpass -p 7szbUPrP ssh dbouche@tsi-cluster.enst.fr 'bash -s' < ./zip_outputs.sh
chmod +x $PWD/get_outputs.sh
$PWD/get_outputs.sh

