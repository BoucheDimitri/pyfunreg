import os
import sys


def create_run_file(expe_file_name, path):
    with open(path + '/run_expe.sh', 'w') as the_file:
        the_file.write('#!/bin/zsh\n')
        the_file.write('/tsi/clusterhome/dbouche/miniconda/bin/python /tsi/clusterhome/dbouche/pyfunreg/scripts/' + expe_file_name)


if __name__ == "__main__":
    create_run_file(sys.argv[1], os.getcwd())
