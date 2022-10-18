import os
import zipfile
from datetime import datetime

path = "/home/dimitri/Desktop/Telecom/Latex/These/code/outputs_cluster"


def find_latest(zip_files, prefix="outputs_"):
    raw_dates = [file.replace(prefix, "").replace(".zip", "") for file in zip_files]
    dates = [datetime.strptime(date, '%d-%m-%Y_%H-%M') for date in raw_dates]
    dates.sort()
    last_date = datetime.strftime(dates[-1], '%d-%m-%Y_%H-%M')
    return prefix + last_date


def path_to_latest(path=path, prefix="outputs_"):
    zip_files = [file for file in os.listdir(path) if file.endswith("zip")]
    latest = find_latest(zip_files, prefix)
    folders = [folder for folder in os.listdir(path) if "." not in folder]
    if latest not in folders:
        os.mkdir(path + "/" + latest)
        with zipfile.ZipFile(path + "/" + latest + ".zip", "r") as zip_ref:
            zip_ref.extractall(path + "/" + latest)
    return path + "/" + latest


