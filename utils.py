import os
from data_prepare.DownloadVerdict import DownloadVerdict

# Folders
DATA_FOLDER           = "data"
RAW_DATA_FOLDER       = os.path.join(DATA_FOLDER, "01-raw")
INSTRUCTIONS_FOLDER   = os.path.join(RAW_DATA_FOLDER, "instructions")
PETRAIN_DATA_FOLDER   = os.path.join(RAW_DATA_FOLDER, "pretrain")
PROCESS_DATA_FOLDER   = os.path.join(DATA_FOLDER, "02-processed")
COMBINED_DATA_FOLDER  = os.path.join(DATA_FOLDER, "03-combined")

DATASETS = {
    "verdict": DownloadVerdict
}

TYPE_DATA = {
    "pretrain": PETRAIN_DATA_FOLDER,
    "instructions": INSTRUCTIONS_FOLDER
}

SEED = 3407

# main
if __name__ == '__main__':
    # create folders
    os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
    os.makedirs(PROCESS_DATA_FOLDER, exist_ok=True)
    os.makedirs(INSTRUCTIONS_FOLDER, exist_ok=True)
    os.makedirs(PETRAIN_DATA_FOLDER, exist_ok=True)
    os.makedirs(COMBINED_DATA_FOLDER, exist_ok=True)