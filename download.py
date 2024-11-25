import argparse
from utils import DATASETS, TYPE_DATA


def parse_args():
    parser = argparse.ArgumentParser(description='Download data')
    parser.add_argument('--name', type=str, choices=DATASETS, help='Name of the dataset')
    choices = ['pretrain', 'instructions']
    parser.add_argument('--type', type=str, default='pretrain', choices=choices, help='Type of data')
    return parser.parse_args()

# main
if __name__ == '__main__':
    args = parse_args()
    folder_to_save = TYPE_DATA[args.type]
    dataset = DATASETS[args.name](folder_to_save)
    dataset.download_dataset()