import urllib.request
import os

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
folder = "/home/miguel/data/01-raw/pretrain"
file_path = os.path.join(folder, "the-verdict.txt")
urllib.request.urlretrieve(url, file_path)