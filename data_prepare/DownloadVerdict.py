import urllib.request
import os

from data_prepare.Download import Download

class DownloadVerdict(Download):

    def __init__(self, folder_to_save: str) -> None:
        super().__init__(folder_to_save)
        self.url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
        self.file_path = os.path.join(self.folder_to_save, "the-verdict.txt")

    def download_dataset(self):
        urllib.request.urlretrieve(self.url, self.file_path)