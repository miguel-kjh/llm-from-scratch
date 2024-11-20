import os
import re

class Tokenizer:

    def encode(self):
        pass

class SimpleTokenizer(Tokenizer):

    def __init__(self, text):
        self.text = text
        self.regular_expression = r'([,.:;?_!"()\']|--|\s)'

    def encode(self):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', self.text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return preprocessed



# main
if __name__ == '__main__':
    FOLDER = "/home/miguel/data/01-raw/pretrain"
    FILE_PATH = os.path.join(FOLDER, "the-verdict.txt")

    with open(FILE_PATH, "r") as f:
        text = f.read()

    print("Total characters:", len(text))
    print(text[:99])

    tokenizer = SimpleTokenizer(text)
    tokens = tokenizer.encode()
    print("Total tokens:", len(tokens))
    print(tokens[:10])
